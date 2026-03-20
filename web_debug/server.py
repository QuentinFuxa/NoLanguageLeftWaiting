"""FastAPI debug server for interactive SimulMT testing.

Wraps the NLLW SimulMT backends in an HTTP API for interactive testing,
side-by-side comparison, and real-time visualization of the translation
process (word-by-word emission, border detection, SSBD stats).

Run with:
    HYMT_MODEL_PATH=/path/to/model.gguf python web_debug/server.py

Or:
    uvicorn web_debug.server:app --port 8777 --reload
"""

import os
import sys
import time
from typing import List, Optional, Dict, Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from nllw.backend_protocol import BackendConfig, create_backend, list_backends

# Register all backends
import nllw.alignatt_backend  # noqa: F401
import nllw.alignatt_la_backend  # noqa: F401
import nllw.baselines  # noqa: F401

app = FastAPI(title="NLLW SimulMT Debug Server", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_backend = None
_config: Optional[BackendConfig] = None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LoadRequest(BaseModel):
    model_path: Optional[str] = None
    backend_type: str = "alignatt"
    direction: str = "en-fr"
    border_distance: int = 3
    word_batch: int = 3
    context_sentences: int = 0
    entropy_veto_threshold: Optional[float] = None
    aggregation: str = "ts_vote"
    dynamic_border: bool = False
    ssbd_beta: Optional[float] = None
    heads_path: Optional[str] = None
    n_ctx: int = 2048


class TranslateWordRequest(BaseModel):
    word: str
    is_final: bool = False


class TranslateFullRequest(BaseModel):
    text: str
    word_by_word: bool = True


class CompareRequest(BaseModel):
    text: str
    configs: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve minimal debug UI."""
    return _FRONTEND_HTML


@app.get("/api/backends")
async def get_backends():
    return {"backends": list_backends()}


@app.get("/api/heads")
async def get_heads():
    heads_dir = os.path.join(_PROJECT_ROOT, "nllw", "heads", "configs")
    files = []
    if os.path.isdir(heads_dir):
        for f in sorted(os.listdir(heads_dir)):
            if f.endswith(".json"):
                files.append({"name": f, "path": os.path.join(heads_dir, f)})
    return {"heads": files}


@app.get("/api/status")
async def get_status():
    if _backend is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "backend_type": _config.backend_type,
        "direction": _config.direction,
        "border_distance": _config.border_distance,
        "word_batch": _config.word_batch,
        "aggregation": _config.aggregation,
        "dynamic_border": _config.dynamic_border,
        "ssbd_beta": _config.ssbd_beta,
    }


@app.post("/api/load")
async def load_model(req: LoadRequest):
    global _backend, _config

    model_path = req.model_path or os.environ.get("HYMT_MODEL_PATH")
    if not model_path:
        return {"ok": False, "error": "No model_path. Set HYMT_MODEL_PATH env var."}

    parts = req.direction.split("-")
    target_lang = parts[1] if len(parts) > 1 else "fr"

    try:
        if _backend is not None:
            _backend.close()

        _config = BackendConfig(
            backend_type=req.backend_type,
            model_path=model_path,
            heads_path=req.heads_path or "",
            direction=req.direction,
            border_distance=req.border_distance,
            word_batch=req.word_batch,
            context_sentences=req.context_sentences,
            entropy_veto_threshold=req.entropy_veto_threshold,
            aggregation=req.aggregation,
            dynamic_border=req.dynamic_border,
            ssbd_beta=req.ssbd_beta,
            target_lang=target_lang,
            n_ctx=req.n_ctx,
        )
        _backend = create_backend(_config)
        return {"ok": True, "config": _config.__dict__}
    except Exception as e:
        _backend = None
        return {"ok": False, "error": str(e)}


@app.post("/api/translate/word")
async def translate_word(req: TranslateWordRequest):
    if _backend is None:
        return {"error": "Backend not loaded"}

    step = _backend.translate(req.word, is_final=req.is_final)
    result = {
        "text": step.text,
        "is_final": step.is_final,
        "committed_tokens": step.committed_tokens,
        "stopped_at_border": step.stopped_at_border,
        "source_words_seen": step.source_words_seen,
        "generation_time_ms": round(step.generation_time_ms, 2),
    }

    # Add SSBD stats if available
    if hasattr(_backend, "get_ssbd_stats"):
        result["ssbd_stats"] = _backend.get_ssbd_stats()

    return result


@app.post("/api/translate/full")
async def translate_full(req: TranslateFullRequest):
    if _backend is None:
        return {"error": "Backend not loaded"}

    words = req.text.strip().split()
    if not words:
        return {"error": "Empty text"}

    _backend.reset()
    steps = []
    full_text = ""
    t0 = time.perf_counter()

    for i, word in enumerate(words):
        is_final = (i == len(words) - 1)
        step = _backend.translate(word, is_final=is_final)
        full_text += step.text
        steps.append({
            "word": word,
            "emitted": step.text,
            "stopped_at_border": step.stopped_at_border,
            "time_ms": round(step.generation_time_ms, 2),
        })

    total_ms = (time.perf_counter() - t0) * 1000.0

    result = {
        "full_translation": full_text.strip(),
        "steps": steps,
        "total_time_ms": round(total_ms, 2),
        "n_words": len(words),
    }

    # Add SSBD and NE stats if available
    if hasattr(_backend, "get_ssbd_stats"):
        result["ssbd_stats"] = _backend.get_ssbd_stats()
    if hasattr(_backend, "get_revision_history"):
        history = _backend.get_revision_history()
        if len(history) >= 2:
            from nllw.metrics import compute_normalized_erasure
            result["normalized_erasure"] = round(
                compute_normalized_erasure(history), 3
            )

    return result


@app.post("/api/translate/compare")
async def translate_compare(req: CompareRequest):
    words = req.text.strip().split()
    if not words:
        return {"error": "Empty text"}

    model_path = os.environ.get("HYMT_MODEL_PATH", "")
    results = []

    for cfg_dict in req.configs:
        cfg_dict.setdefault("model_path", model_path)
        if "direction" not in cfg_dict and _config:
            cfg_dict["direction"] = _config.direction
        parts = cfg_dict.get("direction", "en-fr").split("-")
        cfg_dict.setdefault("target_lang", parts[1] if len(parts) > 1 else "fr")

        try:
            config = BackendConfig.from_dict(cfg_dict)
            backend = create_backend(config)

            steps = []
            full_text = ""
            t0 = time.perf_counter()

            for i, word in enumerate(words):
                is_final = (i == len(words) - 1)
                step = backend.translate(word, is_final=is_final)
                full_text += step.text
                steps.append({
                    "word": word,
                    "emitted": step.text,
                    "stopped_at_border": step.stopped_at_border,
                    "time_ms": round(step.generation_time_ms, 2),
                })

            total_ms = (time.perf_counter() - t0) * 1000.0
            entry = {
                "config": cfg_dict,
                "full_translation": full_text.strip(),
                "steps": steps,
                "total_time_ms": round(total_ms, 2),
            }

            if hasattr(backend, "get_ssbd_stats"):
                entry["ssbd_stats"] = backend.get_ssbd_stats()

            backend.close()
            results.append(entry)
        except Exception as e:
            results.append({"config": cfg_dict, "error": str(e)})

    return {"text": req.text, "results": results}


@app.post("/api/reset")
async def reset():
    if _backend is not None:
        _backend.reset()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Minimal frontend
# ---------------------------------------------------------------------------

_FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NLLW SimulMT Debug</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; max-width: 1200px; margin: auto; }
h1 { font-size: 1.5rem; margin-bottom: 1.5rem; color: #38bdf8; }
.panel { background: #1e293b; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }
.panel h2 { font-size: 1rem; color: #94a3b8; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }
label { display: block; font-size: 0.85rem; color: #94a3b8; margin: 0.5rem 0 0.2rem; }
input, select, textarea { width: 100%; padding: 0.5rem; background: #0f172a; border: 1px solid #334155; border-radius: 4px; color: #e2e8f0; font-family: inherit; font-size: 0.9rem; }
textarea { resize: vertical; min-height: 80px; }
button { background: #2563eb; color: white; border: none; padding: 0.6rem 1.5rem; border-radius: 4px; cursor: pointer; font-size: 0.9rem; margin-top: 0.5rem; margin-right: 0.5rem; }
button:hover { background: #3b82f6; }
button:disabled { background: #334155; cursor: not-allowed; }
.row { display: flex; gap: 1rem; flex-wrap: wrap; }
.row > * { flex: 1; min-width: 200px; }
.steps { margin-top: 1rem; }
.step { display: flex; gap: 0.5rem; padding: 0.3rem 0; border-bottom: 1px solid #1e293b; font-size: 0.85rem; font-family: 'SF Mono', 'Cascadia Code', monospace; }
.step .src { color: #94a3b8; width: 120px; flex-shrink: 0; }
.step .emit { color: #4ade80; flex: 1; }
.step .border { color: #f97316; font-size: 0.75rem; }
.step .time { color: #64748b; width: 60px; text-align: right; flex-shrink: 0; }
.result { background: #0f172a; border-radius: 4px; padding: 1rem; margin-top: 1rem; font-family: 'SF Mono', monospace; }
.result .translation { color: #4ade80; font-size: 1.1rem; margin-bottom: 0.5rem; }
.result .stats { color: #64748b; font-size: 0.8rem; }
.status { font-size: 0.8rem; color: #64748b; }
.status.loaded { color: #4ade80; }
#status-bar { position: fixed; top: 0; left: 0; right: 0; padding: 0.5rem 2rem; background: #1e293b; z-index: 10; display: flex; justify-content: space-between; font-size: 0.8rem; }
body { padding-top: 3rem; }
</style>
</head>
<body>
<div id="status-bar">
  <span id="conn-status" class="status">Not loaded</span>
  <span id="backend-info" class="status"></span>
</div>

<h1>NLLW SimulMT Debug Server</h1>

<div class="panel">
  <h2>Load Backend</h2>
  <div class="row">
    <div><label>Backend</label><select id="backend-type"><option>alignatt</option><option>alignatt-la</option><option>full-sentence</option><option>eager</option><option>wait-k</option></select></div>
    <div><label>Direction</label><input id="direction" value="en-fr"></div>
    <div><label>Border Distance</label><input id="bd" type="number" value="3"></div>
    <div><label>Word Batch</label><input id="wb" type="number" value="3"></div>
  </div>
  <div class="row">
    <div><label>Aggregation</label><select id="agg"><option>ts_vote</option><option>softmax_mean</option><option>entropy_weighted</option><option>consensus</option><option>geomean</option><option>top_p</option><option>ensemble</option></select></div>
    <div><label>SSBD Beta (empty=off)</label><input id="ssbd" placeholder="0.2"></div>
    <div><label>Dynamic Border</label><select id="dynbd"><option value="false">Off</option><option value="true">On</option></select></div>
    <div><label>Entropy Veto</label><input id="entropy" placeholder="0.75"></div>
  </div>
  <button onclick="loadBackend()">Load</button>
</div>

<div class="panel">
  <h2>Translate</h2>
  <textarea id="source" placeholder="Enter source text...">The president of France announced new economic reforms yesterday during a press conference at the Elysee Palace.</textarea>
  <button onclick="translateFull()">Translate</button>
  <button onclick="resetBackend()">Reset</button>
  <div id="output"></div>
</div>

<script>
const API = '';

async function fetchStatus() {
  try {
    const r = await fetch(API + '/api/status');
    const d = await r.json();
    const el = document.getElementById('conn-status');
    const info = document.getElementById('backend-info');
    if (d.loaded) {
      el.textContent = 'Loaded';
      el.className = 'status loaded';
      info.textContent = `${d.backend_type} | ${d.direction} | bd=${d.border_distance} | wb=${d.word_batch} | agg=${d.aggregation}` + (d.ssbd_beta !== null ? ` | ssbd=${d.ssbd_beta}` : '');
    } else {
      el.textContent = 'Not loaded';
      el.className = 'status';
      info.textContent = '';
    }
  } catch(e) { document.getElementById('conn-status').textContent = 'Disconnected'; }
}

async function loadBackend() {
  const ssbd = document.getElementById('ssbd').value;
  const entropy = document.getElementById('entropy').value;
  const body = {
    backend_type: document.getElementById('backend-type').value,
    direction: document.getElementById('direction').value,
    border_distance: parseInt(document.getElementById('bd').value),
    word_batch: parseInt(document.getElementById('wb').value),
    aggregation: document.getElementById('agg').value,
    dynamic_border: document.getElementById('dynbd').value === 'true',
    ssbd_beta: ssbd ? parseFloat(ssbd) : null,
    entropy_veto_threshold: entropy ? parseFloat(entropy) : null,
  };
  const r = await fetch(API + '/api/load', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const d = await r.json();
  if (!d.ok) alert('Load failed: ' + d.error);
  fetchStatus();
}

async function translateFull() {
  const text = document.getElementById('source').value.trim();
  if (!text) return;
  const r = await fetch(API + '/api/translate/full', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})});
  const d = await r.json();
  if (d.error) { alert(d.error); return; }

  let html = '<div class="result"><div class="translation">' + escHtml(d.full_translation) + '</div>';
  html += '<div class="stats">' + d.n_words + ' words | ' + d.total_time_ms + ' ms';
  if (d.ssbd_stats) html += ' | SSBD: ' + d.ssbd_stats.accepted_tokens + '/' + d.ssbd_stats.draft_tokens + ' accepted (' + (d.ssbd_stats.acceptance_rate * 100).toFixed(0) + '%)';
  if (d.normalized_erasure !== undefined) html += ' | NE: ' + d.normalized_erasure;
  html += '</div></div>';

  html += '<div class="steps">';
  for (const s of d.steps) {
    html += '<div class="step">';
    html += '<span class="src">' + escHtml(s.word) + '</span>';
    html += '<span class="emit">' + (s.emitted ? escHtml(s.emitted) : '<span style="color:#334155">---</span>') + '</span>';
    if (s.stopped_at_border) html += '<span class="border">BORDER</span>';
    html += '<span class="time">' + s.time_ms + 'ms</span>';
    html += '</div>';
  }
  html += '</div>';
  document.getElementById('output').innerHTML = html;
}

async function resetBackend() {
  await fetch(API + '/api/reset', {method:'POST'});
  document.getElementById('output').innerHTML = '';
}

function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

fetchStatus();
setInterval(fetchStatus, 5000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8777)
