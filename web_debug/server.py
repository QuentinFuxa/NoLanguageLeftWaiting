"""FastAPI debug server for interactive translation testing and evaluation.

Supports multiple backends (alignatt, alignatt-la, alignatt-kv), side-by-side
comparison, and corpus evaluation via the NLLW evaluator.

Run with:
    python web_debug/server.py

Or:
    cd web_debug && uvicorn server:app --port 8777 --reload
"""

import os
import sys
from typing import List, Optional

# Ensure the project root is on sys.path so `nllw` can be imported
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend_wrapper import AVAILABLE_BACKENDS, TranslationService

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="NLLW Debug Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singleton translation service
# ---------------------------------------------------------------------------

service = TranslationService.get_instance()

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class LoadRequest(BaseModel):
    model_path: Optional[str] = None
    target_lang: str = "fr"
    source_lang: str = "en"
    backend_type: str = "alignatt"
    heads_path: Optional[str] = None
    prompt_format: str = "hymt"
    custom_template: Optional[str] = None
    border_distance: int = 3
    word_batch: int = 3
    entropy_veto_threshold: Optional[float] = None
    lora_path: Optional[str] = None
    lora_scale: float = 1.0


class TranslateRequest(BaseModel):
    text: str


class SetLangRequest(BaseModel):
    lang: str


class CompareConfig(BaseModel):
    backend_type: Optional[str] = None
    border_distance: Optional[int] = None
    word_batch: Optional[int] = None
    entropy_veto_threshold: Optional[float] = None
    heads_path: Optional[str] = None


class CompareRequest(BaseModel):
    text: str
    configs: List[CompareConfig]


class EvalTestCase(BaseModel):
    source: str
    reference: str
    source_lang: str = "en"
    target_lang: str = "fr"
    tag: Optional[str] = None


class EvaluateRequest(BaseModel):
    test_cases: List[EvalTestCase]
    backend_type: Optional[str] = None
    params: Optional[dict] = None


# ---------------------------------------------------------------------------
# Endpoints — static
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_HEADS_DIR = os.path.join(_PROJECT_ROOT, "nllw", "heads")


@app.get("/")
async def index():
    """Serve the debug frontend."""
    return FileResponse(os.path.join(_HERE, "index.html"))


# ---------------------------------------------------------------------------
# Endpoints — discovery
# ---------------------------------------------------------------------------


@app.get("/backends")
async def list_backends():
    """Return the list of available backend types."""
    return {"backends": list(AVAILABLE_BACKENDS)}


@app.get("/heads")
async def list_heads():
    """Return available alignment head config files from nllw/heads/."""
    heads: list[dict] = []
    if os.path.isdir(_HEADS_DIR):
        for fname in sorted(os.listdir(_HEADS_DIR)):
            if fname.endswith(".json"):
                heads.append({
                    "filename": fname,
                    "path": os.path.join(_HEADS_DIR, fname),
                })
    return {"heads": heads}


@app.get("/prompt_formats")
async def list_prompt_formats():
    """Return available prompt format names for AlignAtt backends."""
    from nllw.alignatt_backend import PROMPT_FORMATS
    return {"prompt_formats": list(PROMPT_FORMATS.keys())}


# ---------------------------------------------------------------------------
# Endpoints — lifecycle
# ---------------------------------------------------------------------------


@app.post("/load")
async def load_model(req: LoadRequest):
    """Load a translation backend."""
    result = service.load(
        model_path=req.model_path,
        target_lang=req.target_lang,
        source_lang=req.source_lang,
        backend_type=req.backend_type,
        heads_path=req.heads_path,
        prompt_format=req.prompt_format,
        custom_template=req.custom_template,
        border_distance=req.border_distance,
        word_batch=req.word_batch,
        entropy_veto_threshold=req.entropy_veto_threshold,
        lora_path=req.lora_path,
        lora_scale=req.lora_scale,
    )
    return result


@app.post("/set_lang")
async def set_lang(req: SetLangRequest):
    """Change target language."""
    return service.set_target_lang(req.lang)


@app.post("/reset")
async def reset():
    """Reset translation state."""
    service.reset()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Endpoints — translation
# ---------------------------------------------------------------------------


@app.post("/translate")
async def translate(req: TranslateRequest):
    """Translate an incremental text chunk.

    Response includes timing information::

        {
            "stable": str,
            "buffer": str,
            "source_words": list[str],
            "committed_tokens": int,
            "time_ms": float
        }
    """
    return service.translate(req.text)


@app.post("/finish")
async def finish():
    """Flush remaining translation.

    Response::

        {
            "remaining": str,
            "full_translation": str,
            "time_ms": float
        }
    """
    return service.finish()


# ---------------------------------------------------------------------------
# Endpoints — comparison
# ---------------------------------------------------------------------------


@app.post("/compare")
async def compare(req: CompareRequest):
    """Run the same text through multiple backend configurations.

    Request body::

        {
            "text": "The weather is nice today",
            "configs": [
                {"backend_type": "alignatt", "border_distance": 3},
                {"backend_type": "alignatt", "border_distance": 5},
                {"backend_type": "alignatt-la", "word_batch": 2}
            ]
        }

    Returns a list of result dicts, one per config.
    """
    configs = [cfg.model_dump(exclude_none=True) for cfg in req.configs]
    results = service.compare(req.text, configs)
    return {"text": req.text, "results": results}


# ---------------------------------------------------------------------------
# Endpoints — evaluation
# ---------------------------------------------------------------------------


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    """Run the SimulMTEvaluator on provided test cases.

    Request body::

        {
            "test_cases": [
                {"source": "...", "reference": "...", "source_lang": "en", "target_lang": "fr"}
            ],
            "backend_type": "alignatt",
            "params": {"border_distance": 3, "word_batch": 2}
        }

    Returns BLEU, committed_ratio, timing, and per-sentence details.
    """
    cases = [tc.model_dump() for tc in req.test_cases]
    results = service.evaluate(
        test_cases=cases,
        backend_type=req.backend_type,
        params=req.params,
    )
    return results


# ---------------------------------------------------------------------------
# Endpoints — introspection
# ---------------------------------------------------------------------------


@app.get("/status")
async def status():
    """Return current service status.

    Response::

        {
            "loaded": bool,
            "source_lang": str,
            "target_lang": str,
            "backend_type": str,
            "border_distance": int,
            "word_batch": int,
            "entropy_veto_threshold": float | null,
            "heads_file": str,
            "source_words": list[str],
            "committed_tokens": int
        }
    """
    return service.get_status()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8777)
