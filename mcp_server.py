#!/usr/bin/env python3
"""NLLW MCP Server — gives Claude Code direct access to the SimulMT backend.

Exposes translation, evaluation, and benchmarking tools so Claude Code can
autonomously test translations, compare backends, sweep parameters, and
iterate on quality.

Usage:
    # In Claude Code settings, add as MCP server:
    python /path/to/mcp_server.py --url http://localhost:8778

    # Or with direct URL:
    python mcp_server.py --url http://quest.ms.mff.cuni.cz:8777
"""

import argparse
import json
import sys
import time

import requests

# MCP protocol over stdio
def send_response(id, result):
    msg = {"jsonrpc": "2.0", "id": id, "result": result}
    out = json.dumps(msg)
    sys.stdout.write(f"Content-Length: {len(out)}\r\n\r\n{out}")
    sys.stdout.flush()

def send_error(id, code, message):
    msg = {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}
    out = json.dumps(msg)
    sys.stdout.write(f"Content-Length: {len(out)}\r\n\r\n{out}")
    sys.stdout.flush()

# Global config
BASE_URL = "http://localhost:8778"

def api_post(endpoint, body=None):
    r = requests.post(f"{BASE_URL}{endpoint}", json=body or {}, timeout=120)
    return r.json()

def api_get(endpoint):
    r = requests.get(f"{BASE_URL}{endpoint}", timeout=30)
    return r.json()


# --- Tool implementations ---

def tool_translate_sentence(args):
    """Translate a full sentence word by word and return the result."""
    sentence = args["sentence"]
    source_lang = args.get("source_lang", "en")
    target_lang = args.get("target_lang", "fr")
    backend_type = args.get("backend_type", "alignatt")
    border_distance = args.get("border_distance", 3)
    word_batch = args.get("word_batch", 3)
    prompt_format = args.get("prompt_format", "hymt")

    # Load if needed
    status = api_get("/status")
    if not status.get("loaded") or status.get("backend_type") != backend_type:
        api_post("/load", {
            "target_lang": target_lang,
            "source_lang": source_lang,
            "backend_type": backend_type,
            "border_distance": border_distance,
            "word_batch": word_batch,
            "prompt_format": prompt_format,
        })
    else:
        api_post("/reset")

    words = sentence.strip().split()
    steps = []
    for w in words:
        r = api_post("/translate", {"text": w + " "})
        steps.append({
            "word": w,
            "stable": r.get("stable", ""),
            "buffer": r.get("buffer", ""),
            "committed_tokens": r.get("committed_tokens", 0),
            "time_ms": r.get("time_ms", 0),
        })

    fin = api_post("/finish")
    full = fin.get("full_translation", "")

    committed_text = ""
    for s in steps:
        if s["stable"].strip():
            committed_text += s["stable"]

    return {
        "source": sentence,
        "translation": full,
        "committed_before_finish": steps[-1]["committed_tokens"] if steps else 0,
        "finish_remaining": fin.get("remaining", ""),
        "steps": steps,
    }


def tool_compare_backends(args):
    """Compare the same sentence across multiple backend configs."""
    sentence = args["sentence"]
    configs = args.get("configs", [
        {"backend_type": "alignatt", "border_distance": 3, "word_batch": 3},
        {"backend_type": "alignatt-la", "border_distance": 3, "word_batch": 3},
        {"backend_type": "alignatt-kv", "border_distance": 3, "word_batch": 3},
    ])
    source_lang = args.get("source_lang", "en")
    target_lang = args.get("target_lang", "fr")

    results = []
    for cfg in configs:
        cfg.setdefault("source_lang", source_lang)
        cfg.setdefault("target_lang", target_lang)
        api_post("/reset")
        api_post("/load", cfg)

        words = sentence.strip().split()
        t0 = time.time()
        for w in words:
            api_post("/translate", {"text": w + " "})
        fin = api_post("/finish")
        elapsed = (time.time() - t0) * 1000

        results.append({
            "config": cfg,
            "translation": fin.get("full_translation", ""),
            "time_ms": round(elapsed, 1),
        })

    return {"sentence": sentence, "results": results}


def tool_batch_translate(args):
    """Translate multiple sentences and return all results."""
    sentences = args["sentences"]
    source_lang = args.get("source_lang", "en")
    target_lang = args.get("target_lang", "fr")

    # Ensure loaded
    status = api_get("/status")
    if not status.get("loaded"):
        api_post("/load", {"target_lang": target_lang, "source_lang": source_lang})

    results = []
    for sent in sentences:
        api_post("/reset")
        for w in sent.strip().split():
            api_post("/translate", {"text": w + " "})
        fin = api_post("/finish")
        results.append({
            "source": sent,
            "translation": fin.get("full_translation", ""),
        })

    return {"results": results}


def tool_parameter_sweep(args):
    """Sweep parameters and show quality for each config."""
    sentence = args["sentence"]
    reference = args.get("reference", "")
    border_distances = args.get("border_distances", [2, 3, 4])
    word_batches = args.get("word_batches", [2, 3])
    source_lang = args.get("source_lang", "en")
    target_lang = args.get("target_lang", "fr")

    results = []
    for bd in border_distances:
        for wb in word_batches:
            api_post("/reset")
            api_post("/load", {
                "target_lang": target_lang,
                "source_lang": source_lang,
                "border_distance": bd,
                "word_batch": wb,
            })
            t0 = time.time()
            for w in sentence.strip().split():
                api_post("/translate", {"text": w + " "})
            fin = api_post("/finish")
            elapsed = (time.time() - t0) * 1000

            status = api_get("/status")
            results.append({
                "border_distance": bd,
                "word_batch": wb,
                "translation": fin.get("full_translation", ""),
                "time_ms": round(elapsed, 1),
                "committed_tokens": status.get("committed_tokens", 0),
            })

    return {"sentence": sentence, "reference": reference, "results": results}


def tool_get_status(args):
    """Get the current server status."""
    return api_get("/status")


def tool_load_model(args):
    """Load a model with specific configuration."""
    return api_post("/load", args)


def tool_list_backends(args):
    """List available backends."""
    return api_get("/backends")


def tool_list_heads(args):
    """List available head config files."""
    return api_get("/heads")


TOOLS = {
    "translate_sentence": {
        "description": "Translate a sentence word-by-word through the SimulMT backend on the A40 GPU. Returns the full translation, per-word steps, and timing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sentence": {"type": "string", "description": "The sentence to translate"},
                "source_lang": {"type": "string", "default": "en"},
                "target_lang": {"type": "string", "default": "fr"},
                "backend_type": {"type": "string", "default": "alignatt", "enum": ["alignatt", "alignatt-la", "alignatt-kv", "wait-k", "full-sentence", "eager"]},
                "border_distance": {"type": "integer", "default": 3},
                "word_batch": {"type": "integer", "default": 3},
                "prompt_format": {"type": "string", "default": "hymt"},
            },
            "required": ["sentence"],
        },
        "handler": tool_translate_sentence,
    },
    "compare_backends": {
        "description": "Compare the same sentence across multiple backend configurations. Great for A/B testing alignatt vs alignatt-la vs alignatt-kv.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sentence": {"type": "string"},
                "configs": {"type": "array", "items": {"type": "object"}},
                "source_lang": {"type": "string", "default": "en"},
                "target_lang": {"type": "string", "default": "fr"},
            },
            "required": ["sentence"],
        },
        "handler": tool_compare_backends,
    },
    "batch_translate": {
        "description": "Translate multiple sentences at once. Returns all translations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sentences": {"type": "array", "items": {"type": "string"}},
                "source_lang": {"type": "string", "default": "en"},
                "target_lang": {"type": "string", "default": "fr"},
            },
            "required": ["sentences"],
        },
        "handler": tool_batch_translate,
    },
    "parameter_sweep": {
        "description": "Sweep border_distance and word_batch parameters for a sentence. Shows how each config affects translation quality and speed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sentence": {"type": "string"},
                "reference": {"type": "string", "description": "Optional reference translation"},
                "border_distances": {"type": "array", "items": {"type": "integer"}, "default": [2, 3, 4]},
                "word_batches": {"type": "array", "items": {"type": "integer"}, "default": [2, 3]},
                "source_lang": {"type": "string", "default": "en"},
                "target_lang": {"type": "string", "default": "fr"},
            },
            "required": ["sentence"],
        },
        "handler": tool_parameter_sweep,
    },
    "get_status": {
        "description": "Get the current NLLW server status — loaded model, backend type, parameters.",
        "inputSchema": {"type": "object", "properties": {}},
        "handler": tool_get_status,
    },
    "load_model": {
        "description": "Load a model with specific configuration (backend type, border_distance, word_batch, prompt_format, lora_path, etc.)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "backend_type": {"type": "string", "default": "alignatt"},
                "target_lang": {"type": "string", "default": "fr"},
                "source_lang": {"type": "string", "default": "en"},
                "border_distance": {"type": "integer", "default": 3},
                "word_batch": {"type": "integer", "default": 3},
                "prompt_format": {"type": "string", "default": "hymt"},
                "lora_path": {"type": "string"},
            },
        },
        "handler": tool_load_model,
    },
    "list_backends": {
        "description": "List available translation backends.",
        "inputSchema": {"type": "object", "properties": {}},
        "handler": tool_list_backends,
    },
    "list_heads": {
        "description": "List available alignment head configuration files.",
        "inputSchema": {"type": "object", "properties": {}},
        "handler": tool_list_heads,
    },
}


def handle_message(msg):
    method = msg.get("method")
    id = msg.get("id")
    params = msg.get("params", {})

    if method == "initialize":
        send_response(id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "nllw-simulmt", "version": "0.1.0"},
        })
    elif method == "notifications/initialized":
        pass  # no response needed
    elif method == "tools/list":
        tools_list = []
        for name, spec in TOOLS.items():
            tools_list.append({
                "name": name,
                "description": spec["description"],
                "inputSchema": spec["inputSchema"],
            })
        send_response(id, {"tools": tools_list})
    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        if tool_name in TOOLS:
            try:
                result = TOOLS[tool_name]["handler"](tool_args)
                send_response(id, {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}],
                })
            except Exception as e:
                send_response(id, {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "isError": True,
                })
        else:
            send_error(id, -32601, f"Unknown tool: {tool_name}")
    else:
        if id is not None:
            send_error(id, -32601, f"Method not found: {method}")


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(description="NLLW MCP Server")
    parser.add_argument("--url", default="http://localhost:8778",
                        help="NLLW FastAPI server URL")
    args = parser.parse_args()
    BASE_URL = args.url

    # Read JSON-RPC messages from stdin
    while True:
        try:
            # Read Content-Length header
            line = sys.stdin.readline()
            if not line:
                break
            if line.startswith("Content-Length:"):
                content_length = int(line.split(":")[1].strip())
                sys.stdin.readline()  # empty line
                body = sys.stdin.read(content_length)
                msg = json.loads(body)
                handle_message(msg)
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            sys.stderr.write(f"MCP error: {e}\n")


if __name__ == "__main__":
    main()
