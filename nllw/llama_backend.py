"""Minimal ctypes wrapper for llama.cpp with attention weight extraction API.

This module provides Python bindings to a custom llama.cpp build that exposes
attention weights via llama_get_attn_ith(). Required for AlignAtt-based
simultaneous machine translation.

The library is found by searching (in order):
  1. LLAMA_CPP_LIB environment variable (explicit path)
  2. ../build/bin/ relative to this file
  3. Common system paths

Requires llama.cpp built with PR #20086 (attention weight extraction).
"""

import ctypes
import os
import sys
import subprocess
import tempfile
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------
_LIB_SEARCH_DIRS = [
    os.environ.get("LLAMA_CPP_LIB", ""),
    os.path.join(os.path.dirname(__file__), "..", "build", "bin"),
    "/usr/local/lib",
    "/usr/lib",
]
_LIB_NAMES = ["libllama.dylib", "libllama.so", "llama.dll"]

_lib = None
_lib_path = None

for search_dir in _LIB_SEARCH_DIRS:
    if not search_dir:
        continue
    # If it's a direct file path
    if os.path.isfile(search_dir):
        try:
            _lib = ctypes.CDLL(search_dir)
            _lib_path = search_dir
            break
        except OSError:
            continue
    # Search directory for library files
    for name in _LIB_NAMES:
        path = os.path.join(search_dir, name)
        if os.path.exists(path):
            try:
                _lib = ctypes.CDLL(path)
                _lib_path = path
                break
            except OSError:
                continue
    if _lib is not None:
        break


def _check_lib():
    """Raise if library not loaded."""
    if _lib is None:
        raise RuntimeError(
            "Cannot find libllama. Set LLAMA_CPP_LIB=/path/to/libllama.{dylib,so} "
            "or build llama.cpp with attention extraction support."
        )


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
class _llama_model(ctypes.Structure):
    pass

class _llama_context(ctypes.Structure):
    pass

class _llama_vocab(ctypes.Structure):
    pass

class _llama_memory_i(ctypes.Structure):
    pass

llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32
llama_memory_t = ctypes.POINTER(_llama_memory_i)


class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens",  ctypes.c_int32),
        ("token",     ctypes.POINTER(llama_token)),
        ("embd",      ctypes.POINTER(ctypes.c_float)),
        ("pos",       ctypes.POINTER(llama_pos)),
        ("n_seq_id",  ctypes.POINTER(ctypes.c_int32)),
        ("seq_id",    ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits",    ctypes.POINTER(ctypes.c_int8)),
    ]


class llama_model_params(ctypes.Structure):
    _fields_ = [("_opaque", ctypes.c_uint8 * 256)]


# ---------------------------------------------------------------------------
# Function signatures (bound lazily on first use)
# ---------------------------------------------------------------------------
_signatures_bound = False

def _bind_signatures():
    global _signatures_bound
    if _signatures_bound:
        return
    _check_lib()

    # Backend
    _lib.llama_backend_init.argtypes = []
    _lib.llama_backend_init.restype = None
    _lib.llama_backend_free.argtypes = []
    _lib.llama_backend_free.restype = None

    # Model
    _lib.llama_model_default_params.argtypes = []
    _lib.llama_model_default_params.restype = llama_model_params
    _lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
    _lib.llama_model_load_from_file.restype = ctypes.POINTER(_llama_model)
    _lib.llama_model_free.argtypes = [ctypes.POINTER(_llama_model)]
    _lib.llama_model_free.restype = None
    _lib.llama_model_get_vocab.argtypes = [ctypes.POINTER(_llama_model)]
    _lib.llama_model_get_vocab.restype = ctypes.POINTER(_llama_vocab)
    _lib.llama_model_n_layer.argtypes = [ctypes.POINTER(_llama_model)]
    _lib.llama_model_n_layer.restype = ctypes.c_int32
    _lib.llama_model_n_head.argtypes = [ctypes.POINTER(_llama_model)]
    _lib.llama_model_n_head.restype = ctypes.c_int32

    # Vocab
    _lib.llama_vocab_n_tokens.argtypes = [ctypes.POINTER(_llama_vocab)]
    _lib.llama_vocab_n_tokens.restype = ctypes.c_int32
    _lib.llama_vocab_bos.argtypes = [ctypes.POINTER(_llama_vocab)]
    _lib.llama_vocab_bos.restype = llama_token
    _lib.llama_vocab_eos.argtypes = [ctypes.POINTER(_llama_vocab)]
    _lib.llama_vocab_eos.restype = llama_token

    # Tokenize / detokenize
    _lib.llama_tokenize.argtypes = [
        ctypes.POINTER(_llama_vocab), ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(llama_token), ctypes.c_int32, ctypes.c_bool, ctypes.c_bool,
    ]
    _lib.llama_tokenize.restype = ctypes.c_int32
    _lib.llama_token_to_piece.argtypes = [
        ctypes.POINTER(_llama_vocab), llama_token,
        ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32, ctypes.c_bool,
    ]
    _lib.llama_token_to_piece.restype = ctypes.c_int32

    # Batch
    _lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    _lib.llama_batch_init.restype = llama_batch
    _lib.llama_batch_free.argtypes = [llama_batch]
    _lib.llama_batch_free.restype = None

    # Decode
    _lib.llama_decode.argtypes = [ctypes.POINTER(_llama_context), llama_batch]
    _lib.llama_decode.restype = ctypes.c_int32

    # Context
    _lib.llama_free.argtypes = [ctypes.POINTER(_llama_context)]
    _lib.llama_free.restype = None
    _lib.llama_n_ctx.argtypes = [ctypes.POINTER(_llama_context)]
    _lib.llama_n_ctx.restype = ctypes.c_uint32
    _lib.llama_synchronize.argtypes = [ctypes.POINTER(_llama_context)]
    _lib.llama_synchronize.restype = None

    # Attention weights (custom API from PR #20086)
    _lib.llama_set_attn_heads.argtypes = [
        ctypes.POINTER(_llama_context),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
    ]
    _lib.llama_set_attn_heads.restype = None
    _lib.llama_get_attn_ith.argtypes = [ctypes.POINTER(_llama_context), ctypes.c_int32]
    _lib.llama_get_attn_ith.restype = ctypes.POINTER(ctypes.c_float)
    _lib.llama_get_attn_n_kv.argtypes = [ctypes.POINTER(_llama_context)]
    _lib.llama_get_attn_n_kv.restype = ctypes.c_int32

    # Logits
    _lib.llama_get_logits_ith.argtypes = [ctypes.POINTER(_llama_context), ctypes.c_int32]
    _lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

    # KV cache / memory
    _lib.llama_get_memory.argtypes = [ctypes.POINTER(_llama_context)]
    _lib.llama_get_memory.restype = llama_memory_t
    _lib.llama_memory_clear.argtypes = [llama_memory_t, ctypes.c_bool]
    _lib.llama_memory_clear.restype = None
    _lib.llama_memory_seq_rm.argtypes = [llama_memory_t, llama_seq_id, llama_pos, llama_pos]
    _lib.llama_memory_seq_rm.restype = ctypes.c_bool
    _lib.llama_memory_seq_cp.argtypes = [llama_memory_t, llama_seq_id, llama_seq_id, llama_pos, llama_pos]
    _lib.llama_memory_seq_cp.restype = None
    _lib.llama_memory_seq_keep.argtypes = [llama_memory_t, llama_seq_id]
    _lib.llama_memory_seq_keep.restype = None
    _lib.llama_memory_seq_pos_max.argtypes = [llama_memory_t, llama_seq_id]
    _lib.llama_memory_seq_pos_max.restype = llama_pos

    # Optional: hybrid model support
    global _has_seq_rm_attn_only
    _has_seq_rm_attn_only = hasattr(_lib, 'llama_memory_seq_rm_attn_only')
    if _has_seq_rm_attn_only:
        _lib.llama_memory_seq_rm_attn_only.argtypes = [llama_memory_t, llama_seq_id, llama_pos, llama_pos]
        _lib.llama_memory_seq_rm_attn_only.restype = ctypes.c_bool

    _signatures_bound = True

_has_seq_rm_attn_only = False


# ---------------------------------------------------------------------------
# Context creation via C shim (avoids struct layout issues)
# ---------------------------------------------------------------------------
_shim_lib = None

def _compile_shim():
    """Compile a small C shim to create contexts with attn_weights enabled."""
    global _shim_lib
    if _shim_lib is not None:
        return _shim_lib

    _check_lib()

    shim_src = r"""
#include "llama.h"
#include <stdlib.h>

__attribute__((visibility("default")))
struct llama_context * create_ctx_with_attn(
        struct llama_model * model,
        int n_ctx, int n_batch, int attn_weights, int n_gpu_layers) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_batch;
    params.attn_weights = attn_weights ? true : false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.offload_kqv = n_gpu_layers > 0;
    return llama_init_from_model(model, params);
}
"""
    lib_dir = os.path.dirname(_lib_path)
    # Try to find include dirs relative to the lib
    base_dir = os.path.dirname(lib_dir)
    include_candidates = [
        os.path.join(base_dir, "include"),
        os.path.join(base_dir, "..", "include"),
    ]
    ggml_candidates = [
        os.path.join(base_dir, "ggml", "include"),
        os.path.join(base_dir, "..", "ggml", "include"),
    ]
    include_dir = next((d for d in include_candidates if os.path.isdir(d)), include_candidates[0])
    ggml_include = next((d for d in ggml_candidates if os.path.isdir(d)), ggml_candidates[0])

    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(shim_src)
        src_path = f.name

    ext = ".dylib" if sys.platform == "darwin" else ".so"
    shim_path = os.path.join(lib_dir, f"libllama_attn_shim{ext}")

    cmd = [
        "cc", "-shared", "-fPIC", "-o", shim_path, src_path,
        f"-I{include_dir}", f"-I{ggml_include}",
        f"-L{lib_dir}", "-lllama",
        f"-Wl,-rpath,{lib_dir}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(src_path)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to compile llama shim: {result.stderr}")

    _shim_lib = ctypes.CDLL(shim_path)
    _shim_lib.create_ctx_with_attn.argtypes = [
        ctypes.POINTER(_llama_model), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    _shim_lib.create_ctx_with_attn.restype = ctypes.POINTER(_llama_context)
    return _shim_lib


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init():
    """Initialize the llama.cpp backend. Call once at startup."""
    _bind_signatures()
    _lib.llama_backend_init()


def cleanup():
    """Free the llama.cpp backend."""
    _check_lib()
    _lib.llama_backend_free()


def load_model(path: str, n_gpu_layers: int = 0):
    """Load a GGUF model. Returns an opaque model pointer."""
    _bind_signatures()
    params = _lib.llama_model_default_params()
    model = _lib.llama_model_load_from_file(path.encode(), params)
    if not model:
        raise RuntimeError(f"Failed to load model from {path}")
    return model


def free_model(model):
    """Free a loaded model."""
    _lib.llama_model_free(model)


def create_context(model, n_ctx: int = 2048, n_batch: int = 2048, attn_weights: bool = True):
    """Create a llama_context with optional attention weight extraction."""
    shim = _compile_shim()
    ctx = shim.create_ctx_with_attn(model, n_ctx, n_batch, 1 if attn_weights else 0, 0)
    if not ctx:
        raise RuntimeError("Failed to create llama_context")
    return ctx


def free_context(ctx):
    """Free a context."""
    _lib.llama_free(ctx)


def get_vocab(model):
    """Get vocabulary pointer from model."""
    return _lib.llama_model_get_vocab(model)


def n_layer(model) -> int:
    return _lib.llama_model_n_layer(model)


def n_head(model) -> int:
    return _lib.llama_model_n_head(model)


def n_vocab(vocab) -> int:
    return _lib.llama_vocab_n_tokens(vocab)


def n_ctx(ctx) -> int:
    return _lib.llama_n_ctx(ctx)


def vocab_bos(vocab) -> int:
    return _lib.llama_vocab_bos(vocab)


def vocab_eos(vocab) -> int:
    return _lib.llama_vocab_eos(vocab)


# --- Tokenization ---

def tokenize(vocab, text: str, add_bos: bool = True, special: bool = True) -> List[int]:
    """Tokenize text, returning a list of token ids."""
    text_bytes = text.encode("utf-8")
    buf = (llama_token * (len(text_bytes) + 32))()
    n = _lib.llama_tokenize(vocab, text_bytes, len(text_bytes), buf, len(buf), add_bos, special)
    if n < 0:
        buf = (llama_token * (-n))()
        n = _lib.llama_tokenize(vocab, text_bytes, len(text_bytes), buf, len(buf), add_bos, special)
    return list(buf[:n])


def token_to_piece_bytes(vocab, token_id: int, special: bool = True) -> bytes:
    """Convert a single token ID to raw bytes."""
    buf = (ctypes.c_char * 256)()
    n = _lib.llama_token_to_piece(vocab, token_id, buf, 256, 0, special)
    if n > 0:
        return buf[:n]
    return b""


def token_to_piece(vocab, token_id: int, special: bool = True) -> str:
    """Convert a single token ID to string (may produce mojibake for byte-fallback)."""
    return token_to_piece_bytes(vocab, token_id, special).decode("utf-8", errors="replace")


def tokens_to_text(vocab, token_ids: List[int], special: bool = True, errors: str = "replace") -> str:
    """Convert token IDs to text, handling byte-fallback tokens correctly.

    Accumulates all raw bytes and decodes once to avoid mojibake from
    partial UTF-8 sequences (e.g. CJK characters split across byte tokens).
    """
    raw = b"".join(token_to_piece_bytes(vocab, t, special) for t in token_ids)
    return raw.decode("utf-8", errors=errors)


# --- Batch decoding ---

def decode_batch_at(ctx, tokens: List[int], pos_start: int = 0, seq_id: int = 0,
                    output_last_only: bool = True) -> int:
    """Decode a batch of tokens starting at pos_start. Returns decode status."""
    n = len(tokens)
    if n == 0:
        return 0
    batch = _lib.llama_batch_init(n, 0, 1)
    for i in range(n):
        batch.token[i] = tokens[i]
        batch.pos[i] = pos_start + i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = seq_id
        batch.logits[i] = 1 if (not output_last_only or i == n - 1) else 0
    batch.n_tokens = n
    ret = _lib.llama_decode(ctx, batch)
    _lib.llama_batch_free(batch)
    return ret


def decode_batch(ctx, tokens: List[int], output_last_only: bool = True) -> int:
    """Decode a batch at position 0 (convenience for non-KV-cache mode)."""
    return decode_batch_at(ctx, tokens, pos_start=0, output_last_only=output_last_only)


def decode_single_at(ctx, token: int, pos: int, seq_id: int = 0, output: bool = True) -> int:
    """Decode a single token at a specific position."""
    batch = _lib.llama_batch_init(1, 0, 1)
    batch.token[0] = token
    batch.pos[0] = pos
    batch.n_seq_id[0] = 1
    batch.seq_id[0][0] = seq_id
    batch.logits[0] = 1 if output else 0
    batch.n_tokens = 1
    ret = _lib.llama_decode(ctx, batch)
    _lib.llama_batch_free(batch)
    return ret


# --- Attention weights ---

def set_attn_heads(ctx, layers: List[int], heads: List[int]):
    """Configure which (layer, head) pairs to extract attention from."""
    n = len(layers)
    assert len(heads) == n, f"layers ({n}) and heads ({len(heads)}) must match"
    l_arr = (ctypes.c_int32 * n)(*layers)
    h_arr = (ctypes.c_int32 * n)(*heads)
    _lib.llama_set_attn_heads(ctx, l_arr, h_arr, n)


def get_attn_weights(ctx, token_idx: int, n_pairs: int, ctx_size: int = 0) -> Optional[np.ndarray]:
    """Get attention weights for a given output token.

    The internal layout is [n_pairs * n_ctx] floats, where n_ctx is the full
    context window size.  ``ctx_size`` is kept for backward compat but ignored;
    the stride is always the true n_ctx obtained from the context.

    Returns numpy array of shape (n_pairs, n_kv) or None.
    """
    ptr = _lib.llama_get_attn_ith(ctx, token_idx)
    if not ptr:
        return None
    n_kv = _lib.llama_get_attn_n_kv(ctx)
    if n_kv <= 0:
        return None
    # Stride between heads is n_ctx (full context window), NOT current pos
    full_n_ctx = int(n_ctx(ctx))
    result = np.zeros((n_pairs, n_kv), dtype=np.float32)
    for p in range(n_pairs):
        offset = p * full_n_ctx
        arr = (ctypes.c_float * n_kv).from_address(ctypes.addressof(ptr.contents) + offset * 4)
        result[p] = np.frombuffer(arr, dtype=np.float32)
    return result


# --- Logits ---

def argmax_logits(ctx, token_idx: int, n_vocab: int) -> int:
    """Get argmax of logits for a given output token. Returns -1 on failure."""
    ptr = _lib.llama_get_logits_ith(ctx, token_idx)
    if not ptr:
        return -1
    logits = (ctypes.c_float * n_vocab).from_address(ctypes.addressof(ptr.contents))
    return int(np.argmax(np.frombuffer(logits, dtype=np.float32)))


def get_logits_array(ctx, token_idx: int, n_vocab: int) -> Optional[np.ndarray]:
    """Get raw logits as a numpy array for a given output token."""
    ptr = _lib.llama_get_logits_ith(ctx, token_idx)
    if not ptr:
        return None
    logits = (ctypes.c_float * n_vocab).from_address(ctypes.addressof(ptr.contents))
    return np.frombuffer(logits, dtype=np.float32).copy()


# --- KV Cache / Memory management ---

def get_memory(ctx):
    """Get the KV cache memory object for a context."""
    return _lib.llama_get_memory(ctx)


def memory_clear(mem, data: bool = True):
    """Clear all KV cache data."""
    _lib.llama_memory_clear(mem, data)


def memory_seq_rm(mem, seq_id: int, p0: int, p1: int) -> bool:
    """Remove tokens in position range [p0, p1) from a sequence.

    seq_id < 0: match any sequence. p0 < 0: [0, p1]. p1 < 0: [p0, inf).
    """
    return _lib.llama_memory_seq_rm(mem, seq_id, p0, p1)


def memory_seq_rm_attn_only(mem, seq_id: int, p0: int, p1: int) -> bool:
    """Remove tokens from attention cache only (for hybrid models like Qwen3.5).

    Falls back to regular seq_rm for non-hybrid models.
    """
    if _has_seq_rm_attn_only:
        return _lib.llama_memory_seq_rm_attn_only(mem, seq_id, p0, p1)
    return _lib.llama_memory_seq_rm(mem, seq_id, p0, p1)


def memory_seq_cp(mem, src_id: int, dst_id: int, p0: int, p1: int):
    """Copy KV cache from src_id to dst_id for position range [p0, p1)."""
    _lib.llama_memory_seq_cp(mem, src_id, dst_id, p0, p1)


def memory_seq_keep(mem, seq_id: int):
    """Remove all tokens NOT belonging to seq_id."""
    _lib.llama_memory_seq_keep(mem, seq_id)


def memory_seq_pos_max(mem, seq_id: int) -> int:
    """Get max position in sequence. Returns -1 if empty."""
    return _lib.llama_memory_seq_pos_max(mem, seq_id)


# --- Convenience ---

def is_available() -> bool:
    """Check if llama.cpp library is available."""
    return _lib is not None
