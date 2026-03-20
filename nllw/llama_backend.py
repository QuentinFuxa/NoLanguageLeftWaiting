"""Minimal ctypes wrapper for llama.cpp with attention weight extraction API.

Requires a custom llama.cpp build with attention weights extraction support:
  https://github.com/QuentinFuxa/llama.cpp (branch: add-attention-weights-extraction-API--EXPERIMENTAL-)
"""

import ctypes
import os
import subprocess
import sys
import tempfile

import numpy as np


# --- Stderr suppression for clean TUI ---

_stderr_suppressed = False
_saved_stderr_fd = None

def suppress_stderr():
    """Redirect stderr to /dev/null (suppress llama.cpp verbose logs)."""
    global _stderr_suppressed, _saved_stderr_fd
    if not _stderr_suppressed:
        _saved_stderr_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        _stderr_suppressed = True

def restore_stderr():
    """Restore stderr."""
    global _stderr_suppressed, _saved_stderr_fd
    if _stderr_suppressed and _saved_stderr_fd is not None:
        os.dup2(_saved_stderr_fd, 2)
        os.close(_saved_stderr_fd)
        _saved_stderr_fd = None
        _stderr_suppressed = False


# --- Library discovery ---

def _find_lib() -> str | None:
    """Find libllama shared library path."""
    env_path = os.environ.get("LLAMA_LIB_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    lib_name = "libllama.dylib" if sys.platform == "darwin" else "libllama.so"
    search_paths = [
        os.path.expanduser(f"~/.local/lib/{lib_name}"),
        f"/usr/local/lib/{lib_name}",
        f"./build/src/{lib_name}",
        f"./build/bin/{lib_name}",
        os.path.join(os.path.dirname(__file__), "..", "build", "bin", lib_name),
    ]
    for p in search_paths:
        if os.path.exists(p):
            return os.path.abspath(p)

    return None


def _load_library():
    """Load libllama, raising a clear error if not found."""
    lib_path = _find_lib()
    if lib_path is None:
        raise RuntimeError(
            "Cannot find libllama. Build llama.cpp with attention weights API and either:\n"
            "  1. Set LLAMA_LIB_PATH=/path/to/libllama.dylib\n"
            "  2. Install to ~/.local/lib/ or /usr/local/lib/\n"
            "  3. Place build output in ./build/src/ or ./build/bin/\n\n"
            "Build from: https://github.com/QuentinFuxa/llama.cpp "
            "(branch: add-attention-weights-extraction-API--EXPERIMENTAL-)"
        )
    return ctypes.CDLL(lib_path), lib_path


_lib, _lib_path = _load_library()
_lib_dir = os.path.dirname(_lib_path)


# --- Types ---

class llama_model(ctypes.Structure):
    pass

class llama_context(ctypes.Structure):
    pass

class llama_vocab(ctypes.Structure):
    pass

llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32


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
    _fields_ = [
        ("_opaque", ctypes.c_uint8 * 256),
    ]


# --- Function signatures ---

_lib.llama_backend_init.argtypes = []
_lib.llama_backend_init.restype = None

_lib.llama_backend_free.argtypes = []
_lib.llama_backend_free.restype = None

_lib.llama_model_default_params.argtypes = []
_lib.llama_model_default_params.restype = llama_model_params

_lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
_lib.llama_model_load_from_file.restype = ctypes.POINTER(llama_model)

_lib.llama_model_free.argtypes = [ctypes.POINTER(llama_model)]
_lib.llama_model_free.restype = None

_lib.llama_model_get_vocab.argtypes = [ctypes.POINTER(llama_model)]
_lib.llama_model_get_vocab.restype = ctypes.POINTER(llama_vocab)

_lib.llama_model_n_layer.argtypes = [ctypes.POINTER(llama_model)]
_lib.llama_model_n_layer.restype = ctypes.c_int32

_lib.llama_model_n_head.argtypes = [ctypes.POINTER(llama_model)]
_lib.llama_model_n_head.restype = ctypes.c_int32

_lib.llama_tokenize.argtypes = [
    ctypes.POINTER(llama_vocab), ctypes.c_char_p, ctypes.c_int32,
    ctypes.POINTER(llama_token), ctypes.c_int32, ctypes.c_bool, ctypes.c_bool
]
_lib.llama_tokenize.restype = ctypes.c_int32

_lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
_lib.llama_batch_init.restype = llama_batch

_lib.llama_batch_free.argtypes = [llama_batch]
_lib.llama_batch_free.restype = None

_lib.llama_decode.argtypes = [ctypes.POINTER(llama_context), llama_batch]
_lib.llama_decode.restype = ctypes.c_int32

_lib.llama_free.argtypes = [ctypes.POINTER(llama_context)]
_lib.llama_free.restype = None

_lib.llama_set_attn_heads.argtypes = [
    ctypes.POINTER(llama_context),
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t
]
_lib.llama_set_attn_heads.restype = None

_lib.llama_get_attn_ith.argtypes = [ctypes.POINTER(llama_context), ctypes.c_int32]
_lib.llama_get_attn_ith.restype = ctypes.POINTER(ctypes.c_float)

_lib.llama_get_attn_n_kv.argtypes = [ctypes.POINTER(llama_context)]
_lib.llama_get_attn_n_kv.restype = ctypes.c_int32

_lib.llama_get_logits_ith.argtypes = [ctypes.POINTER(llama_context), ctypes.c_int32]
_lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

_lib.llama_n_ctx.argtypes = [ctypes.POINTER(llama_context)]
_lib.llama_n_ctx.restype = ctypes.c_uint32

_lib.llama_synchronize.argtypes = [ctypes.POINTER(llama_context)]
_lib.llama_synchronize.restype = None

_lib.llama_vocab_n_tokens.argtypes = [ctypes.POINTER(llama_vocab)]
_lib.llama_vocab_n_tokens.restype = ctypes.c_int32

_lib.llama_vocab_bos.argtypes = [ctypes.POINTER(llama_vocab)]
_lib.llama_vocab_bos.restype = llama_token

_lib.llama_vocab_eos.argtypes = [ctypes.POINTER(llama_vocab)]
_lib.llama_vocab_eos.restype = llama_token

_lib.llama_token_to_piece.argtypes = [
    ctypes.POINTER(llama_vocab), llama_token,
    ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32, ctypes.c_bool
]
_lib.llama_token_to_piece.restype = ctypes.c_int32


# --- KV Cache Memory ---

class llama_memory_i(ctypes.Structure):
    pass

llama_memory_t = ctypes.POINTER(llama_memory_i)

_lib.llama_get_memory.argtypes = [ctypes.POINTER(llama_context)]
_lib.llama_get_memory.restype = llama_memory_t

_lib.llama_memory_clear.argtypes = [llama_memory_t, ctypes.c_bool]
_lib.llama_memory_clear.restype = None

_lib.llama_memory_seq_rm.argtypes = [llama_memory_t, llama_seq_id, llama_pos, llama_pos]
_lib.llama_memory_seq_rm.restype = ctypes.c_bool


# --- Context creation via C shim ---

def _create_context(model_ptr, n_ctx=512, n_batch=512, attn_weights=True, n_gpu_layers=0):
    """Create a llama_context with attention weights enabled.

    Compiles a small C shim on-the-fly to avoid struct layout issues.
    The shim is cached in the library directory.
    """
    shim_ext = ".dylib" if sys.platform == "darwin" else ".so"
    shim_lib_path = os.path.join(_lib_dir, f"libllama_attn_shim{shim_ext}")

    if not os.path.exists(shim_lib_path):
        real_lib = os.path.realpath(_lib_path)
        real_lib_dir = os.path.dirname(real_lib)
        llama_root = os.path.dirname(real_lib_dir)
        include_dir = None
        ggml_include = None

        env_include = os.environ.get("LLAMA_INCLUDE_DIR")
        if env_include and os.path.exists(os.path.join(env_include, "llama.h")):
            include_dir = env_include
        else:
            for candidate in [llama_root, os.path.dirname(llama_root), os.path.dirname(_lib_dir), os.path.dirname(os.path.dirname(_lib_dir))]:
                inc = os.path.join(candidate, "include")
                if os.path.exists(os.path.join(inc, "llama.h")):
                    include_dir = inc
                    ggml_inc = os.path.join(candidate, "ggml", "include")
                    if os.path.exists(ggml_inc):
                        ggml_include = ggml_inc
                    break

        if include_dir is None:
            raise RuntimeError(
                f"Cannot find llama.h headers near {_lib_path}. "
                "Set LLAMA_INCLUDE_DIR to the directory containing llama.h"
            )

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

__attribute__((visibility("default")))
struct llama_model * load_model_with_gpu(const char * path, int n_gpu_layers) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    return llama_model_load_from_file(path, params);
}
"""
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(shim_src)
            src_path = f.name

        cmd = [
            "cc", "-shared", "-fPIC", "-o", shim_lib_path, src_path,
            f"-I{include_dir}",
            f"-L{_lib_dir}", "-lllama",
            f"-Wl,-rpath,{_lib_dir}",
        ]
        if ggml_include:
            cmd.append(f"-I{ggml_include}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        os.unlink(src_path)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to compile context shim:\n{result.stderr}")

    shim = ctypes.CDLL(shim_lib_path)
    shim.create_ctx_with_attn.argtypes = [
        ctypes.POINTER(llama_model), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    shim.create_ctx_with_attn.restype = ctypes.POINTER(llama_context)
    shim.load_model_with_gpu.argtypes = [ctypes.c_char_p, ctypes.c_int]
    shim.load_model_with_gpu.restype = ctypes.POINTER(llama_model)

    _create_context._shim = shim  # cache on function object

    ctx = shim.create_ctx_with_attn(
        model_ptr, n_ctx, n_batch, 1 if attn_weights else 0, n_gpu_layers
    )
    if not ctx:
        raise RuntimeError("Failed to create llama_context")
    return ctx


_shim_lib = None

def _ensure_shim():
    """Ensure the C shim is compiled and loaded. Returns the shim CDLL."""
    global _shim_lib
    if _shim_lib is not None:
        return
    # Trigger shim compilation by creating a dummy context (which we immediately free)
    # Actually just check if shim exists on disk
    shim_ext = ".dylib" if sys.platform == "darwin" else ".so"
    shim_lib_path = os.path.join(_lib_dir, f"libllama_attn_shim{shim_ext}")
    if not os.path.exists(shim_lib_path):
        # Force compilation by calling _create_context with a None model -- will fail but compile shim
        # Better: just compile directly
        import tempfile, subprocess
        # Resolve symlinks to find the real llama.cpp directory
        real_lib = os.path.realpath(_lib_path)
        real_lib_dir = os.path.dirname(real_lib)
        llama_root = os.path.dirname(real_lib_dir)  # e.g. .../llama.cpp/build
        include_dir = None
        ggml_include = None
        # Search: lib_dir parents, real lib parents, env var
        candidates = [
            llama_root,                          # build/
            os.path.dirname(llama_root),         # llama.cpp/
            os.path.dirname(_lib_dir),           # symlink parent
            os.path.dirname(os.path.dirname(_lib_dir)),
        ]
        for candidate in candidates:
            inc = os.path.join(candidate, "include")
            if os.path.exists(os.path.join(inc, "llama.h")):
                include_dir = inc
                ggml_inc = os.path.join(candidate, "ggml", "include")
                if os.path.exists(ggml_inc):
                    ggml_include = ggml_inc
                break
        env_include = os.environ.get("LLAMA_INCLUDE_DIR")
        if env_include and os.path.exists(os.path.join(env_include, "llama.h")):
            include_dir = env_include
        if include_dir is None:
            raise RuntimeError("Cannot find llama.h headers. Set LLAMA_INCLUDE_DIR.")

        shim_src = r"""
#include "llama.h"
__attribute__((visibility("default")))
struct llama_context * create_ctx_with_attn(struct llama_model * model, int n_ctx, int n_batch, int attn_weights, int n_gpu_layers) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx; params.n_batch = n_batch; params.n_ubatch = n_batch;
    params.attn_weights = attn_weights ? true : false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.offload_kqv = n_gpu_layers > 0;
    return llama_init_from_model(model, params);
}
__attribute__((visibility("default")))
struct llama_model * load_model_with_gpu(const char * path, int n_gpu_layers) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    return llama_model_load_from_file(path, params);
}
"""
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(shim_src)
            src_path = f.name
        cmd = ["cc", "-shared", "-fPIC", "-o", shim_lib_path, src_path, f"-I{include_dir}", f"-L{_lib_dir}", "-lllama", f"-Wl,-rpath,{_lib_dir}"]
        if ggml_include:
            cmd.append(f"-I{ggml_include}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        os.unlink(src_path)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to compile shim:\n{result.stderr}")

    shim = ctypes.CDLL(shim_lib_path)
    shim.create_ctx_with_attn.argtypes = [ctypes.POINTER(llama_model), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    shim.create_ctx_with_attn.restype = ctypes.POINTER(llama_context)
    shim.load_model_with_gpu.argtypes = [ctypes.c_char_p, ctypes.c_int]
    shim.load_model_with_gpu.restype = ctypes.POINTER(llama_model)
    _shim_lib = shim


# --- High-level helpers ---

def tokenize(vocab_ptr, text, add_bos=True, special=True):
    """Tokenize text, returning a list of token ids."""
    text_bytes = text.encode("utf-8")
    buf = (llama_token * (len(text_bytes) + 32))()
    n = _lib.llama_tokenize(vocab_ptr, text_bytes, len(text_bytes), buf, len(buf), add_bos, special)
    if n < 0:
        buf = (llama_token * (-n))()
        n = _lib.llama_tokenize(vocab_ptr, text_bytes, len(text_bytes), buf, len(buf), add_bos, special)
    return list(buf[:n])


def decode_batch(ctx_ptr, tokens, output_last_only=True):
    """Decode a batch of tokens starting at position 0."""
    n = len(tokens)
    batch = _lib.llama_batch_init(n, 0, 1)
    for i in range(n):
        batch.token[i] = tokens[i]
        batch.pos[i] = i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = 1 if (not output_last_only or i == n - 1) else 0
    batch.n_tokens = n
    ret = _lib.llama_decode(ctx_ptr, batch)
    _lib.llama_batch_free(batch)
    return ret


def decode_single(ctx_ptr, token, pos, output=True):
    """Decode a single token at a given position."""
    batch = _lib.llama_batch_init(1, 0, 1)
    batch.token[0] = token
    batch.pos[0] = pos
    batch.n_seq_id[0] = 1
    batch.seq_id[0][0] = 0
    batch.logits[0] = 1 if output else 0
    batch.n_tokens = 1
    ret = _lib.llama_decode(ctx_ptr, batch)
    _lib.llama_batch_free(batch)
    return ret


def get_attn_weights(ctx_ptr, token_idx, n_pairs, n_ctx_size):
    """Get attention weights for a given output token index.

    Returns numpy array of shape (n_pairs, n_kv) or None.
    """
    ptr = _lib.llama_get_attn_ith(ctx_ptr, token_idx)
    if not ptr:
        return None

    n_kv = _lib.llama_get_attn_n_kv(ctx_ptr)
    if n_kv <= 0:
        return None

    result = np.zeros((n_pairs, n_kv), dtype=np.float32)
    for p in range(n_pairs):
        offset = p * n_ctx_size
        arr = (ctypes.c_float * n_kv).from_address(ctypes.addressof(ptr.contents) + offset * 4)
        result[p] = np.frombuffer(arr, dtype=np.float32)

    return result


def argmax_logits(ctx_ptr, token_idx, n_vocab_size):
    """Get the argmax of logits for a given output token."""
    ptr = _lib.llama_get_logits_ith(ctx_ptr, token_idx)
    if not ptr:
        return -1
    logits = (ctypes.c_float * n_vocab_size).from_address(ctypes.addressof(ptr.contents))
    return int(np.argmax(np.frombuffer(logits, dtype=np.float32)))


def get_logits_array(ctx_ptr, token_idx, n_vocab_size):
    """Get raw logits as a numpy array for a given output token.

    Useful for computing entropy or top-k probabilities for confidence
    estimation in border detection.
    """
    ptr = _lib.llama_get_logits_ith(ctx_ptr, token_idx)
    if not ptr:
        return None
    logits = (ctypes.c_float * n_vocab_size).from_address(ctypes.addressof(ptr.contents))
    return np.frombuffer(logits, dtype=np.float32).copy()


def decode_batch_at(ctx_ptr, tokens, pos_start, seq_id=0, output_last_only=True):
    """Decode a batch of tokens starting at a given position.

    Unlike decode_batch, this allows specifying the start position and seq_id,
    enabling KV cache delta decoding (only decode new tokens).
    """
    n = len(tokens)
    batch = _lib.llama_batch_init(n, 0, 1)
    for i in range(n):
        batch.token[i] = tokens[i]
        batch.pos[i] = pos_start + i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = seq_id
        batch.logits[i] = 1 if (not output_last_only or i == n - 1) else 0
    batch.n_tokens = n
    ret = _lib.llama_decode(ctx_ptr, batch)
    _lib.llama_batch_free(batch)
    return ret


def decode_single_at(ctx_ptr, token, pos, seq_id=0, output=True):
    """Decode a single token at a specific position and sequence.

    Unlike decode_single, this allows specifying seq_id for multi-sequence
    KV cache management.
    """
    batch = _lib.llama_batch_init(1, 0, 1)
    batch.token[0] = token
    batch.pos[0] = pos
    batch.n_seq_id[0] = 1
    batch.seq_id[0][0] = seq_id
    batch.logits[0] = 1 if output else 0
    batch.n_tokens = 1
    ret = _lib.llama_decode(ctx_ptr, batch)
    _lib.llama_batch_free(batch)
    return ret


def token_to_piece_bytes(vocab_ptr, token_id, special=True):
    """Convert a single token ID to raw bytes."""
    buf = (ctypes.c_char * 256)()
    n = _lib.llama_token_to_piece(vocab_ptr, token_id, buf, 256, 0, special)
    if n > 0:
        return buf[:n]
    return b""


def token_to_piece(vocab_ptr, token_id, special=True):
    """Convert a single token ID to its string piece."""
    return token_to_piece_bytes(vocab_ptr, token_id, special).decode("utf-8", errors="replace")


def tokens_to_text(vocab_ptr, token_ids, special=True, errors="replace"):
    """Convert a list of token IDs to text, handling byte-fallback tokens."""
    raw = b"".join(token_to_piece_bytes(vocab_ptr, t, special) for t in token_ids)
    return raw.decode("utf-8", errors=errors)


# --- Public API ---

def init(suppress_log=True):
    if suppress_log:
        # Suppress llama.cpp verbose logging to keep TUI clean
        _saved_stderr = os.dup(2)
        _devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull, 2)
        os.close(_devnull)
    _lib.llama_backend_init()
    if suppress_log:
        os.dup2(_saved_stderr, 2)
        os.close(_saved_stderr)

def cleanup():
    _lib.llama_backend_free()

def load_model(path, n_gpu_layers=99, suppress_log=True):
    # Use the C shim to load model with proper n_gpu_layers
    _ensure_shim()
    if suppress_log:
        _saved = os.dup(2)
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    model = _shim_lib.load_model_with_gpu(path.encode(), n_gpu_layers)
    if suppress_log:
        os.dup2(_saved, 2)
        os.close(_saved)
    if not model:
        raise RuntimeError(f"Failed to load model from {path}")
    return model

def create_context(model, n_ctx=512, n_batch=512, attn_weights=True):
    return _create_context(model, n_ctx, n_batch, attn_weights, n_gpu_layers=99)

def set_attn_heads(ctx, layers, heads):
    n = len(layers)
    assert len(heads) == n
    l_arr = (ctypes.c_int32 * n)(*layers)
    h_arr = (ctypes.c_int32 * n)(*heads)
    _lib.llama_set_attn_heads(ctx, l_arr, h_arr, n)

def get_vocab(model):
    return _lib.llama_model_get_vocab(model)

def n_vocab(vocab):
    return _lib.llama_vocab_n_tokens(vocab)

def n_ctx(ctx):
    return _lib.llama_n_ctx(ctx)

def vocab_eos(vocab):
    return _lib.llama_vocab_eos(vocab)

def get_memory(ctx):
    return _lib.llama_get_memory(ctx)

def memory_clear(mem, data=True):
    _lib.llama_memory_clear(mem, data)

def memory_seq_rm(mem, seq_id, p0, p1):
    return _lib.llama_memory_seq_rm(mem, seq_id, p0, p1)

def n_layer(model):
    """Get the number of layers in the model."""
    return _lib.llama_model_n_layer(model)

def n_head(model):
    """Get the number of attention heads per layer."""
    return _lib.llama_model_n_head(model)

def free_context(ctx):
    _lib.llama_free(ctx)

def free_model(model):
    _lib.llama_model_free(model)


# --- LoRA adapter support ---

class llama_adapter_lora(ctypes.Structure):
    """Opaque struct for a LoRA adapter handle."""
    pass


# Probe the C API for LoRA symbols at import time.
_has_lora_api = False
try:
    _lib.llama_adapter_lora_init.argtypes = [
        ctypes.POINTER(llama_model), ctypes.c_char_p
    ]
    _lib.llama_adapter_lora_init.restype = ctypes.POINTER(llama_adapter_lora)

    _lib.llama_set_adapters_lora.argtypes = [
        ctypes.POINTER(llama_context),
        ctypes.POINTER(ctypes.POINTER(llama_adapter_lora)),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    _lib.llama_set_adapters_lora.restype = ctypes.c_int32

    # llama_adapter_lora_free is deprecated in recent builds (adapters are freed
    # together with the model), but we bind it anyway for older builds.
    try:
        _lib.llama_adapter_lora_free.argtypes = [ctypes.POINTER(llama_adapter_lora)]
        _lib.llama_adapter_lora_free.restype = None
    except AttributeError:
        pass

    _has_lora_api = True
except AttributeError:
    # The loaded libllama was built without LoRA support (older build).
    _has_lora_api = False


def has_lora_support() -> bool:
    """Return True if the loaded libllama exposes the LoRA adapter C API."""
    return _has_lora_api


def load_lora(model, lora_path: str, scale: float = 1.0):
    """Load a GGUF LoRA adapter and return its handle.

    Args:
        model: Pointer to a loaded ``llama_model``.
        lora_path: Filesystem path to a ``.gguf`` LoRA adapter file.
        scale: Adapter weight (1.0 = full strength).

    Returns:
        An opaque adapter handle (``ctypes.POINTER(llama_adapter_lora)``).

    Raises:
        RuntimeError: If the LoRA C API is not available or loading fails.
    """
    if not _has_lora_api:
        raise RuntimeError(
            "LoRA adapter API not available in this libllama build. "
            "Rebuild llama.cpp from https://github.com/QuentinFuxa/llama.cpp "
            "(branch: add-attention-weights-extraction-API--EXPERIMENTAL-) "
            "or upgrade to a build that includes llama_adapter_lora_init."
        )

    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f"LoRA adapter file not found: {lora_path}")

    adapter = _lib.llama_adapter_lora_init(model, lora_path.encode("utf-8"))
    if not adapter:
        raise RuntimeError(f"Failed to load LoRA adapter from {lora_path}")

    return adapter


def apply_lora(ctx, adapters, scales=None):
    """Attach one or more LoRA adapters to a llama_context.

    Args:
        ctx: Pointer to a ``llama_context``.
        adapters: A single adapter handle or a list of adapter handles
            returned by :func:`load_lora`.
        scales: A float or list of floats (one per adapter). Defaults to 1.0
            for every adapter.

    Returns:
        0 on success, non-zero on failure.
    """
    if not _has_lora_api:
        raise RuntimeError("LoRA adapter API not available in this libllama build.")

    # Normalise to lists
    if not isinstance(adapters, (list, tuple)):
        adapters = [adapters]
    n = len(adapters)

    if scales is None:
        scales = [1.0] * n
    elif not isinstance(scales, (list, tuple)):
        scales = [scales] * n

    assert len(scales) == n, "adapters and scales must have the same length"

    adapter_arr = (ctypes.POINTER(llama_adapter_lora) * n)(*adapters)
    scale_arr = (ctypes.c_float * n)(*scales)
    return _lib.llama_set_adapters_lora(ctx, adapter_arr, n, scale_arr)


def clear_lora(ctx):
    """Remove all LoRA adapters from a context.

    Equivalent to calling ``llama_set_adapters_lora`` with zero adapters.
    """
    if not _has_lora_api:
        return
    empty_adapters = (ctypes.POINTER(llama_adapter_lora) * 0)()
    empty_scales = (ctypes.c_float * 0)()
    _lib.llama_set_adapters_lora(ctx, empty_adapters, 0, empty_scales)
