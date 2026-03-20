"""
llama.cpp ctypes bindings for attention weight extraction and KV cache management.

Provides a clean Python API for:
- Model loading and context creation
- Tokenization and detokenization
- Batch/single token decoding with position control
- Attention weight extraction from specific heads
- KV cache management (seq_rm, clear, etc.)
- Logits extraction (argmax, full array)

Requires llama.cpp built with attention weight extraction support (PR #20086).
The shared library (libllama.so/.dylib) must be findable via LLAMA_LIB_PATH
environment variable, or in standard locations.

Usage:
    from nllw.llama_backend import LlamaBackend

    backend = LlamaBackend("/path/to/model.gguf", n_ctx=2048)
    tokens = backend.tokenize("Hello world")
    backend.decode_batch(tokens)
    logits = backend.get_logits_array(-1)
    backend.close()
"""

import ctypes
import os
import sys
import tempfile
import subprocess
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _find_llama_lib() -> ctypes.CDLL:
    """Find and load the llama.cpp shared library."""
    # Check environment variable first
    env_path = os.environ.get("LLAMA_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return ctypes.CDLL(env_path)

    # Search common locations
    search_dirs = [
        os.environ.get("LLAMA_LIB_DIR", ""),
        os.path.join(os.path.dirname(__file__), "..", "build", "bin"),
        os.path.join(os.path.dirname(__file__), "..", "..", "llama.cpp", "build", "bin"),
        "/usr/local/lib",
        "/usr/lib",
    ]

    lib_names = ["libllama.dylib", "libllama.so", "llama.dll"]

    for d in search_dirs:
        if not d:
            continue
        for name in lib_names:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return ctypes.CDLL(path)

    raise RuntimeError(
        "Cannot find libllama shared library. Set LLAMA_LIB_PATH or LLAMA_LIB_DIR, "
        "or build llama.cpp with: cmake --build build"
    )


def _load_lib():
    """Load library and set up function signatures. Returns (lib, has_attn_only)."""
    lib = _find_llama_lib()

    # --- Opaque types ---
    # (defined at module level for use in type hints)

    # --- Function signatures ---

    # Backend lifecycle
    lib.llama_backend_init.argtypes = []
    lib.llama_backend_init.restype = None
    lib.llama_backend_free.argtypes = []
    lib.llama_backend_free.restype = None

    # Model params
    lib.llama_model_default_params.argtypes = []
    lib.llama_model_default_params.restype = _llama_model_params

    # Model lifecycle
    lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, _llama_model_params]
    lib.llama_model_load_from_file.restype = ctypes.POINTER(_llama_model)
    lib.llama_model_free.argtypes = [ctypes.POINTER(_llama_model)]
    lib.llama_model_free.restype = None

    # Model info
    lib.llama_model_get_vocab.argtypes = [ctypes.POINTER(_llama_model)]
    lib.llama_model_get_vocab.restype = ctypes.POINTER(_llama_vocab)
    lib.llama_model_n_layer.argtypes = [ctypes.POINTER(_llama_model)]
    lib.llama_model_n_layer.restype = ctypes.c_int32
    lib.llama_model_n_head.argtypes = [ctypes.POINTER(_llama_model)]
    lib.llama_model_n_head.restype = ctypes.c_int32

    # Tokenization
    lib.llama_tokenize.argtypes = [
        ctypes.POINTER(_llama_vocab), ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(_llama_token), ctypes.c_int32, ctypes.c_bool, ctypes.c_bool,
    ]
    lib.llama_tokenize.restype = ctypes.c_int32
    lib.llama_vocab_n_tokens.argtypes = [ctypes.POINTER(_llama_vocab)]
    lib.llama_vocab_n_tokens.restype = ctypes.c_int32
    lib.llama_vocab_bos.argtypes = [ctypes.POINTER(_llama_vocab)]
    lib.llama_vocab_bos.restype = _llama_token
    lib.llama_vocab_eos.argtypes = [ctypes.POINTER(_llama_vocab)]
    lib.llama_vocab_eos.restype = _llama_token
    lib.llama_token_to_piece.argtypes = [
        ctypes.POINTER(_llama_vocab), _llama_token,
        ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32, ctypes.c_bool,
    ]
    lib.llama_token_to_piece.restype = ctypes.c_int32

    # Batch operations
    lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    lib.llama_batch_init.restype = _llama_batch
    lib.llama_batch_free.argtypes = [_llama_batch]
    lib.llama_batch_free.restype = None
    lib.llama_decode.argtypes = [ctypes.POINTER(_llama_context), _llama_batch]
    lib.llama_decode.restype = ctypes.c_int32

    # Context operations
    lib.llama_free.argtypes = [ctypes.POINTER(_llama_context)]
    lib.llama_free.restype = None
    lib.llama_n_ctx.argtypes = [ctypes.POINTER(_llama_context)]
    lib.llama_n_ctx.restype = ctypes.c_uint32
    lib.llama_synchronize.argtypes = [ctypes.POINTER(_llama_context)]
    lib.llama_synchronize.restype = None

    # Attention
    lib.llama_set_attn_heads.argtypes = [
        ctypes.POINTER(_llama_context),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
    ]
    lib.llama_set_attn_heads.restype = None
    lib.llama_get_attn_ith.argtypes = [ctypes.POINTER(_llama_context), ctypes.c_int32]
    lib.llama_get_attn_ith.restype = ctypes.POINTER(ctypes.c_float)
    lib.llama_get_attn_n_kv.argtypes = [ctypes.POINTER(_llama_context)]
    lib.llama_get_attn_n_kv.restype = ctypes.c_int32

    # Logits
    lib.llama_get_logits_ith.argtypes = [ctypes.POINTER(_llama_context), ctypes.c_int32]
    lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

    # KV cache / memory
    lib.llama_get_memory.argtypes = [ctypes.POINTER(_llama_context)]
    lib.llama_get_memory.restype = ctypes.POINTER(_llama_memory)
    lib.llama_memory_clear.argtypes = [ctypes.POINTER(_llama_memory), ctypes.c_bool]
    lib.llama_memory_clear.restype = None
    lib.llama_memory_seq_rm.argtypes = [
        ctypes.POINTER(_llama_memory), _llama_seq_id, _llama_pos, _llama_pos,
    ]
    lib.llama_memory_seq_rm.restype = ctypes.c_bool
    lib.llama_memory_seq_cp.argtypes = [
        ctypes.POINTER(_llama_memory), _llama_seq_id, _llama_seq_id, _llama_pos, _llama_pos,
    ]
    lib.llama_memory_seq_cp.restype = None
    lib.llama_memory_seq_keep.argtypes = [ctypes.POINTER(_llama_memory), _llama_seq_id]
    lib.llama_memory_seq_keep.restype = None
    lib.llama_memory_seq_pos_max.argtypes = [ctypes.POINTER(_llama_memory), _llama_seq_id]
    lib.llama_memory_seq_pos_max.restype = _llama_pos

    # Optional: hybrid model attn-only seq_rm
    has_attn_only = hasattr(lib, "llama_memory_seq_rm_attn_only")
    if has_attn_only:
        lib.llama_memory_seq_rm_attn_only.argtypes = [
            ctypes.POINTER(_llama_memory), _llama_seq_id, _llama_pos, _llama_pos,
        ]
        lib.llama_memory_seq_rm_attn_only.restype = ctypes.c_bool

    return lib, has_attn_only


# ---------------------------------------------------------------------------
# ctypes struct definitions
# ---------------------------------------------------------------------------

class _llama_model(ctypes.Structure):
    pass

class _llama_context(ctypes.Structure):
    pass

class _llama_vocab(ctypes.Structure):
    pass

class _llama_memory(ctypes.Structure):
    pass

_llama_token = ctypes.c_int32
_llama_pos = ctypes.c_int32
_llama_seq_id = ctypes.c_int32


class _llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(_llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(_llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(_llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


class _llama_model_params(ctypes.Structure):
    _fields_ = [("_opaque", ctypes.c_uint8 * 256)]


# ---------------------------------------------------------------------------
# Context creation via C shim
# ---------------------------------------------------------------------------

_shim_cache = {}  # lib_dir -> loaded shim CDLL


def _compile_context_shim(llama_lib_dir: str) -> ctypes.CDLL:
    """Compile a small C shim to create llama_context with correct params."""
    if llama_lib_dir in _shim_cache:
        return _shim_cache[llama_lib_dir]

    shim_src = r"""
#include "llama.h"
#include <stdlib.h>

__attribute__((visibility("default")))
struct llama_context * nllw_create_ctx(
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
    # Find include directories relative to lib dir
    base_dir = os.path.dirname(llama_lib_dir)  # parent of bin/
    if os.path.basename(llama_lib_dir) == "bin":
        base_dir = os.path.dirname(llama_lib_dir)
    include_dir = os.path.join(base_dir, "include")
    ggml_include = os.path.join(base_dir, "ggml", "include")

    # Fallback: try sibling directories
    if not os.path.isdir(include_dir):
        for candidate in [
            os.path.join(base_dir, "..", "include"),
            os.path.join(base_dir, "..", "..", "include"),
        ]:
            if os.path.isdir(candidate):
                include_dir = candidate
                break

    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(shim_src)
        src_path = f.name

    ext = ".dylib" if sys.platform == "darwin" else ".so"
    shim_lib_path = os.path.join(llama_lib_dir, f"libnllw_shim{ext}")

    includes = [f"-I{include_dir}"]
    if os.path.isdir(ggml_include):
        includes.append(f"-I{ggml_include}")

    cmd = [
        "cc", "-shared", "-fPIC", "-o", shim_lib_path, src_path,
        *includes,
        f"-L{llama_lib_dir}", "-lllama",
        f"-Wl,-rpath,{llama_lib_dir}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(src_path)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to compile context shim: {result.stderr}")

    shim = ctypes.CDLL(shim_lib_path)
    shim.nllw_create_ctx.argtypes = [
        ctypes.POINTER(_llama_model), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    shim.nllw_create_ctx.restype = ctypes.POINTER(_llama_context)

    _shim_cache[llama_lib_dir] = shim
    return shim


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

# Module-level library (loaded lazily)
_lib: Optional[ctypes.CDLL] = None
_has_attn_only: bool = False
_backend_initialized: bool = False


def _ensure_lib():
    global _lib, _has_attn_only
    if _lib is None:
        _lib, _has_attn_only = _load_lib()
    return _lib


def init():
    """Initialize llama.cpp backend. Must be called before any other operations."""
    global _backend_initialized
    lib = _ensure_lib()
    lib.llama_backend_init()
    _backend_initialized = True


def cleanup():
    """Free llama.cpp backend resources."""
    global _backend_initialized
    if _lib is not None and _backend_initialized:
        _lib.llama_backend_free()
        _backend_initialized = False


class LlamaModel:
    """Wrapper around a loaded llama.cpp model."""

    def __init__(self, model_path: str, n_gpu_layers: int = 99):
        self._lib = _ensure_lib()
        if not _backend_initialized:
            init()

        params = self._lib.llama_model_default_params()
        self._ptr = self._lib.llama_model_load_from_file(model_path.encode(), params)
        if not self._ptr:
            raise RuntimeError(f"Failed to load model from {model_path}")

        self._vocab_ptr = self._lib.llama_model_get_vocab(self._ptr)
        self._n_vocab = self._lib.llama_vocab_n_tokens(self._vocab_ptr)
        self._eos_id = self._lib.llama_vocab_eos(self._vocab_ptr)
        self._n_layer = self._lib.llama_model_n_layer(self._ptr)
        self._n_head = self._lib.llama_model_n_head(self._ptr)

    @property
    def ptr(self):
        return self._ptr

    @property
    def vocab_ptr(self):
        return self._vocab_ptr

    @property
    def n_vocab(self) -> int:
        return self._n_vocab

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def n_layer(self) -> int:
        return self._n_layer

    @property
    def n_head(self) -> int:
        return self._n_head

    def tokenize(self, text: str, add_bos: bool = True, special: bool = True) -> List[int]:
        """Tokenize text into token IDs."""
        text_bytes = text.encode("utf-8")
        buf = (_llama_token * (len(text_bytes) + 64))()
        n = self._lib.llama_tokenize(
            self._vocab_ptr, text_bytes, len(text_bytes),
            buf, len(buf), add_bos, special,
        )
        if n < 0:
            buf = (_llama_token * (-n))()
            n = self._lib.llama_tokenize(
                self._vocab_ptr, text_bytes, len(text_bytes),
                buf, len(buf), add_bos, special,
            )
        return list(buf[:n])

    def token_to_piece_bytes(self, token_id: int, special: bool = True) -> bytes:
        """Convert a token ID to raw bytes."""
        buf = (ctypes.c_char * 256)()
        n = self._lib.llama_token_to_piece(self._vocab_ptr, token_id, buf, 256, 0, special)
        return buf[:n] if n > 0 else b""

    def token_to_piece(self, token_id: int, special: bool = True) -> str:
        """Convert a token ID to string (may produce mojibake for byte-fallback tokens)."""
        return self.token_to_piece_bytes(token_id, special).decode("utf-8", errors="replace")

    def tokens_to_text(self, token_ids: List[int], special: bool = True, errors: str = "replace") -> str:
        """Convert token IDs to text, handling byte-fallback tokens correctly.

        Accumulates all raw bytes and decodes once to avoid partial UTF-8 mojibake.
        Use errors="ignore" for incremental diff computation.
        """
        raw = b"".join(self.token_to_piece_bytes(t, special) for t in token_ids)
        return raw.decode("utf-8", errors=errors)

    def close(self):
        """Free model resources."""
        if self._ptr:
            self._lib.llama_model_free(self._ptr)
            self._ptr = None

    def __del__(self):
        self.close()


class LlamaContext:
    """Wrapper around a llama.cpp context with attention weight extraction."""

    def __init__(
        self,
        model: LlamaModel,
        n_ctx: int = 2048,
        n_batch: int = 2048,
        attn_weights: bool = True,
        n_gpu_layers: int = 0,
    ):
        self._lib = _ensure_lib()
        self._model = model
        self._n_ctx = n_ctx
        self._attn_weights = attn_weights

        # Find lib directory for shim compilation
        lib_path = None
        for attr in ["_name", "_handle"]:
            if hasattr(self._lib, attr):
                lib_path = getattr(self._lib, attr)
                break
        if lib_path and isinstance(lib_path, str):
            lib_dir = os.path.dirname(lib_path)
        else:
            lib_dir = os.environ.get("LLAMA_LIB_DIR", "")

        shim = _compile_context_shim(lib_dir)
        self._ptr = shim.nllw_create_ctx(
            model.ptr, n_ctx, n_batch,
            1 if attn_weights else 0,
            n_gpu_layers,
        )
        if not self._ptr:
            raise RuntimeError("Failed to create llama_context")

        self._mem = self._lib.llama_get_memory(self._ptr)
        self._num_attn_heads = 0

    @property
    def ptr(self):
        return self._ptr

    @property
    def n_ctx(self) -> int:
        return self._lib.llama_n_ctx(self._ptr)

    def set_attn_heads(self, layers: List[int], heads: List[int]):
        """Set which attention heads to extract weights from."""
        n = len(layers)
        assert len(heads) == n, "layers and heads must have same length"
        l_arr = (ctypes.c_int32 * n)(*layers)
        h_arr = (ctypes.c_int32 * n)(*heads)
        self._lib.llama_set_attn_heads(self._ptr, l_arr, h_arr, n)
        self._num_attn_heads = n

    def decode_batch(self, tokens: List[int], pos_start: int = 0,
                     seq_id: int = 0, output_last_only: bool = True) -> int:
        """Decode a batch of tokens starting at pos_start. Returns status code."""
        n = len(tokens)
        if n == 0:
            return 0
        batch = self._lib.llama_batch_init(n, 0, 1)
        for i in range(n):
            batch.token[i] = tokens[i]
            batch.pos[i] = pos_start + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = seq_id
            batch.logits[i] = 1 if (not output_last_only or i == n - 1) else 0
        batch.n_tokens = n
        ret = self._lib.llama_decode(self._ptr, batch)
        self._lib.llama_batch_free(batch)
        return ret

    def decode_single(self, token: int, pos: int, seq_id: int = 0,
                      output: bool = True) -> int:
        """Decode a single token at a specific position. Returns status code."""
        batch = self._lib.llama_batch_init(1, 0, 1)
        batch.token[0] = token
        batch.pos[0] = pos
        batch.n_seq_id[0] = 1
        batch.seq_id[0][0] = seq_id
        batch.logits[0] = 1 if output else 0
        batch.n_tokens = 1
        ret = self._lib.llama_decode(self._ptr, batch)
        self._lib.llama_batch_free(batch)
        return ret

    def get_attn_weights(self, token_idx: int, n_pairs: int) -> Optional[np.ndarray]:
        """Get attention weights for a given output token.

        Returns numpy array of shape (n_pairs, n_kv) or None.
        """
        ptr = self._lib.llama_get_attn_ith(self._ptr, token_idx)
        if not ptr:
            return None

        n_kv = self._lib.llama_get_attn_n_kv(self._ptr)
        if n_kv <= 0:
            return None

        n_ctx = self.n_ctx
        result = np.zeros((n_pairs, n_kv), dtype=np.float32)
        for p in range(n_pairs):
            offset = p * n_ctx
            arr = (ctypes.c_float * n_kv).from_address(
                ctypes.addressof(ptr.contents) + offset * 4
            )
            result[p] = np.frombuffer(arr, dtype=np.float32)

        return result

    def argmax_logits(self, token_idx: int) -> int:
        """Get the argmax of logits for a given output token."""
        ptr = self._lib.llama_get_logits_ith(self._ptr, token_idx)
        if not ptr:
            return -1
        n_vocab = self._model.n_vocab
        logits = (ctypes.c_float * n_vocab).from_address(ctypes.addressof(ptr.contents))
        return int(np.argmax(np.frombuffer(logits, dtype=np.float32)))

    def get_logits_array(self, token_idx: int) -> Optional[np.ndarray]:
        """Get raw logits as a numpy array for a given output token."""
        ptr = self._lib.llama_get_logits_ith(self._ptr, token_idx)
        if not ptr:
            return None
        n_vocab = self._model.n_vocab
        logits = (ctypes.c_float * n_vocab).from_address(ctypes.addressof(ptr.contents))
        return np.frombuffer(logits, dtype=np.float32).copy()

    # --- KV cache management ---

    def memory_clear(self, data: bool = True):
        """Clear all KV cache data."""
        self._lib.llama_memory_clear(self._mem, data)

    def memory_seq_rm(self, seq_id: int, p0: int, p1: int) -> bool:
        """Remove tokens in position range [p0, p1) from a sequence.

        seq_id < 0: match any sequence. p0 < 0: [0, p1]. p1 < 0: [p0, inf).
        Returns True on success, False if the memory type doesn't support partial removal.
        """
        return self._lib.llama_memory_seq_rm(self._mem, seq_id, p0, p1)

    def memory_seq_rm_attn_only(self, seq_id: int, p0: int, p1: int) -> bool:
        """Remove tokens from attention (KV) cache only, keeping recurrent state.

        For hybrid models (e.g. Qwen3.5) where seq_rm fails on recurrent state.
        Falls back to regular seq_rm for non-hybrid models.
        """
        if _has_attn_only:
            return self._lib.llama_memory_seq_rm_attn_only(self._mem, seq_id, p0, p1)
        return self.memory_seq_rm(seq_id, p0, p1)

    def memory_seq_cp(self, src_id: int, dst_id: int, p0: int, p1: int):
        """Copy KV cache from src_id to dst_id for position range [p0, p1)."""
        self._lib.llama_memory_seq_cp(self._mem, src_id, dst_id, p0, p1)

    def memory_seq_keep(self, seq_id: int):
        """Remove all tokens NOT belonging to seq_id."""
        self._lib.llama_memory_seq_keep(self._mem, seq_id)

    def memory_seq_pos_max(self, seq_id: int) -> int:
        """Get max position in sequence. Returns -1 if empty."""
        return self._lib.llama_memory_seq_pos_max(self._mem, seq_id)

    def close(self):
        """Free context resources."""
        if self._ptr:
            self._lib.llama_free(self._ptr)
            self._ptr = None

    def __del__(self):
        self.close()


class LlamaBackend:
    """Convenience wrapper that manages model + context lifecycle.

    For simple use cases where you just need one model with one context.
    For more control, use LlamaModel and LlamaContext directly.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 99,
        attn_weights: bool = True,
        attn_heads: Optional[List[Tuple[int, int]]] = None,
    ):
        init()
        self.model = LlamaModel(model_path, n_gpu_layers=n_gpu_layers)
        self.ctx = LlamaContext(
            self.model,
            n_ctx=n_ctx,
            n_batch=n_ctx,
            attn_weights=attn_weights,
            n_gpu_layers=n_gpu_layers,
        )
        if attn_heads:
            layers, heads = zip(*attn_heads)
            self.ctx.set_attn_heads(list(layers), list(heads))

    # Delegate common operations
    def tokenize(self, text: str, **kwargs) -> List[int]:
        return self.model.tokenize(text, **kwargs)

    def tokens_to_text(self, token_ids: List[int], **kwargs) -> str:
        return self.model.tokens_to_text(token_ids, **kwargs)

    def decode_batch(self, tokens: List[int], **kwargs) -> int:
        return self.ctx.decode_batch(tokens, **kwargs)

    def decode_single(self, token: int, pos: int, **kwargs) -> int:
        return self.ctx.decode_single(token, pos, **kwargs)

    def get_attn_weights(self, token_idx: int, n_pairs: int) -> Optional[np.ndarray]:
        return self.ctx.get_attn_weights(token_idx, n_pairs)

    def argmax_logits(self, token_idx: int) -> int:
        return self.ctx.argmax_logits(token_idx)

    def get_logits_array(self, token_idx: int) -> Optional[np.ndarray]:
        return self.ctx.get_logits_array(token_idx)

    def close(self):
        self.ctx.close()
        self.model.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
