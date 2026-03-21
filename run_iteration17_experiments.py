"""Iteration 17 GPU experiments: 100-sentence verification + XCOMET-XL scoring.

Goals:
1. Verify top_p + wb=4 best configs at 100 sentences (currently only 50)
2. Score with XCOMET-XL (more discriminative, needed for IWSLT comparison)
3. Test wb=6 to see if the wb quality trend continues
4. Test top_p sigma parameter variants

Run on A40:
    export LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so
    nohup python3 run_iteration17_experiments.py > iter17_results.log 2>&1 &

Expected runtime: ~2-3 hours
"""
import subprocess
import sys
import os
import time
import json

os.environ["LLAMA_CPP_LIB"] = "/home/fuxa/llama.cpp/build/bin/libllama.so"

MODEL = "/home/fuxa/HY-MT1.5-7B.Q8_0.gguf"
N_GPU = "99"
CWD = "/home/fuxa/nllw_deploy"


def run_bench(lang, bd, wb, n=100, agg="ts_vote", extra_args=None, label=""):
    """Run a single benchmark configuration."""
    cmd = [
        sys.executable, "-m", "nllw.bench",
        "--model", MODEL,
        "--lang", lang,
        "--n-gpu-layers", N_GPU,
        "-n", str(n),
        "--border-distance", str(bd),
        "--word-batch", str(wb),
        "--aggregation", agg,
        "--comet",
    ]
    if extra_args:
        cmd.extend(extra_args)

    tag = label or f"{lang} bd={bd} wb={wb} agg={agg} n={n}"
    print(f"\n{'='*60}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] {tag}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=CWD)
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Completed in {elapsed:.0f}s (rc={result.returncode})",
          flush=True)
    return result.returncode


def run_xcomet(lang, bd, wb, n=50, agg="ts_vote", label=""):
    """Run benchmark with XCOMET-XL scoring (separate from COMET to manage VRAM)."""
    cmd = [
        sys.executable, "-m", "nllw.bench",
        "--model", MODEL,
        "--lang", lang,
        "--n-gpu-layers", N_GPU,
        "-n", str(n),
        "--border-distance", str(bd),
        "--word-batch", str(wb),
        "--aggregation", agg,
        "--xcomet",
    ]
    tag = label or f"XCOMET-XL {lang} bd={bd} wb={wb} agg={agg} n={n}"
    print(f"\n{'='*60}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] {tag}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=CWD)
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Completed in {elapsed:.0f}s (rc={result.returncode})",
          flush=True)
    return result.returncode


print("=" * 60)
print("NLLW Iteration 17 Experiments")
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ============================================================
# Phase 1: 100-sentence verification of best configs with top_p
# These were only tested at 50 sentences in iteration 16
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: 100-sentence verification of top_p + optimal wb/bd")
print("=" * 60)

phase1_configs = [
    # Best quality configs (from iter 16 at 50 sentences)
    {"lang": "en-zh", "bd": 3, "wb": 4, "agg": "top_p"},   # COMET=0.895 @ 50
    {"lang": "en-zh", "bd": 3, "wb": 5, "agg": "top_p"},   # Check if wb=5+top_p improves
    {"lang": "en-zh", "bd": 2, "wb": 3, "agg": "top_p"},   # COMET=0.890 @ 50 (balanced)
    {"lang": "en-de", "bd": 2, "wb": 3, "agg": "top_p"},   # COMET=0.881 @ 50
    {"lang": "en-de", "bd": 2, "wb": 4, "agg": "top_p"},   # Test wb=4 for EN-DE
    {"lang": "en-it", "bd": 2, "wb": 3, "agg": "top_p"},   # COMET=0.884 @ 50
    {"lang": "en-it", "bd": 2, "wb": 4, "agg": "top_p"},   # Test wb=4 for EN-IT
    {"lang": "cs-en", "bd": 3, "wb": 3, "agg": "top_p"},   # COMET=0.876 @ 50
    {"lang": "cs-en", "bd": 3, "wb": 4, "agg": "top_p"},   # Test wb=4 for CS-EN
]

for cfg in phase1_configs:
    run_bench(**cfg, n=100)

# ============================================================
# Phase 2: wb=6 exploration (does quality keep improving?)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: wb=6 exploration")
print("=" * 60)

phase2_configs = [
    {"lang": "en-zh", "bd": 3, "wb": 6, "agg": "top_p"},
    {"lang": "en-de", "bd": 2, "wb": 5, "agg": "top_p"},
    {"lang": "en-it", "bd": 2, "wb": 5, "agg": "top_p"},
]

for cfg in phase2_configs:
    run_bench(**cfg, n=50)

# ============================================================
# Phase 3: Dedicated head configs vs cross-lingual transfer
# Use direction-specific heads where available
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: Dedicated heads vs cross-lingual (top_p, 50 sent)")
print("=" * 60)

# These use auto head discovery which should find dedicated configs
# Compare vs explicit en-zh heads
for lang, bd, wb in [("en-de", 2, 3), ("en-it", 2, 3)]:
    # Default (auto-discovers dedicated heads)
    run_bench(lang, bd, wb, n=50, agg="top_p",
              label=f"AUTO-HEADS {lang} bd={bd} wb={wb} top_p")

# ============================================================
# Phase 4: XCOMET-XL scoring (50 sentences, best configs only)
# Uses backend.close() + gc + torch.cuda.empty_cache() to free VRAM
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: XCOMET-XL scoring (best configs)")
print("=" * 60)

xcomet_configs = [
    {"lang": "en-zh", "bd": 3, "wb": 4, "agg": "top_p"},
    {"lang": "en-de", "bd": 2, "wb": 3, "agg": "top_p"},
    {"lang": "en-it", "bd": 2, "wb": 3, "agg": "top_p"},
    {"lang": "cs-en", "bd": 3, "wb": 3, "agg": "top_p"},
]

for cfg in xcomet_configs:
    rc = run_xcomet(**cfg, n=50)
    if rc == -9:
        print(f"  WARNING: OOM (rc=-9) for {cfg['lang']}. "
              "Try reducing n or running XCOMET separately.", flush=True)

# Full-sentence XCOMET-XL baselines
print("\n" + "=" * 60)
print("PHASE 4b: Full-sentence XCOMET-XL baselines")
print("=" * 60)

for lang in ["en-zh", "en-de", "en-it", "cs-en"]:
    cmd = [
        sys.executable, "-m", "nllw.bench",
        "--model", MODEL,
        "--lang", lang,
        "--n-gpu-layers", N_GPU,
        "-n", "50",
        "--backend", "full-sentence",
        "--xcomet",
    ]
    print(f"\n{'='*60}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] Full-sentence XCOMET-XL: {lang}", flush=True)
    print(f"{'='*60}", flush=True)
    subprocess.run(cmd, capture_output=False, text=True, cwd=CWD)

# ============================================================
# Phase 5: Repetition halt safety check
# Verify that repetition_max_repeats=2 doesn't hurt quality
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: Repetition halt safety check (50 sentences)")
print("=" * 60)

for lang, bd, wb in [("en-zh", 3, 4), ("en-de", 2, 3)]:
    # Without repetition halt
    run_bench(lang, bd, wb, n=50, agg="top_p",
              label=f"NO-REP {lang} bd={bd} wb={wb} top_p")
    # With repetition halt
    run_bench(lang, bd, wb, n=50, agg="top_p",
              extra_args=["--repetition-halt", "2"],
              label=f"REP=2 {lang} bd={bd} wb={wb} top_p")

print(f"\n{'='*60}")
print(f"All experiments completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
