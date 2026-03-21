"""Top-p threshold tuning: find optimal p_threshold for each direction.

The top_p aggregation method uses p_threshold=0.8 by default. This was never
tuned. Given that top_p gives +0.008-0.021 COMET, even small threshold
improvements could be significant.

Also tests top_p_weighted variant (weighted mean instead of rightmost position).

Run on A40 AFTER iteration 17 main experiments complete:
    export LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so
    nohup python3 run_topp_tuning.py > topp_tuning_results.log 2>&1 &

Expected runtime: ~1.5 hours
"""
import subprocess
import sys
import os
import time

os.environ["LLAMA_CPP_LIB"] = "/home/fuxa/llama.cpp/build/bin/libllama.so"

MODEL = "/home/fuxa/HY-MT1.5-7B.Q8_0.gguf"
N_GPU = "99"
CWD = "/home/fuxa/nllw_deploy"


def run_bench(lang, bd, wb, n=50, agg="top_p", topp=0.8, label=""):
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
        "--top-p-threshold", str(topp),
        "--comet",
    ]

    tag = label or f"{lang} bd={bd} wb={wb} agg={agg} topp={topp} n={n}"
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
print("Top-p Threshold Tuning")
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ============================================================
# Phase 1: top_p threshold sweep for EN-ZH (best direction)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: top_p threshold sweep EN-ZH (bd=3, wb=4)")
print("=" * 60)

for topp in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    run_bench("en-zh", 3, 4, n=50, agg="top_p", topp=topp)

# ============================================================
# Phase 2: top_p_weighted variant test
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: top_p_weighted variant (50 sentences)")
print("=" * 60)

for lang, bd, wb in [("en-zh", 3, 4), ("en-de", 2, 3), ("en-it", 2, 3), ("cs-en", 3, 3)]:
    run_bench(lang, bd, wb, n=50, agg="top_p_weighted", topp=0.8)

# ============================================================
# Phase 3: Best threshold applied to all directions
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: Threshold sweep for other directions")
print("=" * 60)

# Test the extremes + default for each direction
for lang, bd, wb in [("en-de", 2, 3), ("en-it", 2, 3), ("cs-en", 3, 3)]:
    for topp in [0.6, 0.75, 0.8, 0.9]:
        run_bench(lang, bd, wb, n=50, agg="top_p", topp=topp)

print(f"\n{'='*60}")
print(f"Top-p tuning completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
