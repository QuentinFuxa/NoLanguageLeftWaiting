"""Iteration 16 experiments: larger evaluations, wb=4/5, per-direction sweeps.

Run on A40:
    export LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so
    nohup python3 run_iteration16_experiments.py > iter16_results.log 2>&1 &
"""
import subprocess
import sys
import os
import time

os.environ["LLAMA_CPP_LIB"] = "/home/fuxa/llama.cpp/build/bin/libllama.so"

MODEL = "/home/fuxa/HY-MT1.5-7B.Q8_0.gguf"
N_GPU = "99"
CWD = "/home/fuxa/nllw_deploy"


def run_bench(lang, bd, wb, n=100, extra_args=None, label=""):
    """Run a single benchmark configuration."""
    cmd = [
        sys.executable, "-m", "nllw.bench",
        "--model", MODEL,
        "--lang", lang,
        "--n-gpu-layers", N_GPU,
        "-n", str(n),
        "--border-distance", str(bd),
        "--word-batch", str(wb),
        "--comet",
    ]
    if extra_args:
        cmd.extend(extra_args)

    tag = label or f"bd={bd} wb={wb} lang={lang} n={n}"
    print(f"\n{'='*60}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] {tag}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=CWD)
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Completed in {elapsed:.0f}s", flush=True)
    return result.returncode


# ============================================================
# PHASE 1: 100-sentence evaluations with best known configs
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: 100-sentence evaluations (best configs per direction)")
print("=" * 60)

phase1 = [
    # Best configs from iteration 15 (30-sentence results)
    {"lang": "en-zh", "bd": 2, "wb": 3},  # COMET=0.879
    {"lang": "en-de", "bd": 3, "wb": 3},  # COMET=0.853
    {"lang": "en-it", "bd": 3, "wb": 3},  # COMET=0.864
    {"lang": "cs-en", "bd": 3, "wb": 2},  # COMET=0.853
]

for cfg in phase1:
    run_bench(**cfg, n=100)


# ============================================================
# PHASE 2: Test wb=4 and wb=5 (does the trend continue?)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Test wb=4 and wb=5 (EN-ZH)")
print("=" * 60)

phase2_enzh = [
    {"lang": "en-zh", "bd": 2, "wb": 4},
    {"lang": "en-zh", "bd": 3, "wb": 4},
    {"lang": "en-zh", "bd": 2, "wb": 5},
    {"lang": "en-zh", "bd": 3, "wb": 5},
    {"lang": "en-zh", "bd": 1, "wb": 3},  # Also test bd=1 (aggressive)
    {"lang": "en-zh", "bd": 1, "wb": 4},
]

for cfg in phase2_enzh:
    run_bench(**cfg, n=50)

# wb=4 for other directions if it helps EN-ZH
phase2_multi = [
    {"lang": "en-de", "bd": 3, "wb": 4},
    {"lang": "en-de", "bd": 2, "wb": 3},  # Try lower bd like EN-ZH
    {"lang": "en-it", "bd": 3, "wb": 4},
    {"lang": "en-it", "bd": 2, "wb": 3},
    {"lang": "cs-en", "bd": 3, "wb": 3},  # Try wb=3 for CS-EN
    {"lang": "cs-en", "bd": 2, "wb": 3},
]

for cfg in phase2_multi:
    run_bench(**cfg, n=50)


# ============================================================
# PHASE 3: Full-sentence baseline (quality upper bound)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: Full-sentence baseline (quality upper bound)")
print("=" * 60)

for lang in ["en-zh", "en-de", "en-it", "cs-en"]:
    cmd = [
        sys.executable, "-m", "nllw.bench",
        "--model", MODEL,
        "--lang", lang,
        "--n-gpu-layers", N_GPU,
        "-n", "50",
        "--backend", "full-sentence",
        "--comet",
    ]
    print(f"\n{'='*60}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] Full-sentence baseline: {lang}", flush=True)
    print(f"{'='*60}", flush=True)
    subprocess.run(cmd, capture_output=False, text=True, cwd=CWD)


# ============================================================
# PHASE 4: Aggregation method comparison (EN-ZH, best config)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: Aggregation methods (EN-ZH, bd=2, wb=3)")
print("=" * 60)

aggregations = [
    "ts_vote", "softmax_mean", "entropy_weighted", "consensus",
    "geomean", "top_p", "gaussian_kernel",
]

for agg in aggregations:
    run_bench("en-zh", bd=2, wb=3, n=30,
              extra_args=["--aggregation", agg],
              label=f"agg={agg} lang=en-zh bd=2 wb=3")


# ============================================================
# PHASE 5: Context injection test (Qwen-style models benefit)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: Context injection test (HY-MT, all directions)")
print("=" * 60)

for lang, bd, wb in [("en-zh", 2, 3), ("en-de", 3, 3), ("en-it", 3, 3), ("cs-en", 3, 2)]:
    run_bench(lang, bd=bd, wb=wb, n=30,
              extra_args=["--context-sentences", "1"],
              label=f"ctx=1 lang={lang} bd={bd} wb={wb}")
    run_bench(lang, bd=bd, wb=wb, n=30,
              extra_args=["--context-sentences", "2"],
              label=f"ctx=2 lang={lang} bd={bd} wb={wb}")


# ============================================================
# PHASE 6: Entropy veto threshold
# ============================================================
print("\n" + "=" * 60)
print("PHASE 6: Entropy veto sweep (EN-ZH)")
print("=" * 60)

for ent in ["0.5", "0.75", "1.0", "1.5"]:
    run_bench("en-zh", bd=2, wb=3, n=30,
              extra_args=["--entropy-threshold", ent],
              label=f"entropy_veto={ent} lang=en-zh bd=2 wb=3")


print("\n" + "=" * 60)
print(f"ALL EXPERIMENTS COMPLETE at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
