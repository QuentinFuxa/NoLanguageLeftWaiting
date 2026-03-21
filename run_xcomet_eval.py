"""XCOMET-XL evaluation for best configs per direction.

This scores the same configs with XCOMET-XL (more discriminative than wmt22-comet-da).
Important for comparison with iwslt26-sst results (best: 0.842 EN-ZH with XCOMET-XL).

Note: XCOMET-XL needs ~12GB VRAM. A40 has 46GB, so it fits alongside Q8_0 7B model.
However, to avoid VRAM issues, we free the model between translation and scoring.

Run on A40 (after main experiments complete):
    export LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so
    nohup python3 run_xcomet_eval.py > xcomet_results.log 2>&1 &
"""
import subprocess
import sys
import os
import time

os.environ["LLAMA_CPP_LIB"] = "/home/fuxa/llama.cpp/build/bin/libllama.so"

MODEL = "/home/fuxa/HY-MT1.5-7B.Q8_0.gguf"
N_GPU = "99"
CWD = "/home/fuxa/nllw_deploy"


def run_xcomet(lang, bd, wb, n=50, label=""):
    """Run benchmark with XCOMET-XL scoring."""
    cmd = [
        sys.executable, "-m", "nllw.bench",
        "--model", MODEL,
        "--lang", lang,
        "--n-gpu-layers", N_GPU,
        "-n", str(n),
        "--border-distance", str(bd),
        "--word-batch", str(wb),
        "--xcomet",
    ]
    tag = label or f"XCOMET-XL bd={bd} wb={wb} lang={lang} n={n}"
    print(f"\n{'='*60}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] {tag}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=CWD)
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Completed in {elapsed:.0f}s (rc={result.returncode})",
          flush=True)
    return result.returncode


# Best configs per direction (from iteration 15 parameter sweep)
configs = [
    {"lang": "en-zh", "bd": 2, "wb": 3},   # COMET=0.879
    {"lang": "en-de", "bd": 3, "wb": 3},   # COMET=0.853
    {"lang": "en-it", "bd": 3, "wb": 3},   # COMET=0.864
    {"lang": "cs-en", "bd": 3, "wb": 2},   # COMET=0.853
]

# Also test with COMET for direct comparison
print("=" * 60)
print("XCOMET-XL Evaluation (50 sentences, best configs)")
print("=" * 60)

for cfg in configs:
    # First run with COMET for baseline comparison
    cmd_comet = [
        sys.executable, "-m", "nllw.bench",
        "--model", MODEL,
        "--lang", cfg["lang"],
        "--n-gpu-layers", N_GPU,
        "-n", "50",
        "--border-distance", str(cfg["bd"]),
        "--word-batch", str(cfg["wb"]),
        "--comet",
    ]
    print(f"\n{'='*60}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] COMET baseline: {cfg['lang']} bd={cfg['bd']} wb={cfg['wb']}", flush=True)
    print(f"{'='*60}", flush=True)
    subprocess.run(cmd_comet, capture_output=False, text=True, cwd=CWD)

    # Then XCOMET-XL
    run_xcomet(**cfg, n=50)

# Full-sentence XCOMET-XL baselines (quality upper bound)
print("\n" + "=" * 60)
print("Full-sentence XCOMET-XL baselines")
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


print(f"\nXCOMET-XL evaluation complete at {time.strftime('%Y-%m-%d %H:%M:%S')}")
