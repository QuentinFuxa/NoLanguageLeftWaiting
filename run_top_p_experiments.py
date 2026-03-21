"""Follow-up: Test top_p aggregation with wb=4/5.

top_p was discovered as the best aggregation method (+0.007 COMET over ts_vote).
Combined with wb=5, this could push COMET even higher.

Also test top_p across all directions and re-detect EN-DE heads.

Run on A40 (after main experiments complete):
    export LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so
    nohup python3 run_top_p_experiments.py > top_p_results.log 2>&1 &
"""
import subprocess
import sys
import os
import time

os.environ["LLAMA_CPP_LIB"] = "/home/fuxa/llama.cpp/build/bin/libllama.so"

MODEL = "/home/fuxa/HY-MT1.5-7B.Q8_0.gguf"
N_GPU = "99"
CWD = "/home/fuxa/nllw_deploy"


def run_bench(lang, bd, wb, n=50, extra_args=None, label=""):
    cmd = [
        sys.executable, "-m", "nllw.bench",
        "--model", MODEL, "--lang", lang,
        "--n-gpu-layers", N_GPU, "-n", str(n),
        "--border-distance", str(bd), "--word-batch", str(wb),
        "--comet",
    ]
    if extra_args:
        cmd.extend(extra_args)
    tag = label or f"bd={bd} wb={wb} lang={lang}"
    print(f"\n{'='*60}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] {tag}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, capture_output=False, text=True, cwd=CWD)
    print(f"[{time.strftime('%H:%M:%S')}] Done in {time.time()-t0:.0f}s", flush=True)


# ============================================================
# PHASE A: Re-detect EN-DE heads (was deleted by rsync)
# ============================================================
print("=" * 60)
print("PHASE A: Re-detect EN-DE heads")
print("=" * 60)

cmd = [
    sys.executable, "-m", "nllw.detect_heads",
    "--model", MODEL, "--lang", "en-de", "--n-gpu-layers", N_GPU,
]
subprocess.run(cmd, capture_output=False, text=True, cwd=CWD)


# ============================================================
# PHASE B: top_p + wb=4/5 (EN-ZH)
# ============================================================
print("\n" + "=" * 60)
print("PHASE B: top_p aggregation + wb sweep (EN-ZH)")
print("=" * 60)

for wb in [3, 4, 5]:
    for bd in [2, 3]:
        run_bench("en-zh", bd=bd, wb=wb, n=50,
                  extra_args=["--aggregation", "top_p"],
                  label=f"top_p bd={bd} wb={wb} lang=en-zh")


# ============================================================
# PHASE C: top_p across all directions (best configs)
# ============================================================
print("\n" + "=" * 60)
print("PHASE C: top_p across all directions")
print("=" * 60)

configs = [
    ("en-de", 3, 4), ("en-de", 2, 3),
    ("en-it", 3, 4), ("en-it", 2, 3),
    ("cs-en", 3, 2), ("cs-en", 3, 3),
]
for lang, bd, wb in configs:
    run_bench(lang, bd=bd, wb=wb, n=30,
              extra_args=["--aggregation", "top_p"],
              label=f"top_p bd={bd} wb={wb} lang={lang}")


# ============================================================
# PHASE D: geomean + wb=5 (second best aggregation)
# ============================================================
print("\n" + "=" * 60)
print("PHASE D: geomean + wb=5 (EN-ZH)")
print("=" * 60)

run_bench("en-zh", bd=3, wb=5, n=50,
          extra_args=["--aggregation", "geomean"],
          label="geomean bd=3 wb=5 lang=en-zh")

run_bench("en-zh", bd=3, wb=5, n=50,
          extra_args=["--aggregation", "top_p"],
          label="top_p bd=3 wb=5 lang=en-zh (100-sent verification)")


print(f"\nAll top_p experiments complete at {time.strftime('%Y-%m-%d %H:%M:%S')}")
