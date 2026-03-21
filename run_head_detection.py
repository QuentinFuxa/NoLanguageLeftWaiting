"""Detect dedicated alignment heads for EN-DE, EN-IT, CS-EN.

While cross-lingual transfer works (>97% TS mass), dedicated heads may improve quality.

Run on A40:
    export LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so
    nohup python3 run_head_detection.py > head_detection_results.log 2>&1 &
"""
import subprocess
import sys
import os
import time

os.environ["LLAMA_CPP_LIB"] = "/home/fuxa/llama.cpp/build/bin/libllama.so"

MODEL = "/home/fuxa/HY-MT1.5-7B.Q8_0.gguf"
CWD = "/home/fuxa/nllw_deploy"

# Directions that currently fall back to EN-ZH heads
directions = ["en-de", "en-it", "cs-en"]

for lang in directions:
    print(f"\n{'='*60}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] Detecting heads for {lang}", flush=True)
    print(f"{'='*60}", flush=True)

    cmd = [
        sys.executable, "-m", "nllw.detect_heads",
        "--model", MODEL,
        "--lang", lang,
        "--n-gpu-layers", "99",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=CWD)
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Completed {lang} in {elapsed:.0f}s", flush=True)

print(f"\nAll head detection complete at {time.strftime('%Y-%m-%d %H:%M:%S')}")
