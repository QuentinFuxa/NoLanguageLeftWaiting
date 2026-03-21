"""XCOMET-XL scoring via separate process: avoids OOM by not co-loading models.

Step 1: Translate with the MT model and save hypotheses to JSON
Step 2: Score the saved hypotheses with XCOMET-XL in a fresh process (no MT model)

This avoids the OOM issue where MT model (8GB) + XCOMET-XL (12GB) exceed A40 VRAM.

Run on A40:
    export LLAMA_CPP_LIB=/home/fuxa/llama.cpp/build/bin/libllama.so
    python3 run_xcomet_separate.py

Expected runtime: ~30 minutes
"""
import subprocess
import sys
import os
import json
import time

os.environ["LLAMA_CPP_LIB"] = "/home/fuxa/llama.cpp/build/bin/libllama.so"

MODEL = "/home/fuxa/HY-MT1.5-7B.Q8_0.gguf"
N_GPU = "99"
CWD = "/home/fuxa/nllw_deploy"
HYPO_DIR = "/home/fuxa/nllw_deploy/xcomet_hypotheses"

# Best configs per direction (iteration 17 verified at 100 sentences)
CONFIGS = [
    {"lang": "en-zh", "bd": 3, "wb": 4, "agg": "top_p"},   # COMET=0.892
    {"lang": "en-de", "bd": 2, "wb": 3, "agg": "top_p"},   # COMET=0.881
    {"lang": "en-it", "bd": 2, "wb": 3, "agg": "top_p"},   # COMET=0.890
    {"lang": "cs-en", "bd": 3, "wb": 3, "agg": "top_p"},   # COMET=0.877
]


def step1_translate(n=100):
    """Step 1: Translate and save hypotheses to JSON files."""
    print("=" * 60)
    print("STEP 1: Translate with MT model (save hypotheses)")
    print("=" * 60)

    os.makedirs(HYPO_DIR, exist_ok=True)

    for cfg in CONFIGS:
        lang = cfg["lang"]
        hypo_file = os.path.join(HYPO_DIR, f"hypo_{lang.replace('-', '_')}.json")

        print(f"\n[{time.strftime('%H:%M:%S')}] Translating {lang} (n={n})...", flush=True)

        # Use bench.py with --save-hypotheses to save for offline XCOMET scoring
        cmd = [
            sys.executable, "-m", "nllw.bench",
            "--model", MODEL,
            "--lang", lang,
            "--n-gpu-layers", N_GPU,
            "-n", str(n),
            "--border-distance", str(cfg["bd"]),
            "--word-batch", str(cfg["wb"]),
            "--aggregation", cfg["agg"],
            "--comet",  # Also compute COMET for comparison
            "--save-hypotheses", hypo_file,
        ]

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=CWD)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.0f}s (rc={result.returncode})", flush=True)

    # Also full-sentence baselines
    for lang in ["en-zh", "en-de", "en-it", "cs-en"]:
        hypo_file = os.path.join(HYPO_DIR, f"hypo_fullsent_{lang.replace('-', '_')}.json")
        print(f"\n[{time.strftime('%H:%M:%S')}] Full-sentence {lang} (n={n})...", flush=True)

        cmd = [
            sys.executable, "-m", "nllw.bench",
            "--model", MODEL,
            "--lang", lang,
            "--n-gpu-layers", N_GPU,
            "-n", str(n),
            "--backend", "full-sentence",
            "--comet",
            "--save-hypotheses", hypo_file,
        ]

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=CWD)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.0f}s (rc={result.returncode})", flush=True)


def step2_score_xcomet():
    """Step 2: Score saved hypotheses with XCOMET-XL (separate process, no MT model)."""
    print("\n" + "=" * 60)
    print("STEP 2: Score with XCOMET-XL (no MT model loaded)")
    print("=" * 60)

    # This script runs as a subprocess -- it only loads XCOMET-XL
    score_script = '''
import json
import os
import sys
import time

HYPO_DIR = "{hypo_dir}"

try:
    from comet import download_model, load_from_checkpoint
except ImportError:
    print("ERROR: comet not installed", file=sys.stderr)
    sys.exit(1)

print("Loading XCOMET-XL model...", flush=True)
t0 = time.time()
model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)
print(f"  Loaded in {{time.time()-t0:.1f}}s", flush=True)

# Score each hypothesis file
for fname in sorted(os.listdir(HYPO_DIR)):
    if not fname.endswith(".json"):
        continue

    path = os.path.join(HYPO_DIR, fname)
    with open(path) as f:
        data = json.load(f)

    sources = data.get("sources", [])
    hypotheses = data.get("hypotheses", [])
    references = data.get("references", [])

    if not sources or not hypotheses or not references:
        print(f"  Skipping {{fname}} (missing data)", flush=True)
        continue

    print(f"\\n[{{time.strftime('%H:%M:%S')}}] Scoring {{fname}} ({{len(sources)}} sentences)...", flush=True)

    samples = [{{"src": s, "mt": h, "ref": r}} for s, h, r in zip(sources, hypotheses, references)]
    output = model.predict(samples, batch_size=8, gpus=1)
    score = output.system_score

    print(f"  XCOMET-XL = {{score:.4f}}", flush=True)

    # Save score back
    data["xcomet_xl"] = score
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

print("\\nXCOMET-XL scoring complete!", flush=True)
'''.format(hypo_dir=HYPO_DIR)

    # Write temporary script
    script_path = os.path.join(CWD, "_xcomet_scorer.py")
    with open(script_path, "w") as f:
        f.write(score_script)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False, text=True, cwd=CWD,
    )
    elapsed = time.time() - t0
    print(f"\nXCOMET-XL scoring completed in {elapsed:.0f}s (rc={result.returncode})", flush=True)

    # Cleanup
    os.remove(script_path)


if __name__ == "__main__":
    print("=" * 60)
    print("XCOMET-XL Separate Process Scoring")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    step1_translate(n=100)
    step2_score_xcomet()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for fname in sorted(os.listdir(HYPO_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(HYPO_DIR, fname)) as f:
            data = json.load(f)
        xcomet = data.get("xcomet_xl", "N/A")
        comet = data.get("comet", "N/A")
        print(f"  {fname}: XCOMET-XL={xcomet}, COMET={comet}")

    print(f"\nCompleted: {time.strftime('%Y-%m-%d %H:%M:%S')}")
