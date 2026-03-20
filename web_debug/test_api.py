"""Quick API test script for iterating on AlignAtt translation quality.

Usage:
    python web_debug/test_api.py

Requires server running: python web_debug/server.py
"""

import requests
import json
import sys
import time

BASE = "http://localhost:8777"

def status():
    r = requests.get(f"{BASE}/status")
    return r.json()

def load_model(model_path=None, target_lang="fr", source_lang="en"):
    body = {"target_lang": target_lang, "source_lang": source_lang}
    if model_path:
        body["model_path"] = model_path
    r = requests.post(f"{BASE}/load", json=body)
    return r.json()

def translate(text):
    r = requests.post(f"{BASE}/translate", json={"text": text})
    return r.json()

def finish():
    r = requests.post(f"{BASE}/finish")
    return r.json()

def reset():
    r = requests.post(f"{BASE}/reset")
    return r.json()

def set_lang(lang):
    r = requests.post(f"{BASE}/set_lang", json={"lang": lang})
    return r.json()


def test_sentence(words, target_lang="fr", source_lang="en", label=""):
    """Feed words one by one (simulating typing with spaces) and show results."""
    reset()

    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    source = " ".join(words)
    print(f"  Source: \"{source}\"")
    print(f"  Target: {target_lang}")
    print(f"  {'─'*56}")

    committed_text = ""

    for i, word in enumerate(words):
        chunk = word + " "
        result = translate(chunk)

        stable = result.get("stable", "")
        buffer = result.get("buffer", "")
        tokens = result.get("committed_tokens", 0)
        src_words = result.get("source_words", [])
        error = result.get("error", "")

        if error:
            print(f"  [{i+1}] \"{chunk}\" → ERROR: {error}")
            continue

        committed_text += stable

        status_parts = []
        if stable:
            status_parts.append(f"\033[32mstable=\"{stable}\"\033[0m")
        else:
            status_parts.append(f"stable=\"\"")
        if buffer:
            status_parts.append(f"\033[33mbuffer=\"{buffer}\"\033[0m")
        else:
            status_parts.append(f"buffer=\"\"")
        status_parts.append(f"tokens={tokens}")

        print(f"  [{i+1}] \"{chunk}\" → {' | '.join(status_parts)}")

    # Finish
    fin = finish()
    remaining = fin.get("remaining", "")
    full = fin.get("full_translation", "")

    if remaining:
        committed_text += remaining

    print(f"  {'─'*56}")
    print(f"  Finish remaining: \"{remaining}\"")
    print(f"  \033[1mFull translation: \"{full}\"\033[0m")
    print(f"  Committed text:   \"{committed_text}\"")

    return full


def run_test_suite():
    """Run a comprehensive test suite."""
    s = status()
    if not s.get("loaded"):
        print("Model not loaded. Loading...")
        result = load_model()
        if not result.get("ok"):
            print(f"Failed to load: {result}")
            sys.exit(1)
        print("Model loaded.\n")

    # English → French tests
    print("\n" + "="*60)
    print("  ENGLISH → FRENCH")
    print("="*60)

    tests_en_fr = [
        (["hi", "my", "name", "is", "quentin"], "Basic introduction"),
        (["the", "weather", "is", "nice", "today"], "Simple statement"),
        (["I", "love", "programming", "in", "python"], "Tech sentence"),
        (["can", "you", "help", "me", "please"], "Question/request"),
        (["the", "cat", "is", "sitting", "on", "the", "mat"], "Longer sentence"),
        (["hello", "world"], "Very short"),
        (["the", "president", "of", "france", "announced", "new", "economic", "reforms", "yesterday"], "News-style"),
    ]

    results = []
    for words, label in tests_en_fr:
        full = test_sentence(words, target_lang="fr", label=f"EN→FR: {label}")
        results.append((words, full, label))

    # French → English tests
    print("\n" + "="*60)
    print("  FRENCH → ENGLISH")
    print("="*60)

    # Need to reload with fr source
    reset()
    load_model(source_lang="fr", target_lang="en")

    tests_fr_en = [
        (["bonjour", "je", "m'appelle", "quentin"], "Basic introduction"),
        (["il", "fait", "beau", "aujourd'hui"], "Simple statement"),
        (["le", "chat", "est", "assis", "sur", "le", "tapis"], "Simple description"),
        (["pouvez", "vous", "m'aider", "s'il", "vous", "plait"], "Polite request"),
    ]

    for words, label in tests_fr_en:
        full = test_sentence(words, target_lang="en", source_lang="fr", label=f"FR→EN: {label}")
        results.append((words, full, label))

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for words, full, label in results:
        src = " ".join(words)
        print(f"  {label}")
        print(f"    \"{src}\" → \"{full}\"")
        print()


if __name__ == "__main__":
    run_test_suite()
