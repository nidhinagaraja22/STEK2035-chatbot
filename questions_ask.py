"""
STEK 2035 — Auto-Start Server + Ask Question
=================================================
Checks if your FastAPI server is already running; if not, starts it
automatically in the background (uvicorn rag_server:app --port 8000),
waits until it's ready, then sends your question and prints the answer.

You never need to manually run uvicorn in a separate terminal for
quick testing — just edit QUESTION below and run this one script.

The server keeps running in the background after this script exits
(so repeated runs of this script are fast — no restart needed each
time). To actually stop it, find and kill the process:
    ps aux | grep uvicorn
    kill <pid>

Usage:
    python ask.py
"""

import json
import subprocess
import sys
import time

import requests

# ── EDIT THIS to change the question ────────────────────────────────────────
QUESTION = "Ich möchte ein Haus kaufen – welche Unterlagen werden benötigt??"

# ── EDIT THESE if your setup differs ────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"
# ── EDIT THIS to your actual repo root — the directory that CONTAINS
#    both "backend/" and "corpus/" as siblings. rag_server.py uses
#    relative paths (e.g. "corpus/corpus_v2/meta_v2.json") that only
#    resolve correctly if uvicorn's WORKING DIRECTORY is this repo
#    root, not the backend/ subfolder — a known issue in this project.
REPO_ROOT = "/home/jovyan/vault/STEK2035-chatbot-main"

# --app-dir tells uvicorn WHERE TO IMPORT rag_server.py FROM (backend/),
# while cwd (set below, in start_server_in_background) controls where
# relative file paths INSIDE rag_server.py actually resolve — these are
# two different things and both need to be right.
UVICORN_CMD = ["uvicorn", "rag_server:app", "--port", "8000", "--app-dir", "backend"]
STARTUP_TIMEOUT_SECONDS = 120  # embedding model load + corpus load can be slow
                                  # the first time; increase if yours is slower


def is_server_running():
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_server_in_background():
    print("Server not running — starting it now in the background...")
    print(f"  Command: {' '.join(UVICORN_CMD)}")
    print(f"  Working directory: {REPO_ROOT}")
    log_file = open("uvicorn_startup.log", "w")
    process = subprocess.Popen(UVICORN_CMD, stdout=log_file, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    print(f"  Started (pid={process.pid}), logging to uvicorn_startup.log")
    return process


def wait_for_server(timeout_seconds):
    print(f"Waiting for server to be ready (up to {timeout_seconds}s)...")
    start = time.time()
    while time.time() - start < timeout_seconds:
        if is_server_running():
            elapsed = time.time() - start
            print(f"✅ Server ready after {elapsed:.1f}s\n")
            return True
        print(f"  ...still loading ({time.time() - start:.0f}s elapsed) — "
              f"check uvicorn_startup.log if this takes unusually long")
        time.sleep(5)
    return False


def ask_question(question):
    print(f"Question: {question}")
    print(f"Sending to: {CHAT_ENDPOINT}\n")

    try:
        response = requests.post(CHAT_ENDPOINT, json={"message": question}, timeout=120)
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {CHAT_ENDPOINT} even after startup — "
              f"check uvicorn_startup.log for errors.")
        return

    if response.status_code != 200:
        print(f"❌ Server returned status {response.status_code}")
        print(response.text[:500])
        return

    data = response.json()

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(data.get("reply", "(no 'reply' field in response)"))

    if "verdict" in data:
        print(f"\nVerdict: {data['verdict']}")
    if data.get("tier_used"):
        print(f"Tier used: {data['tier_used']}")

    sources = data.get("sources", [])
    if sources:
        print(f"\n{'-'*70}")
        print(f"SOURCES ({len(sources)})")
        print(f"{'-'*70}")
        for i, s in enumerate(sources, 1):
            citation = s.get("citation", "")
            authority = s.get("authority_level", "?")
            print(f"  {i}. [{citation}]" if citation else f"  {i}. [L{authority}] {s.get('origin', '?')}")
            print(f"     {s.get('text', '')[:150]}...")

    with open("last_response.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*70}")
    print(f"✅ Full raw response saved to last_response.json")


def main():
    if is_server_running():
        print("✅ Server already running — skipping startup.\n")
    else:
        start_server_in_background()
        if not wait_for_server(STARTUP_TIMEOUT_SECONDS):
            print(f"❌ Server didn't become ready within {STARTUP_TIMEOUT_SECONDS}s.")
            print(f"   Check uvicorn_startup.log for the actual error — common causes:")
            print(f"   missing corpus files, Ollama not running, wrong file paths in config.")
            sys.exit(1)

    ask_question(QUESTION)


if __name__ == "__main__":
    main()