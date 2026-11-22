"""Runs the PUPA pilot scripts against one locally served model that plays every role.

Usage:
  python3 pupa_local_trial.py seed       --data-file ... --output-directory ...
  python3 pupa_local_trial.py optimize   --data-file ... --output-directory ... --mode stock --reflection budgetapo ...
  python3 pupa_local_trial.py replay     --ledger <ledger.jsonl> --sample-size 60 --workers 1
"""
import contextlib
import fcntl
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request

import pandas as pd

os.environ.setdefault("DASHSCOPE_API_KEY", "unused")
os.environ.setdefault("TOPCLOUD_BASE_URL", "unused")
os.environ.setdefault("TOPCLOUD_API_KEY", "unused")

import pupa_gepa_pilot as pupa

LOCAL_URL = os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8091/v1") + "/chat/completions"
LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "local-model")
GREEDY_DECODING = os.environ.get("LOCAL_DECODING", "greedy") == "greedy"
TEST_ROWS = int(os.environ.get("LOCAL_TEST_ROWS", str(pupa.TEST_SIZE)))
LEDGER_PATH = os.environ.get("LOCAL_LEDGER", "")
# "all": the served model plays every role. "task": it is only the task model; reflection and judging stay on qwen3.8-27b.
SERVED_ROLES = os.environ.get("LOCAL_ROLES", "all")
API_KEY = os.environ.get("LOCAL_API_KEY", "")
EXTRA_BODY = json.loads(os.environ.get("LOCAL_EXTRA_BODY", "{}"))
# "two_local": a second served model takes the reflection and judge roles.
SECOND_MODEL_NAME = os.environ.get("SECOND_MODEL_NAME", "")
ENDPOINT_URLS = {LOCAL_MODEL_NAME: LOCAL_URL}
if SECOND_MODEL_NAME:
    ENDPOINT_URLS[SECOND_MODEL_NAME] = os.environ["SECOND_BASE_URL"] + "/chat/completions"
endpoint_slots = {model_name: threading.Semaphore(int(os.environ.get("LOCAL_CONCURRENCY", "6"))) for model_name in ENDPOINT_URLS}
# When set, a reflection call waits until nothing else is in flight on the second server, so that it can be replayed exactly.
REFLECTION_LOCK_PATH = os.environ.get("REFLECTION_LOCK_PATH", "")
DEDUPLICATED_SPLIT = os.environ.get("DEDUPLICATED_SPLIT", "") == "1"
DEDUPLICATED_SIZES = {"test": 185, "validation": 88, "train": 95}

ledger_lock = threading.Lock()


@contextlib.contextmanager
def second_server_turn(alone):
    """Shares the second server among ordinary calls, but gives a reflection call the server to itself."""
    with open(REFLECTION_LOCK_PATH + ".gate", "a") as gate_file, open(REFLECTION_LOCK_PATH, "a") as lock_file:
        # Everyone passes the gate first, so a waiting reflection call is not overtaken by new ordinary calls.
        fcntl.flock(gate_file, fcntl.LOCK_EX)
        fcntl.flock(lock_file, fcntl.LOCK_EX if alone else fcntl.LOCK_SH)
        if not alone:
            fcntl.flock(gate_file, fcntl.LOCK_UN)
        yield


def send_request(body, alone=False):
    headers = {"Content-Type": "application/json", "User-Agent": "curl/8.7.1"}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY
    request = urllib.request.Request(ENDPOINT_URLS[body["model"]], data=json.dumps({**body, **EXTRA_BODY}).encode(), headers=headers)
    take_turn = second_server_turn(alone) if REFLECTION_LOCK_PATH and body["model"] == SECOND_MODEL_NAME else contextlib.nullcontext()
    for attempt in range(6):
        try:
            with endpoint_slots[body["model"]], take_turn:
                with urllib.request.urlopen(request, timeout=1800) as response:
                    payload = json.load(response)
            if payload.get("choices"):
                return payload
            last_error = str(payload)[:300]
        except Exception as error:
            last_error = error
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"request failed after retries: {last_error}")


def local_call_model(model, messages, max_tokens, usage_kind, temperature=None):
    # Judges always decode greedily. The task model keeps its 0.6 and reflection its 1.0 unless LOCAL_DECODING=greedy.
    if temperature is None:
        temperature = 1.0 if usage_kind == "reflection" else 0.0
    if GREEDY_DECODING:
        temperature = 0.0
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": 0}
    start_time = time.time()
    payload = send_request(body, alone=usage_kind == "reflection")
    content = payload["choices"][0]["message"].get("content") or ""
    usage = payload.get("usage", {})
    with pupa.usage_lock:
        pupa.token_usage[usage_kind] = pupa.token_usage.get(usage_kind, 0) + usage.get("total_tokens", 0)
        pupa.call_counts[usage_kind] = pupa.call_counts.get(usage_kind, 0) + 1
    if LEDGER_PATH:
        entry = {
            "usage_kind": usage_kind,
            "request": body,
            "response": content,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "seconds": round(time.time() - start_time, 2),
            "finished_at": time.time(),
        }
        with ledger_lock:
            with open(LEDGER_PATH, "a") as ledger_file:
                ledger_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def load_deduplicated_splits(data_file):
    """Removes exact copies, then assigns whole conversations with the same seeded order as the pilot split."""
    frame = pd.read_parquet(data_file)
    frame["example_id"] = [f"row{index}" for index in range(len(frame))]
    frame = frame.drop_duplicates(subset=["user_query", "target_response", "pii_units"], keep="first")
    conversation_hashes = sorted(frame["conversation_hash"].unique())
    random.Random(0).shuffle(conversation_hashes)
    fill_until = {"test": 184, "validation": 86, "train": 94}
    splits = {split_name: [] for split_name in fill_until}
    for conversation_hash in conversation_hashes:
        rows = frame[frame["conversation_hash"] == conversation_hash].to_dict("records")
        for split_name in ("test", "validation", "train"):
            if len(splits[split_name]) < fill_until[split_name]:
                splits[split_name].extend(rows)
                break
    assert {split_name: len(rows) for split_name, rows in splits.items()} == DEDUPLICATED_SIZES
    return splits


def replay(ledger_path, sample_size, workers, usage_kind=""):
    entries = [entry for entry in map(json.loads, open(ledger_path)) if usage_kind in ("", entry["usage_kind"])]
    sampled_entries = random.Random(0).sample(entries, min(sample_size, len(entries)))
    start_time = time.time()
    payloads = pupa.run_parallel(lambda entry: send_request(entry["request"], alone=entry["usage_kind"] == "reflection"), sampled_entries, workers)
    summary = {}
    for entry, payload in zip(sampled_entries, payloads):
        replayed = payload["choices"][0]["message"].get("content") or ""
        counts = summary.setdefault(entry["usage_kind"], {"calls": 0, "identical": 0})
        counts["calls"] += 1
        counts["identical"] += replayed == entry["response"]
    print(json.dumps({"ledger": ledger_path, "workers": workers, "seconds": round(time.time() - start_time), "by_kind": summary}))


if __name__ == "__main__":
    command = sys.argv.pop(1)
    if command == "replay":
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--ledger", required=True)
        parser.add_argument("--sample-size", type=int, default=60)
        parser.add_argument("--workers", type=int, default=1)
        parser.add_argument("--kind", default="", help="only replay calls of this kind, for example reflection")
        arguments = parser.parse_args()
        replay(arguments.ledger, arguments.sample_size, arguments.workers, arguments.kind)
        sys.exit(0)

    pupa.LOCAL_MODEL = LOCAL_MODEL_NAME
    if SERVED_ROLES == "all":
        pupa.call_model = local_call_model
        pupa.SEARCH_JUDGE = pupa.TEST_JUDGE = pupa.REFLECTION_MODEL = LOCAL_MODEL_NAME
    elif SERVED_ROLES == "two_local":
        pupa.call_model = local_call_model
        pupa.SEARCH_JUDGE = pupa.TEST_JUDGE = pupa.REFLECTION_MODEL = SECOND_MODEL_NAME
    else:
        hosted_call_model = pupa.call_model
        pupa.call_model = lambda model, *arguments, **options: (local_call_model if model == LOCAL_MODEL_NAME else hosted_call_model)(model, *arguments, **options)
        pupa.SEARCH_JUDGE = pupa.TEST_JUDGE = pupa.REFLECTION_MODEL
    pupa.TEST_JUDGE_REPEATS = 1
    original_load_splits = pupa.load_splits
    if DEDUPLICATED_SPLIT:
        original_load_splits = load_deduplicated_splits
        pupa.VALIDATION_SIZE = DEDUPLICATED_SIZES["validation"]

    def load_splits_with_shorter_test(data_file):
        splits = original_load_splits(data_file)
        splits["test"] = splits["test"][:TEST_ROWS]
        return splits

    pupa.load_splits = load_splits_with_shorter_test
    if command == "seed":
        pupa.main()
    else:
        import pupa_gepa_configured

        pupa_gepa_configured.main()
