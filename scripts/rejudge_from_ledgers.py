"""Scores already generated held-out outputs again with a different judge, without regenerating anything.

The outputs are rebuilt from the task-model ledgers of finished runs, so both judges see exactly the same text.

Usage:
  python3 rejudge_from_ledgers.py --results-directory <folder with run .json and *_ledger.jsonl> --output-directory <folder>
"""
import argparse
import glob
import json
import os
import re

import pupa_local_trial as trial
import pupa_gepa_pilot as pupa

RESPOND_QUERY_MARKER = "\n\nThe user's request you need to fulfill:\n"
RESPOND_SUFFIX = "\n\nWrite your final response to the user's request."
RESPOND_PREFIX = "Related request to the external LLM:\n"
EXTERNAL_RESPONSE_MARKER = "\n\nResponse from the external LLM (information from a powerful LLM responding to a related request):\n"


def clean(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def rebuild_outputs(ledger_path, candidate, test_rows):
    rollout_entries = [entry for entry in map(json.loads, open(ledger_path)) if entry["usage_kind"] == "rollout"]
    # The held-out evaluation runs after the search has finished, so its calls are the last ones in the ledger.
    test_phase_entries = rollout_entries[-3 * len(test_rows) :]
    outputs_by_query = {}
    for entry in test_phase_entries:
        messages = entry["request"]["messages"]
        if len(messages) != 2 or messages[0]["content"] != candidate["respond_to_query"] or not messages[1]["content"].endswith(RESPOND_SUFFIX):
            continue
        user_content = messages[1]["content"]
        user_query = user_content[: -len(RESPOND_SUFFIX)].rsplit(RESPOND_QUERY_MARKER, 1)[1]
        redacted_request = user_content[len(RESPOND_PREFIX) : user_content.index(EXTERNAL_RESPONSE_MARKER)]
        outputs_by_query.setdefault(user_query, []).append({"llm_request": redacted_request, "response": clean(entry["response"])})
    # Several held-out rows share the same query; each of them was generated separately, so hand the outputs out one per row.
    outputs = [outputs_by_query[row["user_query"]].pop() for row in test_rows]
    assert not any(outputs_by_query.values()), "some held-out outputs were not matched to a row"
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="/tmp/pupa_data/pupa_new.parquet")
    parser.add_argument("--results-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    arguments = parser.parse_args()

    pupa.call_model = trial.local_call_model
    test_rows = pupa.load_splits(arguments.data_file)["test"]
    os.makedirs(arguments.output_directory, exist_ok=True)
    for record_path in sorted(glob.glob(os.path.join(arguments.results_directory, "*.json"))):
        record = json.load(open(record_path))
        run_name = record["run_name"]
        if run_name == "seed_prompts":
            ledger_name = "seed_prompts"
        else:
            ledger_name = ("budgetapo" if run_name.startswith("budgetapo") else "stock") + "_seed" + re.search(r"_seed(\d)", run_name).group(1)
        outputs = rebuild_outputs(os.path.join(arguments.results_directory, ledger_name + "_ledger.jsonl"), record["best_candidate"], test_rows)
        judgments = pupa.run_parallel(pupa.judge_once_steps, [(row, output, trial.LOCAL_MODEL_NAME, "rejudge") for row, output in zip(test_rows, outputs)], 32)
        rejudged = {
            "run_name": run_name,
            "judge": trial.LOCAL_MODEL_NAME,
            "test_scores": [judgment["score"] for judgment in judgments],
            "test_quality": [judgment["quality"] for judgment in judgments],
            "test_leakage": [judgment["leakage"] for judgment in judgments],
            "original_judge_scores": record["test_scores"],
            "original_judge_quality": record["test_quality"],
            "original_judge_leakage": record["test_leakage"],
        }
        rejudged["test_mean"] = sum(rejudged["test_scores"]) / len(rejudged["test_scores"])
        with open(os.path.join(arguments.output_directory, run_name + ".json"), "w") as output_file:
            json.dump(rejudged, output_file)
        print(run_name, "rejudged mean", round(rejudged["test_mean"], 4), "| original judge mean", round(record["test_mean"], 4), flush=True)


if __name__ == "__main__":
    main()
