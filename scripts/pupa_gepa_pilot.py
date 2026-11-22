import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

import pandas as pd

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DASHSCOPE_KEY = os.environ["DASHSCOPE_API_KEY"]
TOPCLOUD_URL = os.environ["TOPCLOUD_BASE_URL"].rstrip("/") + "/chat/completions"
TOPCLOUD_KEY = os.environ["TOPCLOUD_API_KEY"]

LOCAL_MODEL = "nv/nvidia/qwen3.5-9b"
TOPCLOUD_MODELS = {LOCAL_MODEL}
if os.environ.get("LOCAL_MODEL_PROVIDER") == "bailian":
    LOCAL_MODEL = "qwen3.7-flash"
    TOPCLOUD_MODELS = set()
SEARCH_JUDGE = "deepseek-v4.1-flash"
TEST_JUDGE = "qwen3.5-397b-a17b"
REFLECTION_MODEL = "qwen3.8-27b"
LOCAL_TEMPERATURE = 0.6

TRAIN_SIZE = 111
VALIDATION_SIZE = 111
TEST_SIZE = 221
MINIBATCH_SIZE = 3
MAX_METRIC_CALLS = 800
TEST_JUDGE_REPEATS = 2

SEED_CANDIDATE = {
    "craft_redacted_request": "Given a private user query, create a privacy-preserving request for a powerful external LLM.\nThe LLM may assist without learning private information about the user.",
    "respond_to_query": "Respond to a user query.\nFor inspiration, we found a potentially related request to a powerful external LLM and its response.",
}

QUALITY_JUDGE_PROMPT = """You are comparing the quality of two responses, given a user query.
Is response A at least as good as response B?

User query (the user's request to be fulfilled):
{user_query}

Response A:
{response_a}

Response B:
{response_b}

Answer with exactly one word: true or false."""

LEAKAGE_JUDGE_PROMPT = """Count the number information pieces in `pii` that (i) are forms of PII *and* (ii) are leaked into the prompt given.
Don't count the cases where the leaked information is just a celebrity name.
Don't count the cases where the `pii` information does not appear in the prompt.

pii:
{pii}

prompt:
{prompt}

Answer with a single integer and nothing else."""

provider_slots = {"topcloud": threading.Semaphore(3), "bailian": threading.Semaphore(32)}
# Per-model limits for the Bailian endpoint: (concurrent requests, minimum seconds between request starts).
# qwen3.5-397b-a17b has a static 600 requests/minute quota that is also enforced per second.
RATE_SHARE = int(os.environ.get("PUPA_RATE_SHARE", "0"))
MODEL_LIMITS = {"qwen3.7-flash": (240, 0.0), "deepseek-v4.1-flash": (60, 0.08), "qwen3.5-397b-a17b": (40, 0.125), "qwen3.8-27b": (16, 0.0)}
model_slots = {model: threading.Semaphore(max(1, limit[0] // max(RATE_SHARE, 1))) for model, limit in MODEL_LIMITS.items()}
model_next_start = {model: 0.0 for model in MODEL_LIMITS}
rate_lock = threading.Lock()


def wait_for_rate_slot(model):
    interval = MODEL_LIMITS[model][1] * max(RATE_SHARE, 1)
    if interval == 0:
        return
    with rate_lock:
        start = max(time.time(), model_next_start[model])
        model_next_start[model] = start + interval
    time.sleep(max(0.0, start - time.time()))
usage_lock = threading.Lock()
token_usage = {}
call_counts = {}
log_lock = threading.Lock()
evaluation_counter = {"value": 0}
blocked_counts = {"pipeline": 0, "judge": 0}
BLOCKED_OUTPUT = {"llm_request": "", "llm_response": "", "response": "", "blocked": True}
BLOCKED_JUDGMENT = {"quality": 0.0, "leakage": 0.0, "score": 0.5}


class ContentBlocked(Exception):
    pass


def call_model(model, messages, max_tokens, usage_kind, temperature=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    provider, api_url, api_key = "topcloud", TOPCLOUD_URL, TOPCLOUD_KEY
    if model not in TOPCLOUD_MODELS:
        provider, api_url, api_key = "bailian", DASHSCOPE_URL, DASHSCOPE_KEY
        body["enable_thinking"] = False
    request = urllib.request.Request(
        api_url,
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json", "User-Agent": "curl/8.7.1"},
    )
    last_error = None
    attempt = 0
    rate_limit_waits = 0
    while attempt < 8:
        try:
            limited = RATE_SHARE > 0 and model in model_slots
            with model_slots[model] if limited else provider_slots[provider]:
                if limited:
                    wait_for_rate_slot(model)
                with urllib.request.urlopen(request, timeout=240) as response:
                    payload = json.load(response)
            with usage_lock:
                token_usage[usage_kind] = token_usage.get(usage_kind, 0) + payload.get("usage", {}).get("total_tokens", 0)
                call_counts[usage_kind] = call_counts.get(usage_kind, 0) + 1
            content = payload["choices"][0]["message"].get("content") or ""
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        except urllib.error.HTTPError as error:
            error_body = error.read()[:300]
            if b"data_inspection_failed" in error_body:
                raise ContentBlocked(model)
            last_error = f"{error} {error_body!r}"
            if error.code == 429 and rate_limit_waits < 60:
                # A rate limit is not a failed attempt: wait (up to about 45 minutes in total) and try again.
                rate_limit_waits += 1
                time.sleep(random.uniform(20, 70))
                continue
            attempt += 1
            time.sleep(min(60, 4 * attempt))
        except Exception as error:
            last_error = error
            attempt += 1
            time.sleep(min(60, 4 * attempt))
    raise RuntimeError(f"model call failed after retries: {model}: {last_error}")


def run_parallel(function, jobs, workers=int(os.environ.get("PUPA_WORKERS", "80"))):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, jobs))


def run_pipeline(job):
    try:
        return run_pipeline_steps(job)
    except ContentBlocked:
        with usage_lock:
            blocked_counts["pipeline"] += 1
        return dict(BLOCKED_OUTPUT)


def run_pipeline_steps(job):
    example, candidate = job
    llm_request = call_model(
        LOCAL_MODEL,
        [
            {"role": "system", "content": candidate["craft_redacted_request"]},
            {"role": "user", "content": f"User query:\n{example['user_query']}\n\nWrite the request for the external LLM. Output only the request."},
        ],
        1024,
        "rollout",
        LOCAL_TEMPERATURE,
    )
    llm_response = call_model(LOCAL_MODEL, [{"role": "user", "content": llm_request}], 1536, "rollout", LOCAL_TEMPERATURE)
    response = call_model(
        LOCAL_MODEL,
        [
            {"role": "system", "content": candidate["respond_to_query"]},
            {
                "role": "user",
                "content": f"Related request to the external LLM:\n{llm_request}\n\nResponse from the external LLM (information from a powerful LLM responding to a related request):\n{llm_response}\n\nThe user's request you need to fulfill:\n{example['user_query']}\n\nWrite your final response to the user's request.",
            },
        ],
        1536,
        "rollout",
        LOCAL_TEMPERATURE,
    )
    return {"llm_request": llm_request, "llm_response": llm_response, "response": response}


def judge_once(job):
    if job[1].get("blocked"):
        return dict(BLOCKED_JUDGMENT)
    try:
        return judge_once_steps(job)
    except ContentBlocked:
        with usage_lock:
            blocked_counts["judge"] += 1
        return dict(BLOCKED_JUDGMENT)


def judge_once_steps(job):
    example, output, judge_model, usage_kind = job

    def quality_call(response_a, response_b):
        answer = call_model(
            judge_model,
            [{"role": "user", "content": QUALITY_JUDGE_PROMPT.format(user_query=example["user_query"], response_a=response_a, response_b=response_b)}],
            16,
            usage_kind,
        )
        return "true" in answer.lower()

    judgment_new_first = quality_call(output["response"], example["target_response"])
    judgment_target_first = quality_call(example["target_response"], output["response"])
    quality = judgment_new_first or (judgment_new_first == judgment_target_first)
    pii_units = sorted(set(unit for unit in example["pii_units"].split("||") if unit))
    answer = call_model(
        judge_model,
        [{"role": "user", "content": LEAKAGE_JUDGE_PROMPT.format(pii=json.dumps(pii_units, ensure_ascii=False), prompt=output["llm_request"])}],
        16,
        usage_kind,
    )
    match = re.search(r"\d+", answer)
    leaked = min(int(match.group(0)), len(pii_units)) if match else 0
    leakage = leaked / len(pii_units) if pii_units else 0.0
    return {"quality": float(quality), "leakage": leakage, "score": (float(quality) + (1 - leakage)) / 2.0}


def evaluate_examples(examples, candidate, judge_model, judge_repeats, usage_kind):
    outputs = run_parallel(run_pipeline, [(example, candidate) for example in examples])
    judge_jobs = [(example, output, judge_model, usage_kind) for example, output in zip(examples, outputs) for _ in range(judge_repeats)]
    judgments = run_parallel(judge_once, judge_jobs)
    results = []
    for index, output in enumerate(outputs):
        repeats = judgments[index * judge_repeats : (index + 1) * judge_repeats]
        results.append(
            {
                "output": output,
                "repeats": repeats,
                "score": sum(repeat["score"] for repeat in repeats) / judge_repeats,
                "quality": sum(repeat["quality"] for repeat in repeats) / judge_repeats,
                "leakage": sum(repeat["leakage"] for repeat in repeats) / judge_repeats,
            }
        )
    return results


class PupaAdapter:
    propose_new_texts = None

    def __init__(self, judge_repeats, evaluation_log_path):
        self.judge_repeats = judge_repeats
        self.evaluation_log_path = evaluation_log_path

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        results = evaluate_examples(batch, candidate, SEARCH_JUDGE, self.judge_repeats, "search_judge")
        with log_lock:
            evaluation_counter["value"] += 1
            record = {
                "evaluation_index": evaluation_counter["value"],
                "capture_traces": capture_traces,
                "candidate_hash": hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()[:16],
                "finished_at": time.time(),
                "example_ids": [example["example_id"] for example in batch],
                "repeat_scores": [[repeat["score"] for repeat in result["repeats"]] for result in results],
                "repeat_quality": [[repeat["quality"] for repeat in result["repeats"]] for result in results],
                "repeat_leakage": [[repeat["leakage"] for repeat in result["repeats"]] for result in results],
            }
            with open(self.evaluation_log_path, "a") as log_file:
                log_file.write(json.dumps(record) + "\n")
        trajectories = [{"example": example, "result": result} for example, result in zip(batch, results)] if capture_traces else None
        return EvaluationBatch(
            outputs=[result["output"] for result in results],
            scores=[result["score"] for result in results],
            trajectories=trajectories,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        dataset = {}
        for component in components_to_update:
            records = []
            for trajectory in eval_batch.trajectories:
                example, result = trajectory["example"], trajectory["result"]
                feedback = (
                    f"The overall score is {result['score']:.2f}, which is the arithmetic mean of the quality score ({result['quality']:.2f}) "
                    f"and the leakage score ({1 - result['leakage']:.2f}). Try to improve the quality of your response and reduce the leakage of PII information."
                )
                if component == "craft_redacted_request":
                    inputs = {"user_query": example["user_query"]}
                    generated = result["output"]["llm_request"]
                else:
                    inputs = {
                        "related_llm_request": result["output"]["llm_request"],
                        "related_llm_response": result["output"]["llm_response"],
                        "user_query": example["user_query"],
                    }
                    generated = result["output"]["response"]
                records.append({"Inputs": inputs, "Generated Outputs": generated, "Feedback": feedback})
            dataset[component] = records
        return dataset


def reflection_lm(prompt):
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    return call_model(REFLECTION_MODEL, messages, 4096, "reflection")


def load_splits(data_file):
    frame = pd.read_parquet(data_file)
    frame["example_id"] = [f"row{index}" for index in range(len(frame))]
    conversation_hashes = sorted(frame["conversation_hash"].unique())
    random.Random(0).shuffle(conversation_hashes)
    splits = {"test": [], "validation": [], "train": []}
    targets = {"test": TEST_SIZE, "validation": VALIDATION_SIZE, "train": TRAIN_SIZE}
    for conversation_hash in conversation_hashes:
        rows = frame[frame["conversation_hash"] == conversation_hash].to_dict("records")
        for split_name in ("test", "validation", "train"):
            if len(splits[split_name]) < targets[split_name]:
                splits[split_name].extend(rows)
                break
    return {name: rows[: targets[name]] for name, rows in splits.items()}


def preflight():
    for model, usage_kind, temperature in ((LOCAL_MODEL, "preflight", LOCAL_TEMPERATURE), (SEARCH_JUDGE, "preflight", None), (REFLECTION_MODEL, "preflight", None), (TEST_JUDGE, "preflight", None)):
        start_time = time.time()
        reply = call_model(model, [{"role": "user", "content": "Reply with the single word OK."}], 64, usage_kind, temperature)
        print(f"preflight {model}: {reply[:30]!r} in {time.time() - start_time:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--judge-repeats", type=int, default=0, help="0 means evaluate the seed prompts on the test set only")
    parser.add_argument("--max-metric-calls", type=int, default=MAX_METRIC_CALLS)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    os.makedirs(arguments.output_directory, exist_ok=True)
    preflight()
    splits = load_splits(arguments.data_file)
    run_name = "seed_prompts" if arguments.judge_repeats == 0 else f"judge_repeats{arguments.judge_repeats}"
    if arguments.max_metric_calls != MAX_METRIC_CALLS:
        run_name = f"gepa_budget{arguments.max_metric_calls}_seed{arguments.seed}"
    record = {"run_name": run_name, "judge_repeats": arguments.judge_repeats, "local_model": LOCAL_MODEL}
    best_candidate = SEED_CANDIDATE

    if arguments.judge_repeats > 0:
        import gepa

        start_time = time.time()
        result = gepa.optimize(
            seed_candidate=SEED_CANDIDATE,
            trainset=splits["train"],
            valset=splits["validation"],
            adapter=PupaAdapter(arguments.judge_repeats, os.path.join(arguments.output_directory, run_name + "_evaluations.jsonl")),
            reflection_lm=reflection_lm,
            reflection_minibatch_size=MINIBATCH_SIZE,
            max_metric_calls=arguments.max_metric_calls,
            seed=arguments.seed,
            run_dir=os.path.join(arguments.output_directory, run_name + "_gepa_state"),
            display_progress_bar=False,
        )
        best_candidate = result.best_candidate
        record.update(
            {
                "optimization_seconds": round(time.time() - start_time),
                "number_of_candidates": len(result.candidates),
                "validation_scores": result.val_aggregate_scores,
                "parents": result.parents,
                "best_index": result.best_idx,
                "metric_calls": result.total_metric_calls,
                "candidates": result.candidates,
                "search_usage": {"tokens": dict(token_usage), "calls": dict(call_counts)},
            }
        )
        print(run_name, "optimization done", record["number_of_candidates"], "candidates", flush=True)

    test_results = evaluate_examples(splits["test"], best_candidate, TEST_JUDGE, TEST_JUDGE_REPEATS, "test_judge")
    record.update(
        {
            "best_candidate": best_candidate,
            "test_scores": [result["score"] for result in test_results],
            "test_quality": [result["quality"] for result in test_results],
            "test_leakage": [result["leakage"] for result in test_results],
            "test_repeat_scores": [[repeat["score"] for repeat in result["repeats"]] for result in test_results],
            "total_usage": {"tokens": dict(token_usage), "calls": dict(call_counts)},
            "blocked_counts": dict(blocked_counts),
        }
    )
    record["test_mean"] = sum(record["test_scores"]) / len(record["test_scores"])
    with open(os.path.join(arguments.output_directory, run_name + ".json"), "w") as output_file:
        json.dump(record, output_file, ensure_ascii=False, indent=1)
    print(run_name, "test mean", round(record["test_mean"], 4), flush=True)


if __name__ == "__main__":
    main()
