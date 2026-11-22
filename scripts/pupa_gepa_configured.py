import argparse
import json
import os
import random
import threading
import time

import pupa_gepa_pilot as pupa

SLICE_SIZE = 10
PROPOSALS_PER_ITERATION = 2

pupa.provider_slots["bailian"] = threading.Semaphore(24)  # only used when PUPA_RATE_SHARE is unset

BUDGETAPO_TEMPLATE = """You are improving a system instruction that is prepended to every user request an assistant receives. Replies are scored automatically; the instruction's job is to make replies score as high as possible.

TASK CONTEXT
The assistant receives a private user query and must write a request for an untrusted external LLM that helps answer the query without revealing the user's private information. The external LLM's answer is then used to write the final response to the user. Each case is scored as the mean of two parts: whether the final response is at least as good as a reference response, and one minus the fraction of the user's private details that appear in the request sent to the external LLM.

CURRENT INSTRUCTION
```
<curr_param>
```

EXAMPLES OF LOW-SCORING CASES
<side_info>

Rewrite the instruction to raise the score. Cover BOTH:
(a) strategy -- how the assistant should approach this kind of request to get the content right (what to look at, how to derive the answer);
(b) format -- exactly what the output should look like to score highest.
Keep what already works. Be concrete, not exhortative. Output only the new instruction, inside triple backticks."""


def budgetapo_reflection_lm(prompt):
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    return pupa.call_model(pupa.REFLECTION_MODEL, messages, 4096, "reflection", 1.0)


class CountingAdapter(pupa.PupaAdapter):
    """Counts every row that is really rolled out, independently of GEPA's own bookkeeping."""

    real_calls = 0
    reflection_style = "gepa"

    def evaluate(self, batch, candidate, capture_traces=False):
        self.real_calls += len(batch)
        return super().evaluate(batch, candidate, capture_traces)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        if self.reflection_style == "gepa":
            return super().make_reflective_dataset(candidate, eval_batch, components_to_update)
        # BudgetAPO style: at most three low-scoring cases, each shown only as short excerpts.
        results = [trajectory["result"] for trajectory in eval_batch.trajectories]
        average_score = sum(result["score"] for result in results) / len(results)
        failing = [trajectory for trajectory in eval_batch.trajectories if trajectory["result"]["score"] < 1][:3]
        dataset = {}
        for component in components_to_update:
            records = []
            for trajectory in failing:
                example, result = trajectory["example"], trajectory["result"]
                missed = []
                if result["leakage"] > 0:
                    missed.append(f"privacy: {result['leakage']:.0%} of the user's private details appeared in the request to the external LLM")
                if result["quality"] < 1:
                    missed.append("quality: the final response was judged worse than the reference response")
                reply = result["output"]["llm_request"] if component == "craft_redacted_request" else result["output"]["response"]
                records.append({"Request": example["user_query"][:150], "Scored low on": "; ".join(missed), "Reply": reply[:160]})
            if records:
                records[0] = {"Average score of the current instruction on the evaluated cases": f"{average_score:.2f}", **records[0]}
            dataset[component] = records
        return dataset


class WholeSetBatchSampler:
    """Returns every training id in a fixed order, so all proposals are compared on the same rows."""

    def next_minibatch_ids(self, loader, state):
        return sorted(loader.all_ids())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--mode", choices=["stock", "fixed_slice"], required=True)
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--reflection", choices=["gepa", "budgetapo"], default="gepa")
    parser.add_argument("--validation-size", type=int, default=pupa.VALIDATION_SIZE, help="stock mode only: number of validation rows, sampled with the run seed")
    arguments = parser.parse_args()

    import gepa
    from gepa.strategies.proposal_sampling import SameParentSampling

    os.makedirs(arguments.output_directory, exist_ok=True)
    pupa.preflight()
    splits = pupa.load_splits(arguments.data_file)
    run_name = f"{arguments.mode}_budget{arguments.budget}_seed{arguments.seed}"
    if arguments.reflection == "budgetapo":
        run_name = "budgetapo_reflection_" + run_name
    if arguments.validation_size != pupa.VALIDATION_SIZE:
        run_name += f"_validation{arguments.validation_size}"
    adapter = CountingAdapter(1, os.path.join(arguments.output_directory, run_name + "_evaluations.jsonl"))
    adapter.reflection_style = arguments.reflection
    common = {
        "seed_candidate": pupa.SEED_CANDIDATE,
        "adapter": adapter,
        "reflection_lm": budgetapo_reflection_lm if arguments.reflection == "budgetapo" else pupa.reflection_lm,
        "reflection_prompt_template": BUDGETAPO_TEMPLATE if arguments.reflection == "budgetapo" else None,
        "seed": arguments.seed,
        "run_dir": os.path.join(arguments.output_directory, run_name + "_gepa_state"),
        "display_progress_bar": False,
    }
    start_time = time.time()
    if arguments.mode == "stock":
        # One more iteration can cost a minibatch comparison plus a full validation pass.
        largest_iteration_cost = 2 * pupa.MINIBATCH_SIZE + arguments.validation_size
        validation_rows = splits["validation"]
        if arguments.validation_size != pupa.VALIDATION_SIZE:
            validation_rows = random.Random(arguments.seed).sample(list(validation_rows), arguments.validation_size)
        result = gepa.optimize(
            trainset=splits["train"],
            valset=validation_rows,
            reflection_minibatch_size=pupa.MINIBATCH_SIZE,
            stop_callbacks=[lambda gepa_state: arguments.budget - adapter.real_calls < largest_iteration_cost],
            **common,
        )
    else:
        slice_rows = random.Random(arguments.seed).sample(list(splits["validation"]), 16)[:SLICE_SIZE]
        largest_iteration_cost = (1 + PROPOSALS_PER_ITERATION) * SLICE_SIZE
        result = gepa.optimize(
            trainset=slice_rows,
            batch_sampler=WholeSetBatchSampler(),
            candidate_selection_strategy="current_best",
            sampling_strategy=SameParentSampling(PROPOSALS_PER_ITERATION),
            cache_evaluation=True,
            stop_callbacks=[lambda gepa_state: arguments.budget - adapter.real_calls < largest_iteration_cost],
            **common,
        )
    optimization_seconds = round(time.time() - start_time)
    print(run_name, "optimization done; real calls", adapter.real_calls, "candidates", len(result.candidates), "best index", result.best_idx, flush=True)
    search_usage = {"tokens": dict(pupa.token_usage), "calls": dict(pupa.call_counts)}
    if arguments.skip_test:
        print(run_name, "validation scores", result.val_aggregate_scores, "gepa metric calls", result.total_metric_calls, flush=True)
        return

    test_start_time = time.time()
    test_results = pupa.evaluate_examples(splits["test"], result.best_candidate, pupa.TEST_JUDGE, pupa.TEST_JUDGE_REPEATS, "test_judge")
    test_seconds = round(time.time() - test_start_time)
    record = {
        "run_name": run_name,
        "mode": arguments.mode,
        "budget": arguments.budget,
        "real_calls": adapter.real_calls,
        "gepa_metric_calls": result.total_metric_calls,
        "optimization_seconds": optimization_seconds,
        "test_seconds": test_seconds,
        "number_of_candidates": len(result.candidates),
        "validation_scores": result.val_aggregate_scores,
        "parents": result.parents,
        "best_index": result.best_idx,
        "returned_seed": result.best_candidate == pupa.SEED_CANDIDATE,
        "candidates": result.candidates,
        "best_candidate": result.best_candidate,
        "search_usage": search_usage,
        "test_scores": [test_result["score"] for test_result in test_results],
        "test_quality": [test_result["quality"] for test_result in test_results],
        "test_leakage": [test_result["leakage"] for test_result in test_results],
        "test_mean": sum(test_result["score"] for test_result in test_results) / len(test_results),
        "total_usage": {"tokens": dict(pupa.token_usage), "calls": dict(pupa.call_counts)},
        "blocked_counts": dict(pupa.blocked_counts),
    }
    with open(os.path.join(arguments.output_directory, run_name + ".json"), "w") as output_file:
        json.dump(record, output_file, ensure_ascii=False, indent=1)
    print(run_name, "test mean", round(record["test_mean"], 4), flush=True)


if __name__ == "__main__":
    main()
