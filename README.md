# GEPA on PUPA under a tight budget: pilot experiments

Pilot experiments behind the OpenRSI-Index task proposal
[Budget-Aware Reflection for GEPA Prompt Optimization](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/369).

The question: with only 250 rollouts, does changing what GEPA's reflection step is shown, and how much of
the budget goes to validation, give better held-out prompts than stock GEPA?

These are pilots, not a reproduction of the proposed protocol. Each configuration was run once per seed.
I ran them in November 2026 with the help of an AI coding assistant (Claude Code).

## Setup

| Item | Value |
| --- | --- |
| Optimizer | [GEPA](https://github.com/gepa-ai/gepa) at `ce5b0d6e3d6af6c6dbec6de2c54536bbb961d309` |
| Data | [`Columbia-NLP/PUPA`](https://huggingface.co/datasets/Columbia-NLP/PUPA) at `9981b49b6ced0033988a224b6712895ebf119294`, `PUPA_New.csv` |
| Split | Exact duplicate rows removed (539 of 664 remain); whole conversations assigned to 185 held-out, 88 validation, 95 search rows. See `split/split_manifest.json` |
| Program | Two prompts, three steps, adapted from the [DSPy GEPA PAPILLON tutorial](https://github.com/stanfordnlp/dspy/blob/6ae6736d80b8c59550eddba61baa866be2e6712b/docs/docs/tutorials/gepa_papillon/index.ipynb): write a redacted request, answer it in place of the external model, write the final response |
| Score | `(quality + (1 - leakage)) / 2` per row, averaged over the held-out rows |
| Task model | `Qwen/Qwen3.5-4B`, temperature 0.6 |
| Reflection model and judge | `Qwen/Qwen3.5-9B`, reflection temperature 1.0, judging temperature 0 |
| Serving | vLLM 0.32.0, BF16, thinking disabled, both models on one RTX 5880 Ada (48 GB) |
| Budget | 250 rollouts per run. A rollout is one execution of the program on one row plus one judge pass |

Two optimizers were compared, three optimizer seeds each:

- **Stock GEPA**: the defaults of `gepa.optimize` (full 88-row validation, minibatches of 3, one proposal
  per iteration).
- **Changed policy**: a BudgetAPO-style reflection template, at most three short failure excerpts as
  feedback, and a 40-row validation subset.

The changed policy alters several things at once. It shows that stock GEPA can be improved within this
budget; it is not a controlled comparison of any single change.

## Result 1: held-out scores

Mean over the 185 held-out rows; the three values in brackets are optimizer seeds 0, 1, 2.

| Leakage measure | Seed instructions | Stock GEPA | Changed policy |
| --- | --- | --- | --- |
| Tutorial's single-call count (9B judge, as run) | 0.720 | 0.813 (0.799 / 0.822 / 0.817) | 0.879 (0.820 / 0.920 / 0.898) |
| PAPILLON per-item presence check (9B judge) | 0.704 | 0.765 (0.736 / 0.820 / 0.740) | 0.884 (0.820 / 0.922 / 0.911) |
| Presence check, or verbatim occurrence (9B judge) | 0.588 | 0.621 (0.630 / 0.718 / 0.516) | 0.857 (0.760 / 0.915 / 0.896) |
| Presence check, or verbatim occurrence (27B judge) | 0.589 | 0.630 (0.640 / 0.728 / 0.522) | 0.873 (0.773 / 0.934 / 0.911) |

- The first row is what the runs themselves measured and optimized against. The other rows re-score the
  same held-out outputs afterwards, with hosted copies of the judge models, so those prompts were not
  optimized against the measure that scores them.
- Under the last measure with the 9B judge, the changed policy beats stock GEPA by 0.236 (paired 95%
  bootstrap interval over rows: 0.210 to 0.261).
- Stock GEPA used 182 to 185 rollouts and could afford exactly one accepted proposal: a full validation
  pass costs 88 rollouts. The changed policy used 208 to 239 rollouts and accepted three or four.
- Search used 1.6 to 2.2 million tokens per run.

## Result 2: the leakage judge misses text that is literally there

The PAPILLON repository asks a judge, for each PII item, whether it is present in the request. I compared
those answers with a plain check of whether the item appears verbatim in the request.

| Requests produced by | PII items appearing verbatim | Judged absent by 9B | Judged absent by 27B |
| --- | --- | --- | --- |
| Seed instructions | 285 | 80 (28%) | 46 (16%) |
| Stock GEPA prompts | 691 | 339 (49%) | 273 (40%) |
| Changed-policy prompts | 169 | 51 (30%) | 38 (22%) |

A common cause: the prompts found by stock GEPA make the task model append a "sanitization log" that
lists the names it removed. The log itself sends those names to the external model, and the judge then
answers that they are absent because the log says they were removed. A larger judge helps only a little.

This is why the proposal counts an item as leaked if it appears verbatim *or* the judge finds it present.
With that rule the 9B and 27B judges give nearly the same means (last two rows of Result 1).

## Result 3: can a logged call be reproduced exactly?

Every model call of the runs was logged and some were replayed afterwards on the same server, with a
fixed sampling seed.

| Calls replayed | Identical output |
| --- | --- |
| All 25 reflection calls (9B, temperature 1.0), each issued alone | 25 of 25 |
| 300 judge calls (9B, temperature 0), one at a time | 295 of 300 |
| 300 judge calls, 12 at a time | 294 of 300 |
| 150 task-model calls (4B, temperature 0.6), one at a time | 35 of 150 |

Long sampled outputs produced under concurrent serving are not reproducible; short or isolated calls are.

## Result 4: which task model leaves room to improve?

An earlier study on a previous split (221 held-out rows, exact duplicates not yet removed), with
`qwen3.8-27b` as reflection model and judge through a hosted API. The 9B task model was hosted; the 4B
and 2B task models were served with vLLM on a rented RTX 4090.

| Task model | Seed instructions | Stock GEPA (3 seeds) | Changed policy (3 seeds) |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-9B` | 0.790 | 0.890 | 0.930 |
| `Qwen/Qwen3.5-4B` | 0.798 | 0.816 | 0.866 |
| `Qwen/Qwen3.5-2B` | 0.759 | 0.761 | 0.785 |

The 9B task model is close to saturated, and with the 2B model the difference between the two optimizers
is within noise. The 4B model was chosen for the proposal.

## Files

| Path | Content |
| --- | --- |
| `scripts/pupa_gepa_pilot.py` | The program, the metric, the GEPA adapter, and the seed-instruction evaluation |
| `scripts/pupa_gepa_configured.py` | Stock GEPA and the changed policy under a rollout budget |
| `scripts/pupa_local_trial.py` | Runs the two scripts above against local OpenAI-compatible servers; the deduplicated split; the call log; the replay check |
| `scripts/rejudge_from_ledgers.py` | Rebuilds held-out outputs from a call log |
| `scripts/presence_judge.py` | Re-scores leakage with the PAPILLON per-item presence check |
| `scripts/run_all.sh` | The launch script used on the rented GPU for the runs of Result 1 and Result 3 |
| `split/split_manifest.json` | Row numbers of the held-out, validation, and search rows |
| `results/local_4b_9b/` | One JSON per run of Result 1 (prompts found, validation scores, per-row held-out scores, token use), and the replay log |
| `results/leakage_rescoring.json` | Aggregates behind Results 1 and 2 |
| `results/task_model_size_study/` | One JSON per run of Result 4 |

Raw call logs are not included: they are large and repeat the dataset text.

## Running it

The scripts read a parquet copy of `PUPA_New.csv` and need the GEPA source on `PYTHONPATH`. Start two
OpenAI-compatible servers on one GPU, one after the other; the second must be given no more than the
memory still free when it starts.

```bash
vllm serve Qwen/Qwen3.5-4B --port 18741 --dtype bfloat16 --gpu-memory-utilization 0.34 \
  --max-model-len 32768 --max-num-seqs 48 --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
vllm serve Qwen/Qwen3.5-9B --port 18743 --dtype bfloat16 --gpu-memory-utilization 0.66 \
  --max-model-len 32768 --max-num-seqs 48 --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'

export PYTHONPATH=/path/to/gepa/src
export LOCAL_BASE_URL=http://127.0.0.1:18741/v1 LOCAL_MODEL_NAME=Qwen/Qwen3.5-4B
export SECOND_BASE_URL=http://127.0.0.1:18743/v1 SECOND_MODEL_NAME=Qwen/Qwen3.5-9B
export LOCAL_ROLES=two_local LOCAL_DECODING=sampled LOCAL_CONCURRENCY=12 DEDUPLICATED_SPLIT=1

cd scripts
python3 pupa_local_trial.py seed --data-file pupa_new.parquet --output-directory out
python3 pupa_local_trial.py optimize --data-file pupa_new.parquet --output-directory out \
  --mode stock --reflection gepa --budget 250 --seed 0
python3 pupa_local_trial.py optimize --data-file pupa_new.parquet --output-directory out \
  --mode stock --reflection budgetapo --validation-size 40 --budget 250 --seed 0
```

On the pilot GPU, seven such runs side by side (the seed-instruction evaluation and six optimizations,
each followed by a 185-row evaluation) finished in 70 minutes.

## Limits

- One run per configuration and seed. Run-to-run differences are large: the three changed-policy runs
  range from 0.760 to 0.915 under the proposal's leakage measure.
- The held-out rows were used many times while these pilots were developed.
- The re-scoring in Results 1 and 2 used hosted copies of the judge models, which may differ slightly
  from locally served weights.
- Verbatim matching does not catch PII that is rephrased, and it also counts celebrity names, which the
  original PAPILLON check excuses.
