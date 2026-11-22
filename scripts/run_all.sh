#!/bin/bash
# Task model Qwen3.5-4B, reflection and judge Qwen3.5-9B, both local. Deduplicated split, 250-rollout budget.
# Runs the seed prompts, stock GEPA and the BudgetAPO-style reflection for seeds 0-2, then replay checks,
# then stops this instance by itself so that GPU billing ends even if nobody is watching.
cd /root/gepa_run
source venv/bin/activate
export PYTHONPATH=/root/gepa_run/gepa_src
export LOCAL_BASE_URL=http://127.0.0.1:18741/v1 LOCAL_MODEL_NAME=Qwen/Qwen3.5-4B
export SECOND_BASE_URL=http://127.0.0.1:18743/v1 SECOND_MODEL_NAME=Qwen/Qwen3.5-9B
export LOCAL_ROLES=two_local LOCAL_DECODING=sampled LOCAL_CONCURRENCY=12
export REFLECTION_LOCK_PATH=/root/gepa_run/second_server.lock DEDUPLICATED_SPLIT=1
results=/root/gepa_run/results
mkdir -p "$results"
cd scripts
data_file=/root/gepa_run/data/pupa_new.parquet
date -u +%s > "$results/started_at"
LOCAL_LEDGER="$results/seed_prompts_ledger.jsonl" python3 -u pupa_local_trial.py seed --data-file "$data_file" --output-directory "$results" > "$results/seed_prompts.log" 2>&1 &
for seed in 0 1 2; do
  LOCAL_LEDGER="$results/stock_seed${seed}_ledger.jsonl" python3 -u pupa_local_trial.py optimize --data-file "$data_file" --output-directory "$results" --mode stock --reflection gepa --budget 250 --seed "$seed" > "$results/stock_seed${seed}.log" 2>&1 &
  LOCAL_LEDGER="$results/budgetapo_seed${seed}_ledger.jsonl" python3 -u pupa_local_trial.py optimize --data-file "$data_file" --output-directory "$results" --mode stock --reflection budgetapo --validation-size 40 --budget 250 --seed "$seed" > "$results/budgetapo_seed${seed}.log" 2>&1 &
done
wait
date -u +%s > "$results/runs_finished_at"
# Replay with the servers otherwise idle: every reflection call, then samples of judge and task calls.
cat "$results"/stock_seed*_ledger.jsonl "$results"/budgetapo_seed*_ledger.jsonl > "$results/all_search_calls.jsonl"
python3 -u pupa_local_trial.py replay --ledger "$results/all_search_calls.jsonl" --kind reflection --sample-size 1000 --workers 1 >> "$results/replay.log" 2>&1
python3 -u pupa_local_trial.py replay --ledger "$results/all_search_calls.jsonl" --kind search_judge --sample-size 300 --workers 1 >> "$results/replay.log" 2>&1
python3 -u pupa_local_trial.py replay --ledger "$results/all_search_calls.jsonl" --kind search_judge --sample-size 300 --workers 12 >> "$results/replay.log" 2>&1
python3 -u pupa_local_trial.py replay --ledger "$results/all_search_calls.jsonl" --kind rollout --sample-size 150 --workers 1 >> "$results/replay.log" 2>&1
rm -f "$results/all_search_calls.jsonl"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader > "$results/gpu.txt"
echo finished > "$results/finished"
# Give the Mac up to 25 minutes to copy the results, then stop this instance.
for attempt in $(seq 1 150); do [ -f "$results/fetched" ] && break; sleep 10; done
/root/gepa_run/stop_self.sh
