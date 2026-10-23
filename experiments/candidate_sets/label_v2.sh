#!/usr/bin/env bash
# Judge v2 on the events only the channel generator brings into v2 (out/to_label_<u>.json, 408 events, ~85–90 ₽),
# then the v1 vs v2 gain. Approved by the user for 22.10 after the quota reset. Companies run one after another
# (they share the Gemma summary cache file), 32 workers each.
# bash experiments/candidate_sets/label_v2.sh   (from the repo root)
set -euo pipefail
left=$(python -X utf8 common/llm/quota.py | head -1 | sed -E 's/.*left ([0-9.]+) ₽.*/\1/')
echo "quota left: $left ₽"
python -c "import sys; sys.exit(0 if float('$left') >= 110 else 1)" || { echo "less than 110 ₽ left, not starting"; exit 1; }
for u in U_106783 U_150736 U_212593 U_9250; do
  python -X utf8 experiments/judge_v2/judge.py --company "$u" --events "experiments/candidate_sets/out/to_label_$u.json" \
    --inputs gemma_summary --workers 32
done
python -X utf8 common/llm/quota.py | head -1
python -X utf8 experiments/candidate_sets/sets.py
