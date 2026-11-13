#!/usr/bin/env bash
# Reproduce the README numbers. Usage: scripts/run_real.sh DATA_DIR
# DATA_DIR holds the JSONL files written by scripts/fetch_data.py.
# Runs one command at a time; needs uv and about 1 GB of free memory.
set -euo pipefail
data="${1:?usage: scripts/run_real.sh DATA_DIR}"
out="results"
mkdir -p "$out"
sets="dolly alpaca magpie personas"

for s in $sets; do
  uv run divscore measure "$data/$s.jsonl" --name "$s" --field instruction \
    --budget 20000 --texts 500 --resamples 100 --seed 0 --format json -o "$out/instruction-$s.json"
  uv run divscore measure "$data/$s.jsonl" --name "$s" --field instruction --truncate 15 \
    --budget 20000 --texts 500 --resamples 100 --seed 0 --format json -o "$out/instruction15-$s.json"
  uv run divscore measure "$data/$s.jsonl" --name "$s" --field response --truncate 100 \
    --budget 50000 --texts 500 --resamples 100 --seed 0 --format json -o "$out/response-$s.json"
  uv run divscore openings "$data/$s.jsonl" --name "$s" --field instruction --top 10 \
    --templates 5 --format json -o "$out/openings-$s.json"
done

# magpie responses cut to dolly's mean response length, token metrics only
uv run divscore measure "$data/magpie.jsonl" --name magpie --field response --truncate 47 \
  --budget 50000 --texts 0 --resamples 100 --seed 0 --format json -o "$out/response47-magpie.json"

uv run divscore compare "$data/dolly.jsonl" "$data/alpaca.jsonl" --name-a dolly --name-b alpaca \
  --field instruction --budget 20000 --texts 500 --resamples 100 -o "$out/compare-dolly-alpaca.md"

uv run python scripts/outlier.py "$data" > "$out/outlier-dolly.txt"

uv run divscore curve "$data/dolly.jsonl" "$data/alpaca.jsonl" "$data/magpie.jsonl" \
  "$data/personas.jsonl" --names dolly,alpaca,magpie,personas --field instruction \
  --max-tokens 200000 --orders 5 --plot docs/curve-instruction.png \
  --title "Instruction vocabulary growth" -o "$out/curve-instruction.md"
