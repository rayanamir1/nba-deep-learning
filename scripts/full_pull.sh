#!/bin/bash
# Full historical pull (option 2: two-tier history). Resumable: rerun to retry failures.
#   games + schedule: 2013-14 .. current
#   SBR odds:         2019-10-01 .. today (SBR has no lines before the 2019-20 season)
#   play-by-play + summaries: newest completed season first, back to 2013-14
set -u
cd "$(dirname "$0")/.."
NB=.venv/bin/nbadl
$NB pull-games --first 2013 --last 2026
$NB pull-odds --start 2019-10-01 --end "$(date +%F)"
for y in $(seq 2025 -1 2013); do
  $NB pull-games-detail --season "$y-$(printf '%02d' $(( (y + 1) % 100 )))"
done
echo "FULL PULL COMPLETE $(date)"
