#!/usr/bin/env bash
# Rsyncs s1_occupancy.sh and s2_speculation.sh to the GPU box and prints the
# nohup launch commands. Does NOT ssh in or launch anything itself -- run
# the printed commands manually once you're ready.
#
# Usage: bash harness/deploy_s1s2.sh [host]   (default host: ollama)
set -uo pipefail

HOST="${1:-ollama}"
REMOTE_HARNESS_DIR="/data/projects/q27b_on_12gb/harness"
REMOTE_ROOT_DIR="/data/projects/q27b_on_12gb"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[deploy] rsyncing s1_occupancy.sh + s2_speculation.sh to ${HOST}:${REMOTE_HARNESS_DIR}/"
rsync -avz \
  "$LOCAL_DIR/s1_occupancy.sh" \
  "$LOCAL_DIR/s2_speculation.sh" \
  "${HOST}:${REMOTE_HARNESS_DIR}/"

cat <<EOF

[deploy] rsync complete. To launch on ${HOST}, ssh in and run:

  ssh ${HOST}
  cd ${REMOTE_ROOT_DIR}
  nohup bash harness/s1_occupancy.sh > results/s1_occupancy.nohup.log 2>&1 &
  nohup bash harness/s2_speculation.sh > results/s2_speculation.nohup.log 2>&1 &

NOTE: s1 and s2 both wait_gpu_free before every GPU op, but they are NOT
coordinated with each other -- running both at once means they will fight
over the same 12GB card. Recommended sequencing: launch s1 first, wait for
results/s1_occupancy_done.txt to appear, then launch s2:

  nohup bash harness/s1_occupancy.sh > results/s1_occupancy.nohup.log 2>&1 &
  wait
  nohup bash harness/s2_speculation.sh > results/s2_speculation.nohup.log 2>&1 &
EOF
