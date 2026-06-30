#!/bin/bash
#SBATCH --job-name=reasoning-live
#SBATCH --partition=cpu          # <-- set to a CPU partition on your cluster
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00          # tunnel stays up for the whole walltime
#SBATCH --output=live-%j.log     # public URL gets printed in here
# No GPU requested on purpose — all model compute happens on OpenRouter.

set -euo pipefail

PORT=8001                        # single port: serves index + game24 + protocols + strategies
PROJECT_DIR="$SLURM_SUBMIT_DIR"  # assumes you `sbatch` from the repo root
cd "$PROJECT_DIR"

# --- preflight: the node needs outbound internet (for ngrok AND OpenRouter) ---
if ! curl -sSf --max-time 10 https://openrouter.ai >/dev/null 2>&1; then
  echo "ERROR: no outbound internet from $(hostname). This node is likely"
  echo "firewalled — ngrok and OpenRouter are both unreachable. Talk to your"
  echo "cluster admin or run the live server off-cluster instead."
  exit 1
fi

# --- ngrok auth (free account token); set NGROK_AUTHTOKEN before submitting ---
# Either: export NGROK_AUTHTOKEN=... in your shell before `sbatch`
#     or: run once `ngrok config add-authtoken <token>` and delete this block.
if [[ -n "${NGROK_AUTHTOKEN:-}" ]]; then
  ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null 2>&1 || true
fi

# --- start the unified demo launcher (binds to localhost; ngrok forwards to it).
#     One port serves everything: index + Game24 tree + scoring-protocols + the
#     live reasoning-strategies demo (and its /events SSE stream). ---
uv run python demo/serve.py --no-open --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!

# --- start the ngrok tunnel agent ---
ngrok http "$PORT" --log=stdout > "ngrok-$SLURM_JOB_ID.log" 2>&1 &
NGROK_PID=$!

# clean up both processes whenever the job ends (walltime, scancel, error)
trap 'kill $SERVER_PID $NGROK_PID 2>/dev/null || true' EXIT

# --- fish the public URL out of ngrok's local API and print it loudly ---
PUBLIC_URL=""
for _ in $(seq 1 30); do
  PUBLIC_URL=$(curl -s http://127.0.0.1:4040/api/tunnels \
    | uv run python -c "import sys,json; t=json.load(sys.stdin)['tunnels']; print(t[0]['public_url'] if t else '')" 2>/dev/null || true)
  [[ -n "$PUBLIC_URL" ]] && break
  sleep 2
done

echo "=================================================================="
if [[ -n "$PUBLIC_URL" ]]; then
  echo "  LIVE DEMO PUBLIC URL:  $PUBLIC_URL"
  echo "  (forwards to the launcher on $(hostname):$PORT — all demos on this one URL:"
  echo "   index, /game24, /protocols, and the live /strategies trace explorer)"
else
  echo "  ngrok did not report a URL — see ngrok-$SLURM_JOB_ID.log"
fi
echo "=================================================================="

# keep the job alive until walltime / scancel; tunnel lives as long as this runs
wait $SERVER_PID
