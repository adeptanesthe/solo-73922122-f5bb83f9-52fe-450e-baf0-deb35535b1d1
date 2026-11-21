#!/usr/bin/env bash
# Demo walkthrough for the theatre speaker EQ trial API.
# Usage:  ./examples/demo.sh [BASE_URL]     (default http://127.0.0.1:8000)
set -euo pipefail

BASE="${1:-http://127.0.0.1:8000}"
cd "$(dirname "$0")/.."

echo "== GET /health =="
curl -sS "$BASE/health"
echo; echo

echo "== GET /v1/eq/example (a ready-to-post demo payload) =="
curl -sS "$BASE/v1/eq/example"
echo; echo

echo "== POST /v1/eq/optimize  (two seats, opposite reflections -> improved) =="
curl -sS -X POST "$BASE/v1/eq/optimize" \
  -H 'Content-Type: application/json' \
  -d @examples/request_improved.json
echo; echo

echo "== POST /v1/eq/optimize  (baseline already perfect -> no_improvement) =="
curl -sS -X POST "$BASE/v1/eq/optimize" \
  -H 'Content-Type: application/json' \
  -d @examples/request_no_improvement.json
echo; echo

echo "== POST /v1/eq/optimize  (invalid: gain_limit < 1 -> HTTP 422) =="
curl -sS -o /dev/null -w 'HTTP %{http_code}\n' -X POST "$BASE/v1/eq/optimize" \
  -H 'Content-Type: application/json' \
  -d '{"impulse_responses": [[1.0, 0.5], [1.0, -0.5]], "target_delay": 1, "gain_limit": 0.5}'
