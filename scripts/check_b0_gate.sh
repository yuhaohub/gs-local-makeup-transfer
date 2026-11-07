#!/usr/bin/env bash
# Thin B0 gate. The summary schema belongs to src/baselines/b0_aegis_reproduce.py,
# which writes results/b0_aegis_sanity_summary.json and .csv.
# Formal B0 is the merged B0-a record: NeRSemble subject 306,
# status=pipeline_ok, return_code=0. B0-b (FaceScape 527) is cancelled.
# Tier-1 statuses deps_missing and gpu_required exit 0 only when the summary
# is not a FaceScape 527 / B0-b record.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODULE="src/baselines/b0_aegis_reproduce.py"
SUMMARY_JSON="results/b0_aegis_sanity_summary.json"
SUMMARY_CSV="results/b0_aegis_sanity_summary.csv"
# Frozen formal subject. Do not wait for FaceScape 527.
EXPECTED_SUBJECT="306"
RUN_BOOTSTRAP=0

usage() {
  cat <<'EOF'
Usage: scripts/check_b0_gate.sh [--bootstrap]

Formal B0 subject is NeRSemble 306. The merged B0-a summary
(status=pipeline_ok, return_code=0, subject=306) is the formal pass.
B0-b (FaceScape 527) is cancelled and is not on the critical path.

If results/b0_aegis_sanity_summary.json already records that pass, this
gate exits 0 and does not re-run or rewrite the summary. Otherwise it
checks the frozen scaffold, optionally clones third_party/AEGIS, then runs:

  python src/baselines/b0_aegis_reproduce.py

That module writes results/b0_aegis_sanity_summary.json and
results/b0_aegis_sanity_summary.csv. A FaceScape 527 / B0-b record is not
a pass. Other tier-1 deps_missing / gpu_required (also bootstrap_required,
not_run, ok) still exit 0.

  --bootstrap   If third_party/AEGIS is missing, run scripts/00_bootstrap_aegis.sh.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --bootstrap) RUN_BOOTSTRAP=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCAFFOLD_DIRS=(
  baselines/midterm
  configs
  docs
  eval
  makeup
  results
  scripts
  src/baselines
  src/data
  src/eval
  src/methods
  third_party
  transfer
)

missing=()
for rel in "${SCAFFOLD_DIRS[@]}"; do
  if [[ ! -d "$ROOT/$rel" ]]; then
    missing+=("$rel")
  fi
done
if [[ ! -f "$ROOT/scripts/00_bootstrap_aegis.sh" ]]; then
  missing+=("scripts/00_bootstrap_aegis.sh")
fi
if ((${#missing[@]} > 0)); then
  echo "ERROR: expected scaffold is incomplete:" >&2
  for rel in "${missing[@]}"; do
    echo "  - $rel" >&2
  done
  exit 1
fi
echo "OK: scaffold directories present"

# Prefer `python` as the B0 entry; python3 is the same module when `python` is absent.
PY=""
if command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
fi

# formal | cancelled | reject | continue
summary_verdict() {
  local file="$1"
  "$PY" - "$file" "$EXPECTED_SUBJECT" <<'PY'
import json
import sys

path, expected = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path, encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    print("continue")
    raise SystemExit(0)

metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
status = str(data.get("status") or "")
raw_subject = data.get("subject")
subject = "" if raw_subject is None else str(raw_subject).strip()
return_code = metrics.get("return_code")
experiment = str(data.get("experiment_id") or "")
avatar = str(metrics.get("avatar_dir") or "")
target = str(metrics.get("target_image") or "")
facescape = (
    experiment == "B0-b"
    or subject == "527"
    or "FaceScape" in avatar
    or "FaceScape" in target
    or "/527" in avatar
    or avatar.rstrip("/").endswith("/527")
    or "/527" in target
    or target.endswith("/527.png")
)
formal = (
    status == "pipeline_ok"
    and subject == expected
    and str(return_code) == "0"
    and experiment != "B0-b"
    and not facescape
)
if formal:
    print("formal")
elif facescape:
    print("cancelled")
elif status == "pipeline_ok":
    print("reject")
else:
    print("continue")
PY
}

apply_summary_verdict() {
  local verdict="$1"
  case "$verdict" in
    formal)
      echo "B0 gate status=pipeline_ok"
      echo "B0 gate: formal pass (NeRSemble subject=${EXPECTED_SUBJECT}, status=pipeline_ok, return_code=0)"
      echo "B0-a is the formal B0. B0-b (FaceScape 527) is cancelled and is not on the critical path."
      echo "summary: ${SUMMARY_JSON}"
      echo "summary: ${SUMMARY_CSV}"
      exit 0
      ;;
    cancelled)
      echo "ERROR: B0-b (FaceScape 527) is cancelled and is not a B0 pass." >&2
      echo "Formal B0 is NeRSemble subject=${EXPECTED_SUBJECT}, status=pipeline_ok, return_code=0." >&2
      exit 1
      ;;
    reject)
      echo "ERROR: summary is not the formal B0 pass." >&2
      echo "Need status=pipeline_ok, return_code=0, subject=${EXPECTED_SUBJECT}." >&2
      exit 1
      ;;
    continue)
      ;;
    *)
      echo "ERROR: could not read ${SUMMARY_JSON}" >&2
      exit 1
      ;;
  esac
}

# Recorded B0-a is the formal pass. Do not re-run: the reproduce module still
# preflights the upstream makefile default (FaceScape 527) and would rewrite it.
if [[ -n "$PY" && -f "$ROOT/$SUMMARY_JSON" ]]; then
  apply_summary_verdict "$(summary_verdict "$ROOT/$SUMMARY_JSON")"
fi

aegis_dir="$ROOT/third_party/AEGIS"
if [[ ! -d "$aegis_dir/.git" ]]; then
  echo "third_party/AEGIS is missing."
  echo "Run: bash scripts/00_bootstrap_aegis.sh"
  echo "Or:  bash scripts/check_b0_gate.sh --bootstrap"
  do_boot=0
  if [[ "$RUN_BOOTSTRAP" == "1" ]]; then
    do_boot=1
  elif [[ -t 0 && -t 1 ]]; then
    ans=""
    read -r -p "Run scripts/00_bootstrap_aegis.sh now? [y/N] " ans || true
    ans="${ans,,}"
    if [[ "$ans" == "y" || "$ans" == "yes" ]]; then
      do_boot=1
    fi
  else
    echo "Non-interactive: not cloning. Pass --bootstrap to run scripts/00_bootstrap_aegis.sh."
  fi
  if [[ "$do_boot" == "1" ]]; then
    echo "Running scripts/00_bootstrap_aegis.sh"
    bash "$ROOT/scripts/00_bootstrap_aegis.sh"
  fi
fi

finish_status() {
  local status="$1"
  echo "B0 gate status=${status}"
  case "$status" in
    deps_missing|gpu_required)
      echo "tier-1 ${status}: exit 0"
      exit 0
      ;;
    bootstrap_required|not_run|ok)
      exit 0
      ;;
    pipeline_ok)
      echo "ERROR: status=pipeline_ok is the formal B0 pass only for NeRSemble subject=${EXPECTED_SUBJECT} and return_code=0." >&2
      exit 1
      ;;
    *)
      echo "ERROR: unexpected B0 status: ${status}" >&2
      exit 1
      ;;
  esac
}

if [[ -f "$ROOT/$MODULE" && -n "$PY" ]]; then
  echo "Running: python src/baselines/b0_aegis_reproduce.py"
  set +e
  "$PY" "$ROOT/$MODULE"
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "ERROR: python src/baselines/b0_aegis_reproduce.py exited ${rc}" >&2
    exit "$rc"
  fi
  recorded="$(summary_verdict "$ROOT/$SUMMARY_JSON")"
  if [[ "$recorded" != "continue" ]]; then
    apply_summary_verdict "$recorded"
  fi
  status="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$ROOT/$SUMMARY_JSON")"
  echo "summary: ${SUMMARY_JSON}"
  echo "summary: ${SUMMARY_CSV}"
  echo "note: formal B0 subject is NeRSemble ${EXPECTED_SUBJECT}. B0-b (FaceScape 527) is cancelled."
  finish_status "$status"
fi

# Module or interpreter missing: same top-level keys, note points at the module.
bootstrapped=0
if [[ -d "$aegis_dir/.git" ]]; then
  bootstrapped=1
fi
cuda_available=0
if [[ ! -v CUDA_VISIBLE_DEVICES || ( -n "${CUDA_VISIBLE_DEVICES// /}" && "$CUDA_VISIBLE_DEVICES" != "-1" ) ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    if gpu_list="$(nvidia-smi -L 2>/dev/null)" && [[ "$gpu_list" == *GPU* ]]; then
      cuda_available=1
    fi
  fi
fi
uv_present=0
if command -v uv >/dev/null 2>&1; then
  uv_present=1
fi
python_ok=0
if [[ -n "$PY" ]] && "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  python_ok=1
fi

status="bootstrap_required"
if [[ "$bootstrapped" == "1" ]]; then
  if [[ "$uv_present" != "1" || "$python_ok" != "1" ]]; then
    status="deps_missing"
  elif [[ "$cuda_available" != "1" ]]; then
    status="gpu_required"
  else
    status="not_run"
  fi
fi

note="Formal B0 subject is NeRSemble 306 (status=pipeline_ok, return_code=0). B0-b FaceScape 527 is cancelled and is not on the critical path. Run: python src/baselines/b0_aegis_reproduce.py. That writes ${SUMMARY_JSON} and ${SUMMARY_CSV}."
if [[ ! -f "$ROOT/$MODULE" ]]; then
  note="src/baselines/b0_aegis_reproduce.py is not on this tree. ${note}"
elif [[ -z "$PY" ]]; then
  note="python is not on PATH. ${note}"
fi
if [[ -z "$PY" ]]; then
  echo "MISSING: python is not on PATH." >&2
  echo "$note" >&2
  exit 1
fi

mkdir -p "$ROOT/results"
B0_STATUS="$status" \
B0_NOTE="$note" \
B0_BOOTSTRAPPED="$bootstrapped" \
B0_CUDA="$cuda_available" \
"$PY" - "$ROOT/results/b0_gate_status.json" <<'PY'
import json, os, sys
from pathlib import Path

status = os.environ["B0_STATUS"]
blockers = []
if status == "bootstrap_required":
    blockers.append(
        "third_party/AEGIS is missing. Run scripts/00_bootstrap_aegis.sh. "
        "The upstream tree stays gitignored."
    )
elif status == "deps_missing":
    blockers.append("deps_missing: B0 module could not be started")
elif status == "gpu_required":
    blockers.append("gpu_required: CUDA device not available")
payload = {
    "experiment_id": "B0",
    "desc": "AEGIS original de-identification",
    "subject": "306",
    "status": status,
    "bootstrapped": os.environ.get("B0_BOOTSTRAPPED") == "1",
    "metrics": {
        "command": "make mask_avatar",
        "command_attempted": False,
        "return_code": None,
        "log_tail": "",
        "cuda_available": os.environ.get("B0_CUDA") == "1",
        "makefile": None,
        "mask_script": None,
        "embedder": None,
        "avatar_dir": None,
        "target_image": None,
        "blockers": blockers,
    },
    "note": os.environ.get("B0_NOTE", ""),
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY
echo "summary: results/b0_gate_status.json"
echo "note: run python src/baselines/b0_aegis_reproduce.py for ${SUMMARY_JSON} and ${SUMMARY_CSV}"
finish_status "$status"
