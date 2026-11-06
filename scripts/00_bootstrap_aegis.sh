#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/third_party"
if [[ ! -d "$ROOT/third_party/AEGIS/.git" ]]; then
  git clone --depth 1 https://github.com/Woleek/AEGIS.git "$ROOT/third_party/AEGIS"
else
  echo "AEGIS already present"
fi
echo "Next: follow third_party/AEGIS/README to install deps, then implement B0 in src/baselines/"
