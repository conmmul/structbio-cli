#!/usr/bin/env bash
# One command to make this workstation ready to run jobs.
#
#   ./install.sh
#
# Creates a private Python environment for structbio, installs it, then runs
# `structbio setup`, which finds the scientific software already on the
# machine, writes the short tool commands, and puts them on your PATH.
#
# Safe to run again: an existing environment is reused, never replaced. The
# tool commands name this environment's interpreter outright, so deleting it
# would break them.

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
venv="$here/.venv"
minimum="3.10"

say() { printf '%s\n' "$*"; }
fail() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

find_python() {
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null
    then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

say "structbio installer"
say ""

if [ -x "$venv/bin/python" ]; then
  say "Environment       $venv (already there, reusing it)"
else
  python=$(find_python) || fail "no Python $minimum or newer was found. Ask whoever runs this workstation to install one."
  say "Python            $python ($("$python" -c 'import platform; print(platform.python_version())'))"
  "$python" -m venv "$venv" || fail "could not create the environment at $venv"
  say "Environment       $venv (created)"
fi

say "Installing structbio..."
"$venv/bin/python" -m pip install --quiet --upgrade pip >/dev/null
"$venv/bin/python" -m pip install --quiet -e "$here" || fail "the installation failed; the output above says why"
say ""

exec "$venv/bin/structbio" setup "$@"
