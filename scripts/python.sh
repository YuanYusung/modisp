#!/usr/bin/env bash
set -euo pipefail
MODISP_PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODISP_PACKAGE_PYTHON="${MODISP_PYTHON:-${MODISP_PACKAGE_ROOT}/.venv-release/bin/python}"
if [[ ! -x "${MODISP_PACKAGE_PYTHON}" ]]; then
  echo 'Install the release wheels into .venv-release, or set MODISP_PYTHON to your installed environment.' >&2
  exit 1
fi
exec "${MODISP_PACKAGE_PYTHON}" "$@"
