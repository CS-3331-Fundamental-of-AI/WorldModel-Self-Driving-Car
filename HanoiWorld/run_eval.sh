#!/bin/bash

# Always run relative to this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure repo root (one level up) and JEPA subdirs are on PYTHONPATH so packages resolve
export PYTHONPATH="${SCRIPT_DIR}/..:${SCRIPT_DIR}/../JEPA:${SCRIPT_DIR}/../JEPA/JEPA_PrimitiveLayer:${SCRIPT_DIR}/../JEPA/JEPA_SecondLayer:${SCRIPT_DIR}/../JEPA/JEPA_ThirdLayer:${PYTHONPATH}"

# Avoid matplotlib/font cache permission issues in sandboxed envs
export MPLCONFIGDIR="${SCRIPT_DIR}/.matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"

# Enable faulthandler to surface native crashes
export PYTHONFAULTHANDLER=1

# Prefer the project venv if available, otherwise fall back to system python
PYTHON_BIN="${SCRIPT_DIR}/../worldmodel31/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate.py" \
    --episodes 5 \
    --logdir test \
    --config merge \
    --device cpu
