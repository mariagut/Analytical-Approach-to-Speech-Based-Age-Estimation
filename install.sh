#!/usr/bin/env bash
# Creates a virtual environment, installs requirements.txt and registers a Jupyter kernel.
# Usage: ./install.sh            (environment in ./.venv)
#        VENV_DIR=/path ./install.sh
set -euo pipefail

cd "$(dirname "$0")"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"
KERNEL_NAME="${KERNEL_NAME:-speech-age}"

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 9), "Python >= 3.9 required"'

"$PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt
"$VENV_DIR/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" \
    --display-name "Python ($KERNEL_NAME)"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "WARNING: ffmpeg not found — required by features.ipynb for .m4a -> .wav conversion." >&2
fi

echo
echo "Done. Activate with:  source $VENV_DIR/bin/activate"
echo "Jupyter kernel:       Python ($KERNEL_NAME)"
