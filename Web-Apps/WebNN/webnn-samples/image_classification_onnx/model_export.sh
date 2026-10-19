#!/bin/bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIRECTORY"

# Install the fully resolved, hash-pinned export toolchain.
pip install --require-hashes -r requirements-export.txt

# Export the pinned Hugging Face image-classification model to ONNX.
python export_model.py
