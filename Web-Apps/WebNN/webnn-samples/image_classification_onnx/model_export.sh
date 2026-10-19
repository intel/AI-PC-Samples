#!/bin/bash
set -euo pipefail

# Install the fully resolved, hash-pinned export toolchain.
pip install --require-hashes -r requirements-export.txt

# export HF image classification models to ONNX
optimum-cli export onnx --model google/mobilenet_v2_1.0_224 google_mobilenet_v2_1.0_224
#optimum-cli export onnx --model facebook/convnextv2-atto-1k-224 facebook_convnextv2-atto-1k-224
#optimum-cli export onnx --model microsoft/resnet-18 microsoft_resnet-18
