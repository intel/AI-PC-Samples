#!/bin/bash

# install optimum exporters cli tools
pip uninstall optimum
pip install "optimum[exporters]==2.3.0"

# downgrade huggingface_hub to work-around cached_download import error
pip uninstall huggingface_hub
pip install huggingface_hub==0.25.2

# export HF image classification models to ONNX
optimum-cli export onnx --model google/mobilenet_v2_1.0_224 google_mobilenet_v2_1.0_224
#optimum-cli export onnx --model facebook/convnextv2-atto-1k-224 facebook_convnextv2-atto-1k-224
#optimum-cli export onnx --model microsoft/resnet-18 microsoft_resnet-18
