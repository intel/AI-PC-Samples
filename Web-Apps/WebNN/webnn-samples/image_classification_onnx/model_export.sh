#!/bin/bash

# install optimum exporters cli tools
pip uninstall optimum
pip install "optimum[exporters]==2.3.0" \
    --hash=sha256:3e9b217b4ab21fd4cf894a987002ee7d3626114e009592babf084c2f1a0f3b5f \
    --hash=sha256:aa96ad535a5cec68d12c6372574125452284632fe13699633a61e8bbfb09c4df

# downgrade huggingface_hub to work-around cached_download import error
pip uninstall huggingface_hub
pip install "huggingface_hub==0.25.2" \
    --hash=sha256:1897caf88e7f97fe0110603d8f66ac264e3ba6accdf30cd66cc0fed5282ad25 \
    --hash=sha256:a1014ea111a5f40ccd23f7f7ba8ac46e20fa3b658ced1f86a00c5c06ec6423c

# export HF image classification models to ONNX
optimum-cli export onnx --model google/mobilenet_v2_1.0_224 google_mobilenet_v2_1.0_224
#optimum-cli export onnx --model facebook/convnextv2-atto-1k-224 facebook_convnextv2-atto-1k-224
#optimum-cli export onnx --model microsoft/resnet-18 microsoft_resnet-18
