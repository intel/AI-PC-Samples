# Image Classification using WebNN with onnxruntime-web

Sample created using [Microsoft WebNN Tutorial](https://learn.microsoft.com/en-us/windows/ai/directml/webnn-tutorial), with these changes:
- Uses an ONNX model exported from Hugging Face using the bundled Python export helper
- labels are parsed from model's _config.json_ file
- HTML UI modified to select images using image selector
- added device selection priority to try in this order: webnn-npu, webnn-gpu, webgpu, wasm, webnn-cpu.
- includes export scripts that recreate the pinned MobileNetV2 model with the Python ONNX exporter

The sample uses ONNX model exported from HuggingFace to build an image classification system on the web that uses __WebNN with ONNX Runtime Web__. 

| Files | Details
|---|---|
|index.html|HTML for basic UI|
|main.js|main javasvript file to perform image selection and classification using ONNX Runtime Web|
|main_device_priority.js| (optional) same as _main.js_ but with device selection priority added to try webnn-npu first, if not available tries webnn-gpu, then webgpu, wasm, ...|
|model_export.sh|script for Linux to export the pinned google/mobilenet_v2_1.0_224 Hugging Face model|
|model_export.bat|script for Windows to export the pinned google/mobilenet_v2_1.0_224 Hugging Face model|
|export_model.py|Python helper used by the shell and batch scripts to download the pinned model revision and export it to ONNX|
|google_mobilenet_v2_1.0_224/ |ONNX exported model folder generated from _model_export.sh_ script. The folder contains 3 files: [config.json, model.onnx, preprocessor_config.json]|

Can be tested using the "Live Preview" extension for VSCode

### Exporting HuggingFace models to ONNX

The export scripts recreate the included MobileNetV2 model from its pinned Hugging Face revision. After exporting a compatible alternative model, update the value of _path_ in _main.js_.

#### Instructions for Linux

Make sure conda is installed (install mini-forge) and then run commands below in terminal.

To run the _model_export.sh_ script, first create an virtual environment and then run the script:
```
conda create -n onnx_export python=3.10
conda activate onnx_export
source model_export.sh
```

#### Instructions for Windows

Make sure conda is installed (install mini-forge) and then run commands below in terminal.

To run the _model_export.bat_ script, first create an virtual environment and then run the script:
```
conda create -n onnx_export python=3.10
conda activate onnx_export
model_export.bat
```

#### model_export script details

The export scripts install the fully hash-pinned Python dependencies and call _export_model.py_. The helper loads the pinned MobileNetV2 revision with _trust_remote_code=False_, exports its logits model with PyTorch, and validates the generated ONNX file.

The pip packages installed are:
- torch
- torchvision
- transformers
- onnx

Command used by the scripts to export the pinned Hugging Face model to ONNX:
```
python export_model.py
```

The helper exports this pinned Hugging Face model checkpoint to ONNX:
```
google/mobilenet_v2_1.0_224 at revision 75e607b00aeae1297cc89d026a118bce012f5c5a
```
- _google/mobilenet_v2_1.0_224_ is the Hugging Face checkpoint model name.
- _google_mobilenet_v2_1.0_224_ is the directory where the ONNX model is exported.

