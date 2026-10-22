# PyTorch to OpenVINO IR tutorial

This notebook converts the pretrained torchvision RegNetY-800MF image-classification model to OpenVINO Intermediate Representation (IR) and compares PyTorch and OpenVINO inference results.

It demonstrates dynamic-shape and static-shape conversion, TorchScript scripting and tracing, inference on an OpenVINO device, and simple timing comparisons. CPU execution is supported; GPU and NPU can be selected when available.

## Requirements

- Python 3.12 or later
- `uv`
- Internet access for the sample image, ImageNet labels, and pretrained torchvision weights

## Run the notebook

From this directory, create the locked environment and start JupyterLab:

```powershell
uv sync
uv run jupyter lab pytorch-to-openvino.ipynb
```

PyTorch and torchvision are resolved from the CPU wheel index. Launching Jupyter with `uv run` ensures the notebook uses that environment.

## Expected results

On the first run, the notebook downloads the sample image, label map, and RegNetY-800MF weights. It then displays matching top classification results for the PyTorch and converted OpenVINO models and writes generated IR and TorchScript files to `model/`.

## Troubleshooting

### A download fails

Check network access, delete the incomplete file from `data/` or `model/`, and rerun the affected cell.

### A GPU or NPU compilation fails

Select CPU in the device widget. Accelerator execution requires a compatible OpenVINO plugin and driver.

### A notebook import fails

Run `uv sync` in this directory and relaunch it with `uv run jupyter lab pytorch-to-openvino.ipynb`.

## References

- [OpenVINO PyTorch conversion documentation](https://docs.openvino.ai/openvino-workflow/model-preparation/convert-model-pytorch.html)
- [RegNet paper](https://arxiv.org/abs/2003.13678)
