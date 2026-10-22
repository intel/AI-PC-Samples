# OpenVINO API tutorial

This notebook introduces the OpenVINO Runtime API. It loads models from OpenVINO IR, ONNX, PaddlePaddle, TensorFlow, TensorFlow Lite, and PyTorch; inspects model inputs and outputs; runs inference; and reshapes model inputs.

The downloaded example models are cached in the local `model/` directory, so later runs reuse them. You can select any OpenVINO device available on your system; CPU-only execution is supported.

## Requirements

- Python 3.12 or later
- `uv`
- Internet access for the example-model downloads

## Run the notebook

From this directory, create the locked environment and start JupyterLab:

```powershell
uv sync
uv run jupyter lab openvino-api.ipynb
```

The project resolves PyTorch and torchvision from the CPU wheel index. Launching Jupyter with `uv run` ensures the notebook uses this environment.

## Expected results

The notebook displays the detected OpenVINO devices, downloads the small example models on its first run, and shows the input and output shapes for each model format. The final reshape examples demonstrate changes to image size and batch size.

## Troubleshooting

### A model download is interrupted

Delete the incomplete file from `model/` and rerun its download cell.

### The TensorFlow Lite download fails

The TensorFlow Lite example is downloaded through Kaggle Hub. Sign in with `kagglehub.login()` when prompted, or skip that independent section in a network-restricted environment.

### A notebook import fails

Run `uv sync` in this directory and relaunch it with `uv run jupyter lab openvino-api.ipynb`.

## References

- [OpenVINO Runtime API documentation](https://docs.openvino.ai/openvino-workflow/running-inference.html)
- [OpenVINO model preparation documentation](https://docs.openvino.ai/openvino-workflow/model-preparation.html)
