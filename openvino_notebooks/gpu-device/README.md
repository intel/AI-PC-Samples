# What this notebook does

This notebook provides a practical, engineering-focused introduction to using **Intel® GPU devices** with OpenVINO. It walks through device discovery, GPU property inspection, idempotent model download, model loading, GPU-targeted compilation, performance hints, model caching, and simple object detection on a sample video.

The goal is to help ISVs understand how to reliably target Intel GPUs, inspect device capabilities, and keep applications portable across systems with different hardware configurations.

# Hardware & device support

This notebook supports the following devices:

- GPU - **primary target**
- CPU - **fallback**
- NPU - **not required for this notebook**

## Device fallback logic

```python
import openvino as ov

core = ov.Core()
available = core.available_devices
available_gpus = [name for name in available if name == "GPU" or name.startswith("GPU.")]
device = "GPU" if "GPU" in available_gpus else available_gpus[0] if available_gpus else "CPU"
print(f"Selected device: {device}")
```

## Setup

Make sure that `uv` is installed.

**Windows**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**macOS/Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify with:

```bash
uv --version
```

From the repository root:

```bash
cd openvino_notebooks/gpu-device
uv sync
uv run jupyter lab gpu-device.ipynb
```

## Expected output

When the notebook runs successfully, you should see:

- **Device discovery output**, e.g. `['CPU', 'GPU']` or `['CPU', 'GPU', 'NPU']` depending on availability.
- **GPU property output**, including the GPU device name and supported properties.
- **Model download logs**, or a message that the model already exists locally.
- **Model compilation and caching logs**, including first-run and cached compile timings.
- **`benchmark_app` comparisons** for CPU and GPU latency/throughput.
- **Object detection video output** with bounding boxes, FPS, selected device, and performance hint.

Exact devices, timings, and throughput values vary by hardware, driver, and OpenVINO version.

# Tested-on

| OS | Python | OpenVINO | Device(s) | Status |
|----|--------|----------|-----------|--------|
| Windows 11 | 3.12 | 2026.2 | CPU, GPU | Pass |

# Troubleshooting

### GPU not detected

**Cause:** Missing or outdated GPU driver.

**Fix:** Install the latest Intel Graphics driver:
https://www.intel.com/content/www/us/en/download-center/home.html

### Model download warnings

**Cause:** Hugging Face Hub may print progress or deprecation notices.

**Fix:** Safe to ignore if the model files finish downloading and the notebook continues.

### CPU selected even though GPU is present

**Cause:** Permissions, driver issues, or unsupported hardware.

**Fix:**

- Update GPU drivers.
- Verify `ov.Core().available_devices`.
- Ensure the notebook is running inside the `uv` environment for this folder.

### OpenCV or video display issues

**Cause:** Missing codecs or unsupported notebook display environment.

**Fix:**

- Open the generated video file with a local media player.
- Ensure OpenCV is installed through `uv sync`.

# References

- Upstream notebook:
  https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/gpu-device
- OpenVINO GPU documentation:
  https://docs.openvino.ai/nightly/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html
- OpenVINO GPU configuration:
  https://docs.openvino.ai/nightly/get-started/install-openvino/configurations/configurations-intel-gpu.html
- OpenVINO performance hints:
  https://docs.openvino.ai/nightly/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html
