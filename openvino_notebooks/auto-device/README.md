# What this notebook does

This notebook shows how to do inference with Automatic Device Selection (AUTO) in OpenVINO and gives a high-level overview of how AUTO chooses the most suitable execution device based on model and hardware availability.

It demonstrates how to compile a model with AUTO, compare first inference latency (model compilation time + first inference time) between GPU and AUTO, and show the difference between THROUGHPUT and LATENCY performance hints.

This notebook provides a practical, engineering-focused introduction to deploying one application across heterogeneous systems (CPU/GPU/NPU) with minimal device-specific branching. It includes explicit device discovery, deterministic fallback behavior, idempotent model preparation/loading, and repeatable runtime measurements for first-inference and steady-state execution.

# Hardware & device support

This notebook supports the following devices:

- CPU — **supported (fallback)**
- GPU — **supported if available**
- NPU — **supported if available**
- AUTO — **primary execution mode**

# Setup

Make sure that **uv** is installed.

## Windows
irm https://astral.sh/uv/install.ps1 | iex

## macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

Verify installation:
uv --version

From the auto-device notebook folder:

uv sync

uv run jupyter lab auto-device.ipynb

The sample image is included with the notebook. An internet connection is required only on the first run to download the pretrained ResNet-50 weights and create model/resnet50.xml.

# Expected output

When the notebook runs successfully, you should see:

- **Device discovery and selection output**, e.g.
  `Available devices: ['CPU', 'GPU.0']`
  `Selected device: GPU`

- **Idempotent model preparation logs**, e.g.
  `IR model saved to model/resnet50.xml` on first run, then
  `Read IR model from model/resnet50.xml` on subsequent runs.

- **Compilation and first-inference timing output**, e.g.
  `Time to load model on GPU device and get first inference: 0.15 seconds.`

- **Performance-hint measurements**, with throughput/latency metrics printed over multiple intervals.

For systems without accelerator devices, CPU fallback output is expected and valid.

# Tested-on

| OS | Python | OpenVINO | Device(s) | Status |
|----|--------|----------|-----------|--------|
| Windows 11 | 3.12 | 2026.2 | CPU, GPU | Pass |

# Troubleshooting

### AUTO always selects CPU
**Cause:** GPU/NPU plugin not available, unsupported hardware, or driver/runtime mismatch.
**Fix:** Verify `ov.Core().available_devices`, then update Intel GPU/NPU drivers and confirm execution inside the correct uv environment.

### First inference is much slower than later runs
**Cause:** Model compilation and backend warm-up overhead during first execution.
**Fix:** Compare first-run timing to repeated runs; use multiple iterations for steady-state performance analysis.

### Notebook import errors (openvino/torchvision/notebook utils)
**Cause:** Environment not synced or wrong interpreter selected.
**Fix:** Run `uv sync` in this directory and launch with `uv run jupyter lab auto-device.ipynb`.

### AUTO behavior differs across machines
**Cause:** Different available hardware backends or plugin versions.
**Fix:** Log `core.available_devices` at startup and keep OpenVINO/runtime stack consistent across systems.

# References

- Upstream OpenVINO auto-device notebook:
  https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/auto-device/auto-device.ipynb

- OpenVINO AUTO device documentation:
  https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html

- OpenVINO performance hints:
  https://docs.openvino.ai/2024/openvino-workflow/running-inference/performance-hints.html
