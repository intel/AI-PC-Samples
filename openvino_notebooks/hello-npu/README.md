# What this notebook does

This notebook provides a practical, engineering‑focused introduction to using the **Intel® AI Boost NPU** with OpenVINO. It walks through device discovery and fallback, compiling a model for NPU execution, enabling both **UMD model caching** and **OpenVINO model caching**, and measuring performance under different **performance hints** (latency vs throughput). It also demonstrates how to benchmark NPU vs CPU using `benchmark_app` with consistent configurations.

The goal is to help ISVs understand how to reliably target the NPU, how caching affects compilation time, and how performance characteristics differ across devices.

# Hardware & device support

This notebook supports the following devices:

- CPU — **supported**
- GPU — **supported if available**
- NPU — **supported (Intel® AI Boost)**

## Device fallback logic

```python
import openvino as ov

core = ov.Core()
available = core.available_devices
device = (
    "NPU" if "NPU" in available
    else "GPU" if "GPU" in available
    else "CPU"
)
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
cd openvino_notebooks/hello-npu
uv sync
uv run jupyter lab hello-npu.ipynb
```

## Expected output

When the notebook runs successfully, you should see:

- **Device selection output**, e.g.
  `Selected device: NPU` or `Selected device: CPU` depending on availability.

- **Model compilation logs**, including:
  - First‑run compilation time
  - Subsequent runs showing reduced time due to **UMD model caching**
  - Subsequent runs showing reduced time due to **OpenVINO model caching**

- **Performance measurements** for:
  - Latency hint on NPU vs CPU
  - Throughput hint on NPU vs CPU

- **`benchmark_app` comparisons**, including:
  - NPU latency vs CPU latency
  - NPU throughput vs CPU throughput
  - NPU with UMD caching enabled

# Tested‑on

| OS | Python | OpenVINO | Device(s) | Status |
|----|--------|----------|-----------|--------|
| Windows 11 | 3.12 | 2026.2 | CPU, NPU | Pass |

# Troubleshooting

- **NPU not detected**
  *Cause:* Missing or outdated NPU driver.
  *Fix:* Install the correct driver:
  https://docs.openvino.ai/nightly/get-started/install-openvino/configurations/configurations-intel-npu.html

- **Compilation time is slow on first run**
  *Cause:* No UMD or OpenVINO cache yet.
  *Fix:* Re‑run the cell; subsequent runs should be significantly faster.

- **CPU selected even though NPU is present**
  *Cause:* Kernel version < 6.6 on Linux, or missing Windows runtime.
  *Fix:* Update OS or drivers; verify `ov.Core().available_devices`.

- **benchmark_app shows inconsistent results**
  *Cause:* Mixing latency and throughput hints or inconsistent batch sizes.
  *Fix:* Ensure identical parameters across devices when comparing.

# References

- Upstream notebook:
  https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-npu
- OpenVINO NPU documentation:
  https://docs.openvino.ai/nightly/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
- OpenVINO performance hints:
  https://docs.openvino.ai/nightly/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html
