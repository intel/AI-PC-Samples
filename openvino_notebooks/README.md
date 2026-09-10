# OpenVINO Notebooks

This folder will contain ten OpenVINO notebooks cleaned up and modernized using UV installation.

The original examples and inference workflows are preserved. The work focused on making each notebook easier to set up, understand, reproduce, and validate.

## Notebooks

| Notebook | Status | What it covers |
| --- | --- | --- |
| `hello-npu` | Available | Getting started with OpenVINO on an Intel NPU |
| `gpu-device` | Available | Discovering and using an Intel GPU with OpenVINO |
| `auto-device` | Planned | Letting OpenVINO select the best available device |
| `async-api` | Planned | Running inference with the asynchronous API |
| `openvino-tokenizers` | Planned | Using OpenVINO Tokenizers in an inference pipeline |
| `deepseek-ocr` | Planned | Running DeepSeek OCR with OpenVINO |
| `text-to-speech-genai` | Planned | Generating speech with OpenVINO GenAI |
| `pytorch-post-training-quantization-nncf` | Planned | Applying post-training quantization with NNCF |
| `pytorch-to-openvino` | Planned | Converting a PyTorch model to OpenVINO |
| `openvino-api` | Planned | Learning the main OpenVINO Runtime API |

## What was modernized

- Reproducible environments and lockfiles using `uv`
- OpenVINO `2026.2.*` with stable Python and PyTorch packages
- Cleaner dependencies, setup steps, explanations, and notebook cells
- Dependency auditing and headless CI validation
- No unnecessary changes to the original inference logic

## Run a notebook

Each notebook is maintained as its own `uv` project:

```bash
cd openvino_notebooks/<notebook-name>
uv sync
uv run jupyter lab <notebook-name>.ipynb
```

Hardware-dependent notebooks require the matching Intel device, driver, and runtime support.

## Source

These examples are based on notebooks from the `openvinotoolkit/openvino_notebooks` project and were modernized for reproducibility, maintainability, and ISV use.
