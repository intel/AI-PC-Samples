# Document Parsing using DeepSeek-OCR/DeepSeek-OCR-2 and OpenVINO

## DeepSeek-OCR-2

**DeepSeek-OCR-2** is an advanced vision-language model (VLM) designed for efficient and accurate document understanding and optical character recognition (OCR). Building upon the success of DeepSeek-OCR, version 2 introduces enhanced capabilities with a deep vision encoder and a mixture-of-experts decoder architecture. DeepSeek-OCR-2 employs innovative vision-text compression techniques to maintain high accuracy while ensuring manageable computational requirements for processing high-resolution documents.

<img width="731" height="263" alt="image" src="https://github.com/user-attachments/assets/d2fb90cc-554b-4b6b-83eb-72ba7b1490d4" />


More details can be found in the [paper](https://github.com/deepseek-ai/DeepSeek-OCR-2/blob/main/DeepSeek_OCR2_paper.pdf), original [repository](https://github.com/deepseek-ai/DeepSeek-OCR-2) and [model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2).

## DeepSeek-OCR

**DeepSeek-OCR** is a VLM designed as a preliminary proof-of-concept for efficient vision-text compression. DeepSeek-OCR consists of two components: DeepEncoder and DeepSeek3B-MoE-A570M as the decoder. Specifically, DeepEncoder serves as the core engine, designed to maintain low activations under high-resolution input while achieving high compression ratios to ensure an optimal and manageable number of vision tokens.

<img width="691" height="212" alt="image" src="https://github.com/user-attachments/assets/31581dac-fb64-4b21-a1ea-686d7f3191a3" />

More details can be found in the [paper](https://arxiv.org/pdf/2510.18234), original [repository](https://github.com/deepseek-ai/DeepSeek-OCR) and [model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR).

---

In this tutorial we consider how to convert and run DeepSeek-OCR models using [OpenVINO](https://github.com/openvinotoolkit/openvino) and optimize it using [NNCF](https://github.com/openvinotoolkit/nncf).

## Notebook contents
The tutorial consists from following steps:

- Install requirements
- Convert and Optimize model
- Run OpenVINO model inference
- Launch Interactive demo

In this demonstration, you'll create interactive chatbot that can answer questions about provided image's content.

<img width="1704" height="1125" alt="image" src="https://github.com/user-attachments/assets/46862d95-5e2e-4b0c-b5e1-55eebf2c86e5" />

## Installation instructions

This example has an isolated Python environment. Install its dependencies and launch the notebook from this directory:

```powershell
uv sync
uv run jupyter lab deepseek-ocr.ipynb
```

The first model-download cell retrieves the selected DeepSeek model. The first conversion and INT4 compression run is resource-intensive and creates model artifacts that remain local and are excluded from Git.

The notebook supports CPU and, when available, GPU inference. NPU is intentionally excluded from the device selector because it is not validated for this workflow.

## Security notes

The selected model is downloaded before conversion and its architecture requires `trust_remote_code=True`, which loads Python modules from the pinned model snapshot. Review the upstream model repository before running it, and use only the model identifiers provided by this notebook. The Gradio demo stays on localhost by default.

## Experimental status

This notebook demonstrates a model that has not been fully validated with OpenVINO. Results and performance may vary by model variant and hardware.
