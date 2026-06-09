# `AI Upscaling with Intel® AI Boost NPU` Sample

This notebook guides you through the process of running AI Upscaling models on Intel® AI Boost NPU using OpenVINO.

| Area                | Description                                       |
| :------------------ | :------------------------------------------------ |
| What you will learn | Running AI Upscaling models on NPU using OpenVINO |
| Time to complete    | 15 minutes                                        |
| Category            | Code Optimization                                 |

## Purpose

This sample notebook shows how to leverage Intel® AI Boost Neural Processing Unit (NPU) to boost the performance of AI upscaling. We will use `OpenVINO` to upscale images and video using the `BSRGAN` model.

The sample starts by loading in the `BSRGAN` model and then converts and compiles the model for the NPU using `OpenVINO`. After loading the model, the sample applies the model to a sample image and video to demonstrate the upscaling capabilities of the model as well as the performance benefits of using the NPU.

## Prerequisites

| Optimized for | Description                                                                                                                                                                                      |
| :------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OS            | Windows 11 64-bit (22H2, 23H2) and newer or Ubuntu* 22.04 64-bit (with Linux kernel 6.6+) and newer                                                                                              |
| Hardware      | Intel® Core™ Ultra                                                                                                                                                                               |
| Software      | [Intel® NPU Driver for Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html) or [Linux NPU Driver](https://github.com/intel/linux-npu-driver/releases) |

## Key Implementation Details

This sample tutorial contains one Jupyter Notebook.

| Notebook                      | Description                                                                                |
| :---------------------------- | :----------------------------------------------------------------------------------------- |
| `AI_Upscaling_With_NPU.ipynb` | Gain performance boost during inference using Intel® AI Boost Neural Processing Unit (NPU) |

## Run the `AI Upscaling with Intel® AI Boost NPU` Sample

### Using `uv`

The sample uses [uv](https://docs.astral.sh/uv/) for environment management. Steps to install `uv` can be found [here](https://docs.astral.sh/uv/getting-started/installation/). 

1. Setup the environment

    ```bash
    # ffmpeg OS binary
    conda install ffmpeg

    # python dependencies
    uv sync
    ```


5. Execute the Jupyter Notebook via command line

    ```bash
    uv run jupyter nbconvert --execute --to notebook --inplace --debug AI_Upscaling_With_NPU.ipynb
    ```

## Running the Streamlit App

To launch the interactive Streamlit demo, run the following command from the project directory:

```bash
uv run streamlit run NPU_Upscaling_demo.py
```

---

## Opening Jupyter Notebook

If you choose to run the Jupyter Notebook directly, you can open the Jupyter Notebook by running the following command:

### Using `uv`

```bash
uv run jupyter notebook AI_Upscaling_With_NPU.ipynb
```

### Using VScode

Open the notebook and select the appropriate Python interpreter:

- For Windows: Open the notebook and select `.venv\Scripts\python.exe` as the Python interpreter.
- For Linux: Open the notebook and select `.venv/bin/python` as the Python interpreter.

## Example Output

Users should be able to see a side-by-side comparison of the original and upscaled images and videos as well as the performance comparison between running the model on the CPU and NPU.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).