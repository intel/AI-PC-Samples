﻿# `Automated Prompt Engineering` Sample

This notebook guides you through the process of running automated prompt engineering on Llama.cpp using DSPy.

| Area                | Description                                                |
| :------------------ | :--------------------------------------------------------- |
| What you will learn | Using DSPy for automated prompt engineering with Llama.cpp |
| Time to complete    | 1 hour                                                     |
| Category            | Concepts and Functionality                                 |

## Purpose

We use DSPy, an automatic prompt engineering framework to create a pipeline for answering riddles and optimize the prompts for that task to improve few-shot performance using Llama.cpp and Intel GPUs (Intel Arc™ Graphics, Intel® Core™ Ultra, or any compatible Intel Graphics).

## Prerequisites

| Optimized for | Description                                                                                                                                                                                                                                                                                                                                         |
| :------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OS            | Ubuntu* 20.04 64-bit and newer or Windows 11 and newer                                                                                                                                                                                                                                                                                              |
| Hardware      | Intel® Core™ Ultra, Intel Arc™ Graphics, or Intel Graphics                                                                                                                                                                                                                                                                                          |
| Software      | [Intel GPU Client Driver For Linux](https://dgpu-docs.intel.com/driver/client/overview.html), [oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html), [MS Visual Studio 2022 for Windows](https://visualstudio.microsoft.com/downloads/?q=build+tools#build-tools-for-visual-studio-2022) |

## Run the `Automated Prompt Engineering` Sample

The command for both Windows and Linux will install the necessary dependencies and run the sample. The sample uses [`pixi`](https://github.com/prefix-dev/pixi/) for environment and task management.

### Windows

1. Download [MS Visual Studio 2022](https://visualstudio.microsoft.com/downloads/?q=build+tools#build-tools-for-visual-studio-2022)
2. Download the [oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). **Make sure to keep the installation directory as `C:\Program Files (x86)\Intel\oneAPI`**.

To run the sample, run the following command inside "Developer Command Prompt for VS 2022"

```cmd
.\tools\run_pixi.cmd run execute
```

### Linux

1. Download the [oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). **Make sure to keep the installation directory as `/opt/intel/oneapi`**.

To run the sample, run the following command:

```bash
./tools/run_pixi.sh run execute
```

## Key Implementation Details

This sample tutorial contains one Jupyter Notebook.

| Notebook                           | Description                                                                                            |
| :--------------------------------- | :----------------------------------------------------------------------------------------------------- |
| `AutomatedPromptEngineering.ipynb` | Gain performance boost using Intel GPU for LLM execution to run automated prompt engineering with DSPy |

### Exporting Dependencies to Conda Environment

**Exporting dependencies to a `conda` environment not necessary for running the sample.**

[`pixi`](https://github.com/prefix-dev/pixi/) relies on the [`conda-forge`](https://conda-forge.org/) project and [`PyPi`](https://pypi.org/) for package retrieval. This means that you can use `pixi` to export dependencies to a `conda` environment and use this exact environment in other projects.

#### Windows

To export the dependencies to a `conda` environment, run the following command:

```
.\tools\run_pixi.ps1 project export conda-environment --platform linux-64 environment.yml
```

This will create a `conda` environment file named `environment.yml`.

#### Linux

To export the dependencies to a `conda` environment, run the following command:

```
./tools/run_pixi.sh project export conda-environment --platform linux-64 environment.yml
```

### Cleaning Up

To clean up the created environment, delete the `.pixi` directory located in the root of the sample.

## Example Output

Users should be able to see an accuracy increase in answering riddles using DSPy for automated prompt engineering. The user will also see GPU utilization as the LLM runs on the GPU.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
