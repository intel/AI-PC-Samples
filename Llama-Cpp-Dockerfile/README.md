# llama-cpp-python with Intel GPU (SYCL) Docker

This repository provides a Dockerfile to build a containerized environment for running [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) with Intel GPU acceleration using SYCL (oneAPI). The image is based on Ubuntu 25.10 and includes all necessary Intel GPU drivers, oneAPI Base Toolkit, and Python dependencies for efficient LLM inference.

## Table of Contents

- [Features](#features)
- [Build the docker image](#build-the-docker-image)
- [Run the container](#run-the-container)
  - [Default container](#default-container)
  - [Custom model or arguments](#custom-model-or-arguments)
  - [Mount a local directory and run a model from it](#mount-a-local-directory-and-run-a-model-from-it)
  - [Run bash shell in the container](#run-bash-shell-in-the-container)
- [Notes](#notes)
- [License](#license)

---

## Features

- **Intel GPU support**: Installs [Intel GPU drivers](https://dgpu-docs.intel.com/driver/client/overview.html) and [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) for SYCL acceleration.
- **llama-cpp-python**: Builds and installs with Intel SYCL support for fast inference.
- **Python environment**: Uses [uv](https://github.com/astral-sh/uv) for fast Python dependency management.
- **Ready-to-use server**: Runs the llama-cpp-python server by default.

---

## Build the Docker Image

Clone this repository and build the Docker image:

```sh
# Build the Docker image (replace <tag> with your preferred tag)
docker build -t llamacpp-intel-sycl:latest .
```

---

## Run the Container

### Default container

To run the server with a default HuggingFace model (Qwen/Qwen2-0.5B-Instruct-GGUF):

```sh
docker run -it --rm \
	--device /dev/dri \
	-p 8000:8000 \
	llamacpp-intel-sycl:latest
```

- `--device /dev/dri` exposes the Intel GPU to the container.
- `-p 8000:8000` maps the server port.

### Custom model or arguments

The default model and arguments can be overridden as needed:

```sh
docker run -it --rm \
	--device /dev/dri \
	-p 8000:8000 \
	llamacpp-intel-sycl:latest \
	--n_gpu_layers -1 \
	--hf_model_repo_id <hf-repo> --model <model-file>
```

- `--n_gpu_layers -1` All layers of the model are offloaded to your GPU.
- Replace `--hf_model_repo_id <hf-repo>` with the HuggingFace model repo id to use.
- Replace `--model <model-file>` with the HuggingFace model file to use. 

### Mount a local directory and run a model from it

To use a model file stored on the host machine, mount the directory containing the model into the container and specify the path to the model file. For example, if the model is located in `/path/to/models` on the host:

```sh
docker run -it --rm \
	--device /dev/dri \
	-p 8000:8000 \
	-v /path/to/models:/models \
	llamacpp-intel-sycl:latest \
	--n_gpu_layers -1 \
	--model /models/<model-file>
```

- `-v /path/to/models:/models` mounts the local directory into the container at `/models`.
- Replace `<model-file>` with the actual filename of the model inside `/path/to/models`.

This approach can be combined with other arguments as needed.

### Run bash shell in the container

To override the entry point and start the container with the bash shell, run the following command:

```sh
docker run -it --rm \
	--device=/dev/dri \
	--net host \
	--entrypoint /bin/bash \
	llamacpp-intel-sycl:latest

```
- `--net host` Instead of creating a own different network in the container, it is going to use host network.
- To override the default server entry point and start the container with a Bash shell instead of the default command, we can use  `--entrypoint /bin/bash`.

---

## Notes

- Make sure your host system has an Intel GPU and the necessary drivers installed.
- For more information about supported models, server options, and how to call inference endpoints, see the [llama-cpp-python OpenAI Server documentation](https://llama-cpp-python.readthedocs.io/en/latest/server/).
- If you're behind some proxies, please update the `config.json` file with the correct proxy settings before running the container.
```sh
## ~/.docker/config.json
{
"proxies":{
	"default": {
		"httpProxy": <your-proxy-details>,
		"httpsProxy": <your-proxy-details>,
		"noProxy": <your-proxy-details>
		}
	}
}
```

---

## License:

View the LICENSE file for the repository [here](./LICENSE).
