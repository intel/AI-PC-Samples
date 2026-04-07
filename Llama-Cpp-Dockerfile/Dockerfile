FROM ubuntu:25.10

# Set noninteractive mode to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages and prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    gpg-agent \
    ca-certificates \
    python3 \
    python3-pip \
    git \
    curl \
    gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Intel GPU drivers
RUN curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" \
    > /etc/apt/sources.list.d/intel-gpu-noble.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    libze-intel-gpu1 \
    libze1 \
    intel-opencl-icd \
    clinfo \
    intel-gsc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Intel oneAPI Base Toolkit
RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
    gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    intel-oneapi-base-toolkit && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -fsSL https://astral.sh/uv/install.sh -o /uv-installer.sh && \
    sh /uv-installer.sh && \
    cp /root/.local/bin/uv /usr/local/bin/uv && \
    rm /uv-installer.sh

# Create virtual environment
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create group and user
RUN groupadd -g 993 render && \
    useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/ /opt/venv/ && \
    usermod -a -G video,render user

# Add oneAPI environment to user's bashrc
RUN echo 'source /opt/intel/oneapi/setvars.sh --force' >> /home/user/.bashrc

# Install Huggingface Hub
RUN uv pip install huggingface-hub

# Install llama-cpp-python with Intel SYCL enabled
ENV CMAKE_ARGS="-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx"
RUN bash -c "source /opt/intel/oneapi/setvars.sh --force && \
    uv pip install llama-cpp-python[server]==0.3.8 -U --force-reinstall --no-cache-dir --verbose"

# Add healthcheck to satisfy Trivy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import llama_cpp; print('OK'); print(llama_cpp.__version__)" || exit 1

# Switch to non-root user
USER user

ENTRYPOINT ["/bin/bash", "-c", "source /opt/intel/oneapi/setvars.sh && uv run python -m llama_cpp.server \"$@\"", "--"]
CMD ["--hf_model_repo_id", "Qwen/Qwen2-0.5B-Instruct-GGUF", "--model", "*q8_0.gguf", "--n_gpu_layers", "-1"]

# CMD ["/bin/bash"]
