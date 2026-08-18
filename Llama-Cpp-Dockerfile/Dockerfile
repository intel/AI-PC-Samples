FROM ubuntu:24.04@sha256:019e8eb29a85e74d64925745884f2ec79aa27e3feab36353d24656f4d6b89467

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

# Set up oneAPI environment for interactive sessions
RUN echo 'source /opt/intel/oneapi/setvars.sh --force' >> /root/.bashrc

# Install UV
RUN curl --proto '=https' --tlsv1.2 -fsSL \
    https://releases.astral.sh/github/uv/releases/download/0.12.5/uv-installer.sh \
    -o /uv-installer.sh && sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Create virtual environment
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Huggingface Hub
RUN uv pip install "huggingface-hub==0.26.0"

# Install llama-cpp-python with Intel SYCL enabled
ENV CMAKE_ARGS="-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx"
RUN bash -c "source /opt/intel/oneapi/setvars.sh --force && \
    uv pip install llama-cpp-python[server]==0.3.8 -U --force-reinstall --no-cache-dir --verbose"

# Create a non-root user
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /opt/venv /root/.local
USER appuser

# Expose default server port
EXPOSE 8000

# Add health check to monitor server status
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/models || curl -f http://localhost:8000/ || exit 1

# Set default command
ENTRYPOINT ["uv", "run", "python", "-m", "llama_cpp.server"]
