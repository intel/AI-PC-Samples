# RAG Chat with Ollama

A Streamlit-based Retrieval-Augmented Generation (RAG) chat application powered by Ollama on Intel® Core™ Ultra Processors.

## Overview

This application demonstrates a RAG (Retrieval-Augmented Generation) system that allows you to chat with documents using Ollama's language models. Upload your documents, and the system will create embeddings and enable semantic search to provide context-aware responses.

## Prerequisites

- Windows 11 or Ubuntu 20.04+
- Intel® Core™ Ultra Processors or Intel Arc™ Graphics
- 16GB+ RAM recommended

## Setup

### 1. Install Ollama

Download and install Ollama:
```powershell
winget install Ollama.Ollama
```

Or download from [https://ollama.com/download](https://ollama.com/download)

### 2. Building Ollama with GPU Support (Vulkan)

For advanced users who want to build Ollama from source with Vulkan GPU acceleration on Windows:

**a. Install Vulkan SDK**
- Download from: [https://vulkan.lunarg.com/sdk/home](https://vulkan.lunarg.com/sdk/home)

**b. Install TDM-GCC**
- Download from: [https://github.com/jmeubank/tdm-gcc/releases/tag/v10.3.0-tdm64-2](https://github.com/jmeubank/tdm-gcc/releases/tag/v10.3.0-tdm64-2)

**c. Install Go SDK**
- Download Go v1.24.9: [https://go.dev/dl/go1.24.9.windows-amd64.msi](https://go.dev/dl/go1.24.9.windows-amd64.msi)

**d. Build Ollama with Vulkan**
```powershell
# Set environment variables
set CGO_ENABLED=1
set CGO_CFLAGS=-IC:\VulkanSDK\1.4.321.1\Include

# Build with CMake
cmake -B build
cmake --build build --config Release -j14

# Build Go binary
go build

# Run Ollama server (Terminal 1)
go run . serve

# Test with a model (Terminal 2)
ollama run gemma3:270m
```

**Note:** This is for advanced users. The pre-built Ollama installation works fine for most users.

### 3. Pull Language Models

Pull the models you want to use:
```bash
ollama pull llama3.2
ollama pull qwen2.5
ollama pull mistral
```

### 4. Install Python Dependencies

Using pip:
```bash
pip install streamlit ollama chromadb sentence-transformers pypdf
```

Using uv (recommended):
```bash
uv pip install streamlit ollama chromadb sentence-transformers pypdf
```

## Running the Application

### 1. Start Ollama Server

If not already running:
```bash
ollama serve
```

### 2. Run the Streamlit App

```bash
# Using Python directly
streamlit run st_rag_chat.py

# Or using uv
uv run streamlit run st_rag_chat.py
```

### 3. Access the App

Open your browser and navigate to:
```
http://localhost:8501
```

## Usage

1. **Upload Documents**: Use the sidebar to upload PDF or text files
2. **Select Model**: Choose your preferred Ollama model from the dropdown
3. **Process Documents**: Click "Process Documents" to create embeddings
4. **Chat**: Ask questions about your documents in the chat interface
5. **View Sources**: See which document sections were used to answer your questions

## Features

- 📄 **Multi-format Support**: Upload PDF and text documents
- 🤖 **Model Selection**: Choose from available Ollama models
- 🔍 **Semantic Search**: Find relevant context using vector embeddings
- 💬 **Context-Aware Chat**: Get answers based on your documents
- 📚 **Source Attribution**: See which parts of documents were used
- 💾 **Persistent Storage**: ChromaDB vector database for efficient retrieval

## Troubleshooting

**Ollama Connection Error:**
- Ensure Ollama is running: `ollama serve`
- Check if models are installed: `ollama list`

**Memory Issues:**
- Use smaller models like `llama3.2:1b` or `qwen2.5:3b`
- Reduce the number of documents processed at once

**Slow Performance:**
- Ensure GPU drivers are up to date
- Use GPU-accelerated Ollama build (Vulkan)
- Try smaller, faster models

## Technical Stack

- **Ollama**: Local LLM runtime
- **Streamlit**: Web interface
- **ChromaDB**: Vector database
- **Sentence Transformers**: Text embeddings
- **Intel Hardware**: Optimized for Intel Core™ Ultra Processors

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.
