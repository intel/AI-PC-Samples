"""
Podcast RAG Streamlit Application
Based on Podcast_RAG.ipynb notebook structure
"""

import os
import io
import re
import math
import json
import time
import hashlib
import warnings
import torch
import logging
import requests
import feedparser
import torchaudio
import numpy as np
import streamlit as st
import soundfile as sf

# Suppress known third-party deprecation warnings that are not actionable from app code
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message='Field name "schema"', category=UserWarning)
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass
from datetime import datetime, timezone, timedelta
from teapotai import TeapotAI
from collections import Counter
from urllib.parse import urlparse
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

try:
    import ollama
except Exception:
    ollama = None

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

TRANSCRIPT_CACHE_DIR = os.path.join(".cache", "podcast_transcripts")
AUDIO_RESPONSE_CACHE_DIR = os.path.join(".cache", "podcast_audio_responses")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "how", "i",
    "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "our", "so",
    "that", "the", "their", "them", "there", "these", "they", "this", "to", "was", "we",
    "what", "when", "where", "which", "who", "why", "will", "with", "you", "your", "here"
}
HOST_NAME_BLOCKLIST = {
    "great", "that", "this", "today", "podcast", "episode", "science", "focus", "first",
    "new", "help", "telling", "running", "around", "picking", "kids", "days", "right", "now",
    "perfect", "memory", "officer", "photos", "suspect", "witness", "crime", "none", "guys"
}
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
# OVMS agentic launcher (ovms_agentic_setup.ps1) serves an OpenAI-compatible API at /v3 on port 8000 by default
DEFAULT_OVMS_ENDPOINT = "http://localhost:8000/v3"
DEFAULT_OVMS_MODEL = "Qwen3-8B"
LLM_BACKEND_OVMS = "OpenVINO Model Server (OVMS)"
LLM_BACKEND_OLLAMA = "Ollama"
LLM_BACKENDS = [LLM_BACKEND_OVMS, LLM_BACKEND_OLLAMA]

# Models known to burn a large hidden chain-of-thought budget (reasoning_content) before
# emitting a visible answer — these need a much higher max_tokens floor than a normal chat model.
REASONING_MODEL_TOKEN_FLOOR = 3072
REASONING_MODEL_KEYWORDS = ("gpt-oss", "deepseek-r1", "qwq", "gemma-4", "gemma4")


def model_needs_reasoning_headroom(model_name):
    """True if model_name is a 'thinking'/reasoning model (Qwen 3.5+, gpt-oss, gemma4, etc.)."""
    if not model_name:
        return False
    name = model_name.lower()
    if any(keyword in name for keyword in REASONING_MODEL_KEYWORDS):
        return True
    match = re.search(r"qwen[\s_-]*(\d+(?:\.\d+)?)", name)
    if match:
        try:
            return float(match.group(1)) >= 3.5
        except ValueError:
            return False
    return False


def resolve_effective_max_tokens(model_name, requested_max_tokens):
    """Auto-raise the token budget for reasoning models so thinking doesn't crowd out the answer."""
    if model_needs_reasoning_headroom(model_name):
        return max(requested_max_tokens, REASONING_MODEL_TOKEN_FLOOR)
    return requested_max_tokens


def get_episode_cache_key(feed_name, selected_episode):
    raw_key = f"{feed_name}|{selected_episode.get('title', '')}|{selected_episode.get('url', '')}"
    return hashlib.md5(raw_key.encode("utf-8")).hexdigest()


def get_cache_file_path(cache_key):
    os.makedirs(TRANSCRIPT_CACHE_DIR, exist_ok=True)
    return os.path.join(TRANSCRIPT_CACHE_DIR, f"{cache_key}.json")


def get_audio_response_cache_path(answer_text):
    """Content-addressed cache path for synthesized answer audio, keyed on the exact answer text."""
    os.makedirs(AUDIO_RESPONSE_CACHE_DIR, exist_ok=True)
    cache_key = hashlib.md5(answer_text.encode("utf-8")).hexdigest()
    return os.path.join(AUDIO_RESPONSE_CACHE_DIR, f"{cache_key}.wav")


def load_cached_transcription(cache_key):
    try:
        cache_path = get_cache_file_path(cache_key)
        if not os.path.isfile(cache_path):
            return None
        with open(cache_path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        if not payload.get("transcription_parts"):
            return None
        return payload
    except Exception as e:
        logging.warning(f" Failed to load cache for key {cache_key}: {str(e)}")
        return None


def is_cache_payload_valid(payload, expected_feed_name, expected_episode_title, expected_episode_url):
    if not payload:
        return False

    source = payload.get("source") or {}
    transcription_parts = payload.get("transcription_parts") or []
    if not transcription_parts:
        return False

    source_feed = source.get("feed_name")
    source_title = source.get("episode_title")
    source_url = source.get("episode_url")

    # Backward-compatibility with older cache files that lacked episode_url
    if source_feed == expected_feed_name and source_title == expected_episode_title:
        if source_url is None:
            return True
        return source_url == expected_episode_url

    return False


def save_cached_transcription(cache_key, transcription_parts, source_meta):
    try:
        cache_path = get_cache_file_path(cache_key)
        payload = {
            "transcription_parts": transcription_parts,
            "source": source_meta
        }
        with open(cache_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f" Failed to save cache for key {cache_key}: {str(e)}")


OLLAMA_HOST = "http://localhost:11434"


class OllamaLLM:
    """Ollama wrapper using ollama.chat() with streaming — mirrors the Perplexity reference flow."""

    def __init__(self, model=DEFAULT_OLLAMA_MODEL):
        if ollama is None:
            raise RuntimeError("Ollama Python package is not available. Install dependency 'ollama'.")
        self.model = model
        self.client = ollama.Client(host=OLLAMA_HOST)

    def list_models(self):
        try:
            response = self.client.list()
            if hasattr(response, "models"):
                models = response.models
            elif isinstance(response, dict) and "models" in response:
                models = response["models"]
            else:
                models = response if isinstance(response, list) else []

            names = []
            for model in models:
                if hasattr(model, "model"):
                    names.append(model.model)
                elif hasattr(model, "name"):
                    names.append(model.name)
                elif isinstance(model, dict):
                    names.append(model.get("model", model.get("name", str(model))))
                else:
                    names.append(str(model))
            return names
        except Exception:
            return []

    def chat_stream(self, prompt, max_tokens=900, temperature=0.2):
        """
        Stream a response using ollama.chat() — exactly as used in the Perplexity reference.
        Yields text chunks as they arrive so Streamlit can display them in real-time.
        """
        # Reasoning models (Qwen 3.5+, gemma4, ...) need extra headroom for hidden chain-of-thought
        max_tokens = resolve_effective_max_tokens(self.model, max_tokens)
        # Scale context window to comfortably fit both prompt and the desired output
        num_ctx = max(8192, len(prompt) // 3 + max_tokens + 512)
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={
                "num_ctx": num_ctx,
                "num_predict": max_tokens,   # no cap — honour the slider fully
                "temperature": temperature,
                "top_p": 0.9,
            },
        )
        for chunk in response:
            delta = chunk.get("message", {}).get("content", "") if isinstance(chunk, dict) else ""
            if not delta and hasattr(chunk, "message"):
                delta = chunk.message.content or ""
            if delta:
                yield delta

    def generate(self, prompt, max_tokens=900, temperature=0.2):
        """Non-streaming generate — collects streamed output for non-display contexts."""
        return "".join(self.chat_stream(prompt, max_tokens=max_tokens, temperature=temperature))


class OVMSLLM:
    """OpenVINO Model Server wrapper using the openai SDK against the /v3 OpenAI-compatible API.

    Matches the ovms_agentic_setup.ps1 launcher default (REST API at http://localhost:8000/v3)
    and the Python usage verified in README_OVMS_AGENTIC.md / README_OVMS.md.
    """

    def __init__(self, model=DEFAULT_OVMS_MODEL, base_url=DEFAULT_OVMS_ENDPOINT):
        if OpenAI is None:
            raise RuntimeError("openai package is not available. Install dependency 'openai'.")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = OpenAI(base_url=self.base_url, api_key="unused")

    def list_models(self):
        try:
            response = self.client.models.list()
            return [m.id for m in getattr(response, "data", [])]
        except Exception:
            return []

    def chat_stream(self, prompt, max_tokens=900, temperature=0.2):
        """Stream a response via the openai SDK's SSE client — same method verified in the OVMS README."""
        # Reasoning models (Qwen 3.5+, gpt-oss, gemma4, ...) need extra headroom for hidden chain-of-thought
        max_tokens = resolve_effective_max_tokens(self.model, max_tokens)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stream=True,
        )
        content_emitted = False
        reasoning_emitted = False
        finish_reason = None
        for chunk in stream:
            choices = chunk.choices
            if not choices:
                continue
            delta = choices[0].delta
            finish_reason = choices[0].finish_reason or finish_reason
            # Reasoning-parser models (e.g. Qwen3 "thinking" mode) stream chain-of-thought
            # separately via reasoning_content; the visible answer only lands in content.
            if getattr(delta, "reasoning_content", None):
                reasoning_emitted = True
            if delta.content:
                content_emitted = True
                yield delta.content
        if not content_emitted and reasoning_emitted:
            hint = (
                f" even after auto-raising the token budget to {max_tokens}"
                if finish_reason == "length" else ""
            )
            yield (
                f"⚠️ The model spent its entire response budget on internal reasoning{hint} "
                "and never produced a visible answer. Try raising the 'Response Length (tokens)' "
                "slider further, or simplify the question."
            )

    def generate(self, prompt, max_tokens=900, temperature=0.2):
        """Non-streaming generate — collects streamed output for non-display contexts."""
        return "".join(self.chat_stream(prompt, max_tokens=max_tokens, temperature=temperature))

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="Podcast RAG | Intel® Core™ Ultra Processors",
    page_icon="🎙️",
    layout="wide"
)

# Title and Overview
st.title("🎙️ Podcast RAG")
st.markdown(
    "<p style='font-size:1.1rem; color:#0068b5; margin-top:-0.6rem; margin-bottom:0.8rem;'>"
    "⚡ Accelerated by <strong>Intel® Core™ Ultra Processors</strong> with "
    "<strong>PyTorch XPU</strong> &nbsp;|&nbsp; Local LLM inference via <strong>OpenVINO Model Server</strong> or <strong>Ollama</strong>"
    "</p>",
    unsafe_allow_html=True,
)
st.header("Overview")
st.markdown("""
This application demonstrates a Retrieval-Augmented Generation (RAG) system using a **Podcast episode** as the knowledge base, with text query capability.

Initially an RSS feed link will be the input to allow the user to select and download a specific podcast episode. The selected audio undergoes preprocessing steps such as resampling and chunking to prepare it for transcription. 
Each chunk is transcribed to text using the [**Whisper base model**](https://huggingface.co/openai/whisper-base) via an Automatic Speech Recognition (ASR) pipeline optimized to run on **Intel® Core™ Ultra Processors** with [**PyTorch XPU backend**](https://pytorch.org/docs/stable/notes/get_start_xpu.html) for hardware acceleration. These transcriptions are then embedded using [**Teapot LLM**](https://huggingface.co/teapotai/teapotllm), creating a knowledge base.
User queries are handled by the Teapot RAG system so that it retrieves a relevant text response.
""")

st.header("Workflow")
st.markdown("""
- User provides an *RSS feed URL* to list available podcast episodes.
- The selected audio podcast episode is downloaded, resampled and then split into chunks.
- Each chunk is transcribed to text using the [*Whisper base model*](https://huggingface.co/openai/whisper-base) (ASR).
- Transcribed text chunks are embedded using the [*Teapot LLM*](https://huggingface.co/teapotai/teapotllm) to create a searchable knowledge base.
- A text query from the user is processed by the Teapot LLM RAG system, to get a relevant text response.
""")

st.divider()

tab_demo, tab_architecture = st.tabs(["🚀 Live Demo", "🏗️ Architecture"])

with tab_architecture:
    st.header("🏗️ Architecture")
    st.markdown(
        "End-to-end view of the technologies used in this demo — from podcast ingestion, through "
        "on-device speech-to-text and retrieval, to local LLM inference."
    )

    _architecture_diagram = """
digraph PodcastRAG {
    rankdir=TB
    graph [fontname="Helvetica" bgcolor="#fafafa" pad="0.4" nodesep="0.5" ranksep="0.6"]
    node  [fontname="Helvetica" fontsize=11 style=filled penwidth=1.5 shape=box]
    edge  [fontname="Helvetica" fontsize=10 penwidth=1.4]

    subgraph cluster_input {
        label="Input"
        style=filled fillcolor="#e8f4fb" color="#0068b5" penwidth=2
        A [label="RSS Feed URL" shape=oval fillcolor="#b3d9ff" color="#0068b5"]
        B [label="Episode Selection\nfeedparser" fillcolor="#cce5ff" color="#0068b5"]
        C [label="Audio Download\n.mp3 via requests" fillcolor="#cce5ff" color="#0068b5"]
    }

    subgraph cluster_prep {
        label="Audio Preprocessing"
        style=filled fillcolor="#fff3e0" color="#e65100" penwidth=2
        D [label="Resample to 16kHz Mono\ntorchaudio" fillcolor="#ffe0b2" color="#e65100"]
        E [label="Chunking" fillcolor="#ffe0b2" color="#e65100"]
    }

    subgraph cluster_asr {
        label="Speech-to-Text"
        style=filled fillcolor="#fce4ec" color="#c62828" penwidth=2
        F [label="Whisper base model\nPyTorch XPU backend\nIntel Core Ultra iGPU" fillcolor="#0068b5" color="#00377c" fontcolor="white"]
        G [label="Transcript Cache\nJSON" shape=cylinder fillcolor="#f2f2f2" color="#888888"]
    }

    subgraph cluster_rag {
        label="Retrieval-Augmented Generation"
        style=filled fillcolor="#e8f5e9" color="#2e7d32" penwidth=2
        H  [label="Teapot LLM\nEmbedding Index" fillcolor="#a5d6a7" color="#2e7d32"]
        Qn [label="User Question" shape=oval fillcolor="#c8e6c9" color="#2e7d32"]
        I  [label="Retriever\nTop-K Relevant Chunks" fillcolor="#a5d6a7" color="#2e7d32"]
        J  [label="RAG Prompt\nQuestion plus Context" fillcolor="#a5d6a7" color="#2e7d32"]
    }

    subgraph cluster_llm {
        label="LLM Inference Backend"
        style=filled fillcolor="#f3e5f5" color="#6a1b9a" penwidth=2
        K [label="Backend Selector" shape=diamond fillcolor="#e1bee7" color="#6a1b9a"]
        L [label="OpenVINO Model Server\nOpenAI-compatible /v3 API\nlocalhost:8000" fillcolor="#0068b5" color="#00377c" fontcolor="white"]
        M [label="Ollama\nlocalhost:11434" fillcolor="#e1bee7" color="#6a1b9a"]
        N [label="Streamed Answer" shape=oval fillcolor="#e1bee7" color="#6a1b9a"]
    }

    O [label="Streamlit UI\nst.write_stream()" shape=box fillcolor="#f2f2f2" color="#888888"]

    A -> B -> C -> D -> E -> F -> G
    G -> H
    Qn -> I
    H -> I -> J
    J -> K
    K -> L [label="Default"]
    K -> M [label="Alternative"]
    L -> N
    M -> N
    N -> O
}
"""

    st.graphviz_chart(_architecture_diagram)

    st.subheader("Core Technologies")
    st.markdown("""
| Layer | Technology | Purpose |
|---|---|---|
| Ingestion | `feedparser` + `requests` | Parse podcast RSS feeds, download episode audio |
| Audio Preprocessing | `torchaudio` | Resample to 16kHz mono, chunk long audio |
| Speech-to-Text | [Whisper base](https://huggingface.co/openai/whisper-base) on **PyTorch XPU** | Transcribe audio chunks on Intel® Core™ Ultra iGPU |
| Embedding / Retrieval | [Teapot LLM](https://huggingface.co/teapotai/teapotllm) | Lightweight local embedding + retrieval index over transcript chunks |
| LLM Inference | **OpenVINO Model Server** (default) or **Ollama** | OpenAI-compatible chat completion for RAG answer synthesis, served locally |
| UI | **Streamlit** | Interactive web app with streamed responses |
""")

# ========================================
# FUNCTION DEFINITIONS (from notebook)
# ========================================

def select_podcast_episode(PODCAST_URL):
    """
    Fetches and displays a dropdown widget to select an episode.

    Args:
        PODCAST_URL: Podcast RSS feed URL

    Returns:
        episodes: List of episodes

    Raises:
        Exception : Raises an exception if there is any error while selecting the episode.
    """
    try:
        logging.info(f" Found podcast URL.")

        # Pre-fetch via requests so we can handle SSL certificate errors gracefully.
        # feedparser's internal urllib does not recover from SSL failures on machines
        # that lack a trusted CA bundle (e.g. corporate proxies, fresh Windows installs).
        try:
            _resp = requests.get(PODCAST_URL, timeout=20)
            _resp.raise_for_status()
            feed = feedparser.parse(_resp.content)
        except requests.exceptions.SSLError as _ssl_err:
            logging.warning(
                f" SSL certificate verification failed ({_ssl_err}). "
                "Retrying without certificate verification — treat the feed as trusted."
            )
            _resp = requests.get(PODCAST_URL, timeout=20, verify=False)
            _resp.raise_for_status()
            feed = feedparser.parse(_resp.content)

        if getattr(feed, "bozo", False):
            logging.warning(f" Feed parser warning: {feed.bozo_exception}")
        
        episodes = []
        for entry in getattr(feed, "entries", []):
            audio_url = None
            if hasattr(entry, "enclosures") and entry.enclosures:
                audio_url = entry.enclosures[0].href
            if audio_url:
                episode_page_url = entry.link if hasattr(entry, "link") else PODCAST_URL
                # Parse publish date — feedparser exposes published_parsed as time.struct_time (UTC)
                pub_dt = None
                for date_field in ("published_parsed", "updated_parsed"):
                    t = getattr(entry, date_field, None)
                    if t:
                        try:
                            pub_dt = datetime(*t[:6], tzinfo=timezone.utc)
                        except Exception:
                            pass
                        break
                episodes.append({
                    "title": entry.title,
                    "url": audio_url,
                    "page_url": episode_page_url,
                    "feed_url": PODCAST_URL,
                    "published": pub_dt.isoformat() if pub_dt else None,
                })
        
        if not episodes:
            raise ValueError(
                "No playable podcast episodes found. Use a direct RSS feed URL (XML), not a webpage URL. "
                "Examples: https://feeds.feedburner.com/tedtalks_audio or https://feeds.npr.org/510289/podcast.xml"
            )
        
        logging.info(f" Found {len(episodes)} episodes")
        return episodes
    except Exception as e:
        logging.exception(f" Error while selecting the podcast: {str(e)}")
        raise


def download_selected_audio(selected_episode, selected_index):
    """
    Download the audio from the selected podcast episode.

    Args:
        selected_episode: Selected episode dictionary
        selected_index: Index of selected episode
        
    Returns:
        audio_path (str): The file path to the saved audio file.

    Raises:
        Exception: Raises an exception if there is any error during downloading the audio file.
    """
    try:
        output_dir = "downloads"
        selected_url = selected_episode["url"]
        logging.info(f" Selected episode: {selected_episode['title']}")
        logging.info(f" Audio URL: {selected_url}")
        os.makedirs(output_dir, exist_ok=True)

        parsed = urlparse(selected_url)
        original_name = os.path.basename(parsed.path)
        default_name = f"episode_{selected_index}.mp3"
        filename = original_name if original_name else default_name
        if not os.path.splitext(filename)[1]:
            filename = f"{filename}.mp3"

        audio_path = os.path.join(output_dir, filename)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        referer = selected_episode.get("page_url") or selected_episode.get("feed_url") or origin
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "audio/*,*/*;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": referer,
            "Origin": origin
        }

        def _do_download(session, url, hdrs, verify_ssl=True):
            """Perform the streaming download, handling 403 header retry internally."""
            resp = session.get(url, headers=hdrs, stream=True, timeout=60,
                               allow_redirects=True, verify=verify_ssl)
            if resp.status_code == 403:
                logging.info(" 403 received. Retrying with minimal headers...")
                minimal = {"User-Agent": hdrs["User-Agent"], "Accept": hdrs["Accept"]}
                resp = session.get(url, headers=minimal, stream=True, timeout=60,
                                   allow_redirects=True, verify=verify_ssl)
            return resp

        with requests.Session() as session:
            try:
                response = _do_download(session, selected_url, headers, verify_ssl=True)
            except requests.exceptions.SSLError as _ssl_err:
                logging.warning(
                    f" SSL certificate verification failed ({_ssl_err}). "
                    "Retrying without certificate verification — treat the host as trusted."
                )
                response = _do_download(session, selected_url, headers, verify_ssl=False)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "audio" not in content_type.lower() and "octet-stream" not in content_type.lower():
                logging.warning(f" Unexpected content type for audio download: {content_type}")

            with open(audio_path, "wb") as file_handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file_handle.write(chunk)

        if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
            raise RuntimeError(f"Downloaded file is missing or empty: {audio_path}")

        logging.info(f" Audio saved to: {audio_path}")
        logging.info(f" Audio file size: {os.path.getsize(audio_path)} bytes")
        return audio_path
    except Exception as e:
        logging.exception(f" Error while downloading the podcast: {str(e)}")
        raise


@st.cache_resource
def initialize_audio_models():
    """
    Initialize Automatic Speech Recognition (ASR) model.

    Returns:
        model : The loaded Whisper model
        processor: processor for pre-processing of audio input

    Raises:
        Exception: Raises an exception if there is any error during model or processor initialization.
    """
    try:
        model_id = "openai/whisper-base"
        device = "xpu" if torch.xpu.is_available() else "cpu"
        logging.info(f" ASR device selected: {device} (torch.xpu.is_available()={torch.xpu.is_available()})")
        processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path=model_id)
        model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_id)
        model = model.to(device)
        logging.info(" Model loaded!")
        return model, processor
    except Exception as e:
        logging.exception(f" Error while initializing models: {str(e)}")
        raise


def process_podcast_audio(audio_path, model, processor):
    """
    Process an audio file for transcription using a ASR model.

    Args:
        audio_path (str): The file path to the saved audio file.    
        model : The loaded Whisper model.
        processor: processor for pre-processing of audio input

    Returns:
        transcription_parts: List of transcribed text chunks from the audio file.

    Raises:
        Exception: Raises an exception if there is any error while processing the audio file.
    """
    try:
        if not audio_path or not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as load_error:
            if "torchcodec" in str(load_error).lower():
                logging.warning(" torchaudio.load requires TorchCodec; falling back to soundfile loader.")
                audio_data, sample_rate = sf.read(audio_path, always_2d=False)
                waveform = torch.tensor(audio_data, dtype=torch.float32)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.transpose(0, 1)
            else:
                raise

        logging.info(f" Original sample rate: {sample_rate} Hz")
        logging.info(f" Audio shape: {waveform.shape}")
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            logging.info(" Converted stereo to mono")
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
            logging.info(" Resampled to 16kHz")
            logging.info(f" Audio shape: {waveform.shape}")
        
        audio = waveform.squeeze().numpy()
        if len(audio) <= 0:
            raise ValueError("Loaded audio is empty.")
        logging.info(f" Audio duration: {len(audio)/sample_rate:.1f} seconds")
        chunk_length = 22 * 16000   # 22 seconds
        overlap_length = 1 * 16000  # 1 second overlap
        transcription_parts = []
        # Calculate total chunks accounting for overlap
        step_size = chunk_length - overlap_length
        total_chunks = max(1, (len(audio) + step_size - 1) // step_size)
        logging.info(f" Processing approximately {total_chunks} chunks of audio..")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        live_transcription_expander = st.expander("📝 Live Transcription (updates during processing)", expanded=False)
        live_transcription_placeholder = live_transcription_expander.empty()
        
        for i in range(0, len(audio), chunk_length - overlap_length):
            chunk = audio[i:i + chunk_length]
            if len(chunk) < 1600:   # Less than 0.1 seconds, skip that chunk
                continue
            
            current_chunk = len(transcription_parts) + 1
            logging.info(f" Processing chunk {current_chunk}/{total_chunks}...")
            status_text.text(f"Processing chunk {current_chunk}/{total_chunks}...")
            progress_bar.progress(min(1.0, current_chunk / total_chunks))
            
            input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to('xpu' if torch.xpu.is_available() else 'cpu')
            chunk_start_time = time.perf_counter()
            gen_kwargs = {"max_new_tokens": 200, "no_repeat_ngram_size": 3}
            try:
                with torch.no_grad():
                    predicted_ids = model.generate(input_features, **gen_kwargs)
            except RuntimeError as gen_error:
                # XPU/oneDNN runtime faults (e.g. GPU contention with OVMS) surface as RuntimeError mid-generate;
                # retry the same chunk on CPU rather than aborting the whole transcription run.
                original_device = next(model.parameters()).device
                logging.warning(f" generate() failed on chunk {current_chunk} ({original_device}): {gen_error}. Retrying on CPU.")
                model.to("cpu")
                with torch.no_grad():
                    predicted_ids = model.generate(input_features.to("cpu"), **gen_kwargs)
                model.to(original_device)
            chunk_elapsed = time.perf_counter() - chunk_start_time
            chunk_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]        
            transcription_parts.append(chunk_transcription)
            logging.info(f" Chunk {len(transcription_parts)} ({chunk_elapsed:.1f}s): {chunk_transcription}..")
            live_transcription_placeholder.text_area(
                "Transcription",
                "\n\n".join(transcription_parts),
                height=300,
                key=f"live_transcription_{current_chunk}"
            )
        
        progress_bar.empty()
        status_text.empty()
        
        transcription = " ".join(transcription_parts)
        logging.info(f"\n\n Total length: {len(transcription)} characters")
        logging.info(f" Audio path: {audio_path}")
        logging.info(f" Word count: {len(transcription.split())}")
        logging.info(transcription)
        return transcription_parts
    except Exception as e:
        logging.exception(f" Error while processing the input audio query : {str(e)}")
        raise


def transcribe_voice_query(audio_bytes, model, processor, max_seconds=29):
    """
    Transcribe a short mic-recorded question in a single pass (no chunking).
    Mirrors the PDF-to-Audio-RAG notebook's process_input_query() flow.

    Args:
        audio_bytes (bytes): Raw audio bytes captured via st.audio_input().
        model : The loaded Whisper model.
        processor: processor for pre-processing of audio input.
        max_seconds (int): Safety cap on recording length fed to the model.

    Returns:
        question (str): Transcribed question text.

    Raises:
        Exception: Raises an exception if there is any error while transcribing the recording.
    """
    try:
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=False)
        waveform = torch.tensor(audio_data, dtype=torch.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1)   # stereo -> mono

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
            sample_rate = 16000

        audio = waveform.numpy()
        max_samples = max_seconds * 16000
        if len(audio) > max_samples:
            logging.warning(f" Voice query longer than {max_seconds}s; truncating.")
            audio = audio[:max_samples]
        if len(audio) <= 0:
            raise ValueError("Recorded audio is empty.")

        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to('xpu' if torch.xpu.is_available() else 'cpu')
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        question = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        logging.info(f" Voice query transcribed: {question}")
        return question
    except Exception as e:
        logging.exception(f" Error while transcribing voice query: {str(e)}")
        raise


@st.cache_resource
def initialize_tts_models():
    """
    Initialize the SpeechT5 Text-to-Speech pipeline and a CMU ARCTIC speaker embedding,
    mirroring the PDF-to-Audio-RAG notebook's initialize_audio_models() TTS setup.

    Returns:
        synthesiser : TTS pipeline for text-to-speech.
        speaker_embedding : Speaker embedding tensor used for voice synthesis.

    Raises:
        Exception: Raises an exception if there is any error while loading the TTS models.
    """
    try:
        if load_dataset is None:
            raise RuntimeError("The 'datasets' package is not available. Install dependency 'datasets'.")
        device = "xpu" if torch.xpu.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.xpu.is_available() else torch.float32
        synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device=device, torch_dtype=torch_dtype)
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device, torch_dtype)
        logging.info(" TTS models loaded!")
        return synthesiser, speaker_embedding
    except Exception as e:
        logging.exception(f" Error while initializing TTS models: {str(e)}")
        raise


def synthesize_speech_response(text, synthesiser, speaker_embedding, max_chars_per_segment=350):
    """
    Convert answer text to a speech WAV byte string using SpeechT5, splitting long answers into
    sentence-grouped segments (synthesized separately and concatenated) to keep each TTS call short.

    Args:
        text (str): Answer text to speak.
        synthesiser : SpeechT5 TTS pipeline.
        speaker_embedding : Speaker embedding tensor used for voice synthesis.
        max_chars_per_segment (int): Approximate character budget per TTS segment.

    Returns:
        wav_bytes (bytes): WAV-encoded audio of the spoken answer.

    Raises:
        Exception: Raises an exception if there is any error while synthesizing speech.
    """
    try:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        segments = []
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) + 1 > max_chars_per_segment:
                segments.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            segments.append(current)
        if not segments:
            segments = [text]

        audio_chunks = []
        sampling_rate = None
        with torch.no_grad():
            for segment in segments:
                speech = synthesiser(segment, forward_params={"speaker_embeddings": speaker_embedding})
                audio_chunks.append(speech["audio"])
                sampling_rate = speech["sampling_rate"]

        full_audio = audio_chunks[0] if len(audio_chunks) == 1 else np.concatenate(audio_chunks)
        buffer = io.BytesIO()
        sf.write(buffer, full_audio, samplerate=sampling_rate, format="WAV")
        return buffer.getvalue()
    except Exception as e:
        logging.exception(f" Error while synthesizing speech response: {str(e)}")
        raise



def generate_embeddings(transcription_parts):
    """
    Generate embeddings for the list of transcribed text chunks.
    
    Args:
        transcription_parts (list): List of transcribed text chunks from the audio file.
    
    Returns:
        teapot_ai : Model with text embeddings.

    Raises:
        Exception: Raises an exception if there is any error while generating embeddings for the audio data chunks.
    """
    try:
        if transcription_parts:
            logging.info(" Found transcriptions.")
            teapot_ai = TeapotAI(documents=transcription_parts)
            logging.info(" Generated embeddings.")
            return teapot_ai
        else:
            logging.info(" Did not find any transcriptions.")
    except Exception as e:
        logging.exception(f" Error while generating embeddings using TeapotAI : {str(e)}")
        raise




def _tokenize(text):
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def _is_host_question(question):
    lowered = question.lower()
    host_terms = ["host", "who hosts", "who is hosting", "presenter", "moderator"]
    return any(term in lowered for term in host_terms)


def _extract_host_from_all_chunks(documents):
    """
    Scan the full transcript to find explicit host-introduction lines.
    Returns a concise host answer if found, else None.
    """
    if not documents:
        return None

    def normalize_name_candidate(raw_candidate):
        if not raw_candidate:
            return None

        head = re.split(
            r"[\.,;:!?\n]|\s+(?:and|with|who|that|because|for|on|in|at|when|where)\s+",
            raw_candidate,
            maxsplit=1,
            flags=re.IGNORECASE
        )[0]

        words = re.findall(r"[a-zA-Z][a-zA-Z'-]*", head.lower())
        words = [word for word in words if len(word) > 1]

        if len(words) < 2 or len(words) > 3:
            return None

        if any(word in STOPWORDS or word in HOST_NAME_BLOCKLIST for word in words):
            return None

        return " ".join(word.capitalize() for word in words)

    weighted_patterns = [
        (r"i['’]?m\s+your\s+host[,:\-]?\s+([a-z][a-z'\-\s]{1,50})", 8),
        (r"your\s+host[,:\-]?\s+([a-z][a-z'\-\s]{1,50})", 7),
        (r"hosted\s+by\s+([a-z][a-z'\-\s]{1,50})", 6),
        (r"my\s+name\s+is\s+([a-z][a-z'\-\s]{1,50})", 3),
        (r"this\s+is\s+([a-z][a-z'\-\s]{1,50})", 2)
    ]

    host_cue_terms = ["host", "intechnology", "welcome", "join us", "thanks for listening"]
    non_host_cue_terms = ["guest", "with", "interview", "sunday pick", "story", "sponsored", "ad"]

    candidate_scores = Counter()

    for chunk_idx, chunk in enumerate(documents):
        lowered_chunk = chunk.lower()
        intro_bonus = 4 if chunk_idx < 8 else 0
        host_cue_bonus = 2 if any(term in lowered_chunk for term in host_cue_terms) else 0
        non_host_penalty = -2 if any(term in lowered_chunk for term in non_host_cue_terms) else 0

        for pattern, base_weight in weighted_patterns:
            matches = re.findall(pattern, chunk, flags=re.IGNORECASE)
            for match in matches:
                cleaned = normalize_name_candidate(str(match).strip())
                if cleaned:
                    candidate_scores[cleaned] += base_weight + intro_bonus + host_cue_bonus + non_host_penalty

    if not candidate_scores:
        return None

    ranked = candidate_scores.most_common(2)
    host_name, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -999

    if top_score < 8 or (top_score - second_score) <= 1:
        logging.info(f" Host extraction low confidence. Candidates: {ranked}")
        return None

    return f"The host mentioned in this transcript appears to be {host_name}."


def build_retrieval_index(transcription_parts):
    """
    Build a lightweight TF-IDF retrieval index over transcription chunks.
    """
    tokenized_docs = [_tokenize(chunk) for chunk in transcription_parts]
    total_docs = len(tokenized_docs)

    doc_freq = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    idf = {
        token: math.log((1 + total_docs) / (1 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }

    vectors = []
    norms = []
    for tokens in tokenized_docs:
        term_counts = Counter(tokens)
        vec = {token: count * idf.get(token, 1.0) for token, count in term_counts.items()}
        norm = math.sqrt(sum(value * value for value in vec.values())) or 1.0
        vectors.append(vec)
        norms.append(norm)

    return {
        "documents": transcription_parts,
        "idf": idf,
        "vectors": vectors,
        "norms": norms
    }


def retrieve_relevant_chunks(question, retrieval_index, top_k=5):
    """
    Retrieve top-k relevant transcription chunks for the input question.
    """
    if not retrieval_index:
        return []

    summary_terms = [
        "summary", "summarize", "detailed summary", "everything", "overview",
        "main points", "key points", "what is this podcast about", "what's this podcast about",
        "main topics", "topics covered", "key takeaways", "takeaways", "central argument",
        "central message", "how does this episode", "how does it start", "how does it end",
        "what happens in", "what was discussed", "what did they talk", "what is covered",
        "what are the insights", "most interesting", "interesting insights",
        "predictions", "future trends", "advice", "recommendations given",
        "surprising", "counterintuitive",
    ]
    question_lower = question.lower()
    is_summary_query = any(term in question_lower for term in summary_terms)

    if is_summary_query:
        docs = retrieval_index["documents"]
        if not docs:
            return []
        # For episode-level summaries, retrieve diversified chunks across the timeline
        target_chunks = min(max(top_k, 12), len(docs))
        step = max(1, len(docs) // target_chunks)
        selected = [docs[idx] for idx in range(0, len(docs), step)][:target_chunks]
        return selected

    if _is_host_question(question):
        docs = retrieval_index["documents"]
        host_markers = ["host", "welcome", "join us", "i'm", "my name is", "intechnology"]
        matched = [doc for doc in docs if any(marker in doc.lower() for marker in host_markers)]
        if matched:
            return matched[:min(max(top_k, 8), len(matched))]
        return docs[:min(max(top_k, 8), len(docs))]

    question_tokens = _tokenize(question)
    if not question_tokens:
        return retrieval_index["documents"][:top_k]

    question_counts = Counter(question_tokens)
    query_vec = {
        token: count * retrieval_index["idf"].get(token, 1.0)
        for token, count in question_counts.items()
    }
    query_norm = math.sqrt(sum(value * value for value in query_vec.values())) or 1.0

    scored = []
    for idx, doc_vec in enumerate(retrieval_index["vectors"]):
        dot_product = sum(query_vec.get(token, 0.0) * value for token, value in doc_vec.items())
        score = dot_product / (query_norm * retrieval_index["norms"][idx])
        scored.append((score, idx))

    scored.sort(reverse=True)
    top_indices = [idx for score, idx in scored[:top_k] if score > 0]
    if not top_indices:
        top_indices = [idx for _, idx in scored[:top_k]]

    return [retrieval_index["documents"][idx] for idx in top_indices]


def _is_longform_request(question):
    """Detect questions that need a long-form response (blog post, email, detailed summary)."""
    q = question.lower()
    return any(term in q for term in [
        "blog post", "write a blog", "listicle", "thought leadership",
        "email", "newsletter", "email template",
        "detailed summary", "full summary", "comprehensive summary",
        "summarize", "summary",
    ])


def _build_rag_prompt(question, retrieved_chunks):
    context = "\n\n---\n\n".join(retrieved_chunks)

    if _is_longform_request(question):
        return f"""You are a helpful podcast assistant producing a long-form written piece.

Use ONLY the transcript excerpts below as your source material.

Rules:
- Write as much as the task demands — do NOT stop early or say the answer is complete before the piece is finished.
- Do NOT use labels like \"Chunk 1\" or \"[Chunk 2]\" anywhere.
- Synthesize and paraphrase; do not copy long verbatim passages.
- Do not say you lack context — the relevant transcript is already provided.
- Never start with \"I'm ready\" or \"What's the question\" — begin writing immediately.

TRANSCRIPT EXCERPTS:
{context}

TASK: {question}

OUTPUT:"""

    return f"""You are a helpful podcast assistant. Answer ONLY from the transcript excerpts provided below.

Rules you must follow:
- Write your answer as flowing prose in 1-3 paragraphs. Do NOT use labels like \"Chunk 1\" or \"[Chunk 2]\" anywhere in your response.
- Synthesize and paraphrase; do not copy long verbatim passages.
- Do not say you lack context — the relevant transcript is already provided below.
- If the question is broad (e.g. main topics, summary, insights), cover the key themes across the excerpts.
- Never start your answer with \"I'm ready\" or \"What's the question\" — always answer directly.

TRANSCRIPT EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


def _build_synthesis_retry_prompt(question, retrieved_chunks):
    context = "\n\n---\n\n".join(retrieved_chunks)
    return f"""Answer the question using ONLY the transcript excerpts below.

Rules:
- Return exactly one concise paragraph (4-7 sentences).
- Synthesize in your own words; do NOT copy transcript lines verbatim.
- Do NOT use chunk labels or say "I'm ready" — answer the question directly.
- If uncertain, give the best grounded answer from the available context.

QUESTION: {question}

TRANSCRIPT EXCERPTS:
{context}

ANSWER:"""


def _generate_with_llm(llm_client, prompt, max_tokens=900):
    """Generate answer text via the active LLM backend (non-streaming, for internal retries)."""
    return llm_client.generate(prompt, max_tokens=max_tokens, temperature=0.2)


def _stream_with_llm(llm_client, prompt, max_tokens=900):
    """Return a streaming generator from the active LLM backend — for use with st.write_stream()."""
    return llm_client.chat_stream(prompt, max_tokens=max_tokens, temperature=0.2)


def _normalize_answer(raw_answer):
    """
    Normalize Teapot output into a plain string.
    """
    if isinstance(raw_answer, str):
        return raw_answer.strip()

    if isinstance(raw_answer, dict):
        for key in ["answer", "content", "text", "response", "message"]:
            value = raw_answer.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return str(raw_answer).strip()

    if isinstance(raw_answer, (list, tuple)):
        parts = []
        for item in raw_answer:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                content = item.get("content") or item.get("text") or item.get("answer")
                if isinstance(content, str):
                    parts.append(content)
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join([part for part in parts if part]).strip()

    return str(raw_answer).strip()


def _is_unusable_answer(answer_text):
    if not answer_text:
        return True

    compact = answer_text.strip().lower()
    if len(compact) < 40:
        return True

    bad_markers = [
        "don't have enough information",
        "not enough information",
        "i don't have enough information",
        "i don't know",
        "i cannot determine",
        "i'm not qualified",
        "i apologize",
        "i don't have access"
    ]
    if any(marker in compact for marker in bad_markers):
        return True

    if compact in ["[chunk 1]", "chunk 1", "[chunk 1].", "[chunk 1]\n"]:
        return True

    return False


def _is_context_dump(answer_text, retrieved_chunks):
    if not answer_text:
        return False

    cleaned_answer = re.sub(r"\s+", " ", answer_text.strip().lower())
    if len(cleaned_answer) < 180:
        return False

    context_text = " ".join(retrieved_chunks).lower()
    context_tokens = set(_tokenize(context_text))
    answer_tokens = _tokenize(cleaned_answer)
    if not answer_tokens:
        return False

    overlap = sum(1 for token in answer_tokens if token in context_tokens)
    overlap_ratio = overlap / max(1, len(answer_tokens))

    copied_prefix = cleaned_answer[:120] in context_text if len(cleaned_answer) >= 120 else False
    return overlap_ratio > 0.88 or copied_prefix


def _extractive_fallback_summary(question, retrieved_chunks):
    """
    Build a deterministic fallback answer from retrieved chunks when model output is unusable.
    """
    combined = " ".join(retrieved_chunks)
    sentences = re.split(r"(?<=[.!?])\s+", combined)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if not sentences:
        return "I retrieved relevant chunks, but could not synthesize a reliable answer from them."

    if _is_host_question(question):
        host_sentences = [
            sentence for sentence in sentences
            if any(marker in sentence.lower() for marker in ["host", "welcome", "i'm", "my name is", "join us"])
        ]
        if host_sentences:
            return " ".join(host_sentences[:4])

    question_tokens = set(_tokenize(question))

    def score(sentence):
        sentence_tokens = set(_tokenize(sentence))
        overlap = len(question_tokens.intersection(sentence_tokens))
        return overlap, len(sentence)

    ranked = sorted(sentences, key=score, reverse=True)
    selected = ranked[:6]

    if not selected:
        selected = sentences[:4]

    return " ".join(selected)


def get_response(question, llm_client, retrieval_index, top_k=5, stream=False, max_tokens=900):
    """
    Get the response from the RAG pipeline using the active LLM backend (OVMS or Ollama).

    Args:
        question (str): Input query
        llm_client : Active LLM backend client (OVMSLLM or OllamaLLM), exposing generate()/chat_stream()
        retrieval_index (dict): Precomputed retrieval index over transcription chunks
        top_k (int): Number of chunks to retrieve
        stream (bool): If True, returns a generator for the final answer instead of a string.
        max_tokens (int): Maximum tokens for the response.

    Returns:
        answer (str | generator): Generated response, or a streaming generator when stream=True.
        retrieved_chunks (list): Context chunks retrieved for this query.

    Raises:
        Exception: Raises an exception if there is any error while getting the response.
    """
    try:
        if question:
            logging.info(f" Query: {question}\n")

            if _is_host_question(question):
                all_documents = retrieval_index.get("documents", [])
                host_answer = _extract_host_from_all_chunks(all_documents)
                if host_answer:
                    logging.info(" Host answer found via full-transcript scan.")
                    host_chunks = retrieve_relevant_chunks(question, retrieval_index, top_k=min(12, len(all_documents)))
                    return host_answer, host_chunks

            retrieved_chunks = retrieve_relevant_chunks(question, retrieval_index, top_k=top_k)
            if not retrieved_chunks:
                raise ValueError("No context chunks were retrieved. Please transcribe and index the podcast first.")
            logging.info(f" Retrieved {len(retrieved_chunks)} chunks for answer generation")

            prompt = _build_rag_prompt(question, retrieved_chunks)

            if stream:
                # Return the streaming generator directly — caller uses st.write_stream()
                return _stream_with_llm(llm_client, prompt, max_tokens=max_tokens), retrieved_chunks

            # Non-streaming path (used for retries / quality checks)
            answer = _normalize_answer(_generate_with_llm(llm_client, prompt, max_tokens=max_tokens))

            if _is_unusable_answer(answer):
                logging.info(" Initial answer was insufficient; retrying with broader retrieved context.")
                broader_chunks = retrieve_relevant_chunks(question, retrieval_index, top_k=min(20, len(retrieval_index["documents"])))
                broader_prompt = _build_rag_prompt(question, broader_chunks)
                answer = _normalize_answer(_generate_with_llm(llm_client, broader_prompt))
                retrieved_chunks = broader_chunks

            if _is_context_dump(answer, retrieved_chunks):
                logging.info(" Answer appears to be context dump; retrying with strict synthesis prompt.")
                synth_prompt = _build_synthesis_retry_prompt(question, retrieved_chunks)
                answer = _normalize_answer(_generate_with_llm(llm_client, synth_prompt))

            if _is_unusable_answer(answer) or _is_context_dump(answer, retrieved_chunks):
                if _is_host_question(question):
                    logging.info(" Host question fallback: extractive host-focused answer.")
                    answer = _extractive_fallback_summary(question, retrieved_chunks)
                else:
                    logging.info(" Model output still poor; returning controlled failure message (no context dump).")
                    answer = (
                        "I retrieved relevant transcript chunks but could not synthesize a reliable answer. "
                        "Please ask a more specific question (for example: main topic, key argument, or named person)."
                    )

            if _is_host_question(question) and "host" not in answer.lower():
                logging.info(" Host question answer lacked host signal; applying host-focused extractive fallback.")
                answer = _extractive_fallback_summary(question, retrieved_chunks)

            logging.info(f" Response: {answer}")
            return answer, retrieved_chunks
        else:
            raise ValueError("Question is empty.")
    except Exception as e:
        logging.exception(f" Error while generating response : {str(e)}")
        raise


# ========================================
# STREAMLIT APPLICATION FLOW
# ========================================

def reset_rag_state(clear_audio=False):
    st.session_state.transcription_parts = None
    st.session_state.teapot_ai = None
    st.session_state.retrieval_index = None
    st.session_state.current_query = None
    st.session_state.transcript_source = None
    st.session_state.current_episode_cache_key = None
    if clear_audio:
        st.session_state.audio_path = None


def set_current_query(new_query):
    st.session_state.current_query = new_query
    st.session_state.needs_generation = True
    st.session_state.last_answer = ""
    st.session_state.last_answer_query = ""
    st.session_state.last_retrieved_count = 0


with tab_demo:
    # Initialize session state
    if 'episodes' not in st.session_state:
        st.session_state.episodes = None
    if 'audio_path' not in st.session_state:
        st.session_state.audio_path = None
    if 'transcription_parts' not in st.session_state:
        st.session_state.transcription_parts = None
    if 'teapot_ai' not in st.session_state:
        st.session_state.teapot_ai = None
    if 'retrieval_index' not in st.session_state:
        st.session_state.retrieval_index = None
    if 'llm_backend' not in st.session_state:
        st.session_state.llm_backend = LLM_BACKEND_OVMS
    if 'active_llm' not in st.session_state:
        st.session_state.active_llm = None
    if 'ollama_llm' not in st.session_state:
        st.session_state.ollama_llm = None
    if 'ollama_model' not in st.session_state:
        st.session_state.ollama_model = DEFAULT_OLLAMA_MODEL
    if 'ovms_llm' not in st.session_state:
        st.session_state.ovms_llm = None
    if 'ovms_model' not in st.session_state:
        st.session_state.ovms_model = DEFAULT_OVMS_MODEL
    if 'ovms_endpoint' not in st.session_state:
        st.session_state.ovms_endpoint = DEFAULT_OVMS_ENDPOINT
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'tts_synthesiser' not in st.session_state:
        st.session_state.tts_synthesiser = None
    if 'tts_speaker_embedding' not in st.session_state:
        st.session_state.tts_speaker_embedding = None
    if 'last_voice_query_hash' not in st.session_state:
        st.session_state.last_voice_query_hash = None
    if 'last_answer_audio' not in st.session_state:
        st.session_state.last_answer_audio = None
    if 'last_answer_audio_query' not in st.session_state:
        st.session_state.last_answer_audio_query = None
    if 'selected_feed_name' not in st.session_state:
        st.session_state.selected_feed_name = None
    if 'selected_episode_title' not in st.session_state:
        st.session_state.selected_episode_title = None
    if 'transcript_source' not in st.session_state:
        st.session_state.transcript_source = None
    if 'current_episode_cache_key' not in st.session_state:
        st.session_state.current_episode_cache_key = None
    if 'last_answer' not in st.session_state:
        st.session_state.last_answer = ""
    if 'last_answer_query' not in st.session_state:
        st.session_state.last_answer_query = ""
    if 'last_retrieved_count' not in st.session_state:
        st.session_state.last_retrieved_count = 0
    if 'needs_generation' not in st.session_state:
        st.session_state.needs_generation = False

    # Inference backend selection — OpenVINO Model Server (OVMS) is the default, Ollama is the alternative
    st.header("🧠 Inference Backend")
    selected_backend = st.selectbox(
        "Choose inference backend:",
        options=LLM_BACKENDS,
        index=LLM_BACKENDS.index(st.session_state.llm_backend)
        if st.session_state.llm_backend in LLM_BACKENDS else 0,
        key="llm_backend_selector"
    )
    st.session_state.llm_backend = selected_backend

    if selected_backend == LLM_BACKEND_OVMS:
        ovms_endpoint = st.text_input(
            "OVMS endpoint (started via ovms_agentic_setup.ps1)",
            value=st.session_state.ovms_endpoint,
            key="ovms_endpoint_input"
        )
        st.session_state.ovms_endpoint = ovms_endpoint
        try:
            temp_ovms = OVMSLLM(st.session_state.ovms_model, base_url=ovms_endpoint)
            available_ovms_models = temp_ovms.list_models()
            model_options = available_ovms_models if available_ovms_models else [st.session_state.ovms_model]
            selected_ovms_model = st.selectbox(
                "Choose OVMS model:",
                options=model_options,
                index=model_options.index(st.session_state.ovms_model)
                if st.session_state.ovms_model in model_options else 0,
                key="ovms_model_selector"
            )
            if (
                st.session_state.ovms_llm is None
                or st.session_state.ovms_model != selected_ovms_model
                or st.session_state.ovms_llm.base_url != ovms_endpoint
            ):
                st.session_state.ovms_llm = OVMSLLM(selected_ovms_model, base_url=ovms_endpoint)
                st.session_state.ovms_model = selected_ovms_model
            st.session_state.active_llm = st.session_state.ovms_llm
            if not available_ovms_models:
                st.warning(
                    f"⚠️ Could not reach OVMS at {ovms_endpoint}. Using default model name "
                    f"'{st.session_state.ovms_model}'. Start it with `ovms_agentic_setup.ps1` "
                    "(defaults to port 8000, path /v3)."
                )
            else:
                st.success(f"✅ OVMS model ready: {selected_ovms_model}")
        except Exception as e:
            st.error(f"❌ OVMS initialization error: {str(e)}")
            st.session_state.active_llm = None
    else:
        if ollama is None:
            st.error("Ollama Python package not found. Install dependency 'ollama' and ensure Ollama server is running.")
            st.session_state.active_llm = None
        else:
            try:
                temp_ollama = OllamaLLM(st.session_state.ollama_model)
                available_models = temp_ollama.list_models()
                if not available_models:
                    st.warning("No Ollama models found. Run `ollama pull llama3.1:8b` and keep Ollama server running.")
                    st.session_state.active_llm = None
                else:
                    selected_ollama_model = st.selectbox(
                        "Choose Ollama model:",
                        options=available_models,
                        index=available_models.index(st.session_state.ollama_model)
                        if st.session_state.ollama_model in available_models else 0,
                        key="ollama_model_selector"
                    )
                    if (
                        st.session_state.ollama_llm is None
                        or st.session_state.ollama_model != selected_ollama_model
                        or not hasattr(st.session_state.ollama_llm, "chat_stream")
                    ):
                        st.session_state.ollama_llm = OllamaLLM(selected_ollama_model)
                        st.session_state.ollama_model = selected_ollama_model
                        st.success(f"✅ Ollama model ready: {selected_ollama_model}")
                    st.session_state.active_llm = st.session_state.ollama_llm
            except Exception as e:
                st.error(f"❌ Ollama initialization error: {str(e)}")
                st.session_state.active_llm = None

    st.divider()

    # Step 1: Select Podcast Feed
    st.header("📡 Step 1: Select Podcast Feed")

    podcast_feed_options = {
        # ── Intel ──────────────────────────────────────────────────────────────────
        "Intel on AI": "https://feeds.libsyn.com/363776/rss",
        # ── AI / Tech (top-chart, verified reachable via HTTPS) ─────────────────────
        "Lex Fridman Podcast": "https://lexfridman.com/feed/podcast/",
        "Hard Fork (NYT Tech)": "https://feeds.simplecast.com/l2i9YnTd",
        "Acquired": "https://acquired.libsyn.com/rss",
    }
    CUSTOM_FEED_LABEL = "Other (enter your own RSS URL)"

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_feed_name = st.selectbox(
            "Choose a podcast feed:",
            options=list(podcast_feed_options.keys()) + [CUSTOM_FEED_LABEL]
        )

        # If user changed feed selection, immediately invalidate stale episode/transcript state
        if st.session_state.selected_feed_name and st.session_state.selected_feed_name != selected_feed_name:
            st.session_state.episodes = None
            st.session_state.selected_episode_title = None
            reset_rag_state(clear_audio=True)

        if selected_feed_name == CUSTOM_FEED_LABEL:
            podcast_url = st.text_input(
                "RSS Feed URL:",
                placeholder="https://example.com/podcast/feed.xml",
                key="custom_feed_rss_url",
            )
        else:
            podcast_url = podcast_feed_options[selected_feed_name]
            st.text_input("RSS Feed URL:", value=podcast_url, disabled=True)

    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Load Episodes", use_container_width=True):
            if not podcast_url:
                st.warning("Enter an RSS feed URL first.")
            else:
                with st.spinner("Loading episodes from feed..."):
                    try:
                        st.session_state.selected_feed_name = selected_feed_name
                        st.session_state.episodes = select_podcast_episode(podcast_url)
                        st.success(f"✅ Found {len(st.session_state.episodes)} episodes!")
                    except Exception as e:
                        st.error(f"❌ Error loading feed: {str(e)}")

    st.divider()

    # Step 2: Select Episode
    if st.session_state.episodes:
        st.header("🎧 Step 2: Select Episode")

        # ── Recency filter ────────────────────────────────────────────────────────
        now_utc = datetime.now(timezone.utc)
        recency_options = {
            "Last 7 days":   timedelta(days=7),
            "Last 30 days":  timedelta(days=30),
            "Last 3 months": timedelta(days=90),
            "Last 6 months": timedelta(days=180),
            "Last year":     timedelta(days=365),
            "All episodes":  None,
        }
        filter_col, info_col = st.columns([2, 3])
        with filter_col:
            selected_recency = st.selectbox(
                "📅 Filter by recency:",
                options=list(recency_options.keys()),
                index=2,  # default: last 3 months
                key="recency_filter"
            )
        cutoff = recency_options[selected_recency]

        def _ep_dt(ep):
            pub = ep.get("published")
            if pub:
                try:
                    return datetime.fromisoformat(pub)
                except Exception:
                    pass
            return None

        if cutoff is not None:
            filtered_episodes = [
                ep for ep in st.session_state.episodes
                if (_ep_dt(ep) or now_utc) >= now_utc - cutoff
            ]
            # Fall back to all if filter returns nothing
            if not filtered_episodes:
                filtered_episodes = st.session_state.episodes
                with info_col:
                    st.warning("⚠️ No episodes in that range — showing all.")
            else:
                with info_col:
                    st.caption(f"Showing {len(filtered_episodes)} of {len(st.session_state.episodes)} episodes.")
        else:
            filtered_episodes = st.session_state.episodes
            with info_col:
                st.caption(f"Showing all {len(filtered_episodes)} episodes.")

        # Build display labels: prepend date when available
        def _ep_label(ep):
            dt = _ep_dt(ep)
            date_str = dt.strftime("%Y-%m-%d") if dt else "date unknown"
            return f"[{date_str}]  {ep['title']}"

        episode_labels = [_ep_label(ep) for ep in filtered_episodes]
        selected_label = st.selectbox(
            "Choose an episode:",
            options=episode_labels,
            key="episode_selector"
        )

        selected_index_in_filtered = episode_labels.index(selected_label)
        selected_episode = filtered_episodes[selected_index_in_filtered]
        selected_title = selected_episode['title']
        if st.session_state.selected_episode_title != selected_title:
            reset_rag_state(clear_audio=True)
        st.session_state.selected_episode_title = selected_title
        st.session_state.current_episode_cache_key = get_episode_cache_key(st.session_state.selected_feed_name, selected_episode)
        cached_payload = load_cached_transcription(st.session_state.current_episode_cache_key)
        cache_exists = is_cache_payload_valid(
            cached_payload,
            st.session_state.selected_feed_name,
            st.session_state.selected_episode_title,
            selected_episode.get("url")
        )
    
        ep_dt = _ep_dt(selected_episode)
        date_display = ep_dt.strftime("%B %d, %Y") if ep_dt else "date unknown"
        st.info(f"**Selected:** {selected_episode['title']}  \n📅 Published: {date_display}")
        if cache_exists:
            st.success("✅ Cached transcript found for this episode. You can skip re-transcription.")
    
        if st.button("⬇️ Download Audio", use_container_width=True):
            with st.spinner("Downloading audio file..."):
                try:
                    reset_rag_state(clear_audio=False)
                    st.session_state.audio_path = download_selected_audio(selected_episode, selected_index_in_filtered)
                    st.success(f"✅ Audio downloaded: {st.session_state.audio_path}")
                    st.info(f"File size: {os.path.getsize(st.session_state.audio_path) / (1024*1024):.2f} MB")
                except Exception as e:
                    st.error(f"❌ Error downloading audio: {str(e)}")
    
        st.divider()

    # Step 3: Initialize Models and Process Audio
    if st.session_state.audio_path or st.session_state.current_episode_cache_key:
        st.header("🤖 Step 3: Initialize Models & Process Audio")
    
        if st.button("🚀 Start Transcription", use_container_width=True):
            try:
                cache_key = st.session_state.current_episode_cache_key
                cached_payload = load_cached_transcription(cache_key) if cache_key else None
                if cached_payload and not is_cache_payload_valid(
                    cached_payload,
                    st.session_state.selected_feed_name,
                    st.session_state.selected_episode_title,
                    selected_episode.get("url")
                ):
                    logging.warning(" Cache payload did not match selected episode source. Ignoring cache.")
                    cached_payload = None

                if cached_payload:
                    st.session_state.transcription_parts = cached_payload["transcription_parts"]
                    st.success(f"✅ Loaded transcription from cache! ({len(st.session_state.transcription_parts)} chunks)")
                else:
                    if not st.session_state.audio_path:
                        raise ValueError("No audio file available for this episode. Download audio first or use a cached transcript.")

                    # Initialize models
                    with st.spinner("Loading Whisper model..."):
                        if st.session_state.model is None:
                            st.session_state.model, st.session_state.processor = initialize_audio_models()
                            st.success("✅ Model loaded!")
                
                    # Process audio
                    with st.spinner("Transcribing audio... This may take a few minutes."):
                        st.session_state.transcription_parts = process_podcast_audio(
                            st.session_state.audio_path,
                            st.session_state.model,
                            st.session_state.processor
                        )
                        st.success(f"✅ Transcription complete! ({len(st.session_state.transcription_parts)} chunks)")
            
                # Generate embeddings
                with st.spinner("Generating embeddings..."):
                    st.session_state.teapot_ai = generate_embeddings(st.session_state.transcription_parts)
                    st.session_state.retrieval_index = build_retrieval_index(st.session_state.transcription_parts)
                    st.session_state.transcript_source = {
                        "audio_path": st.session_state.audio_path,
                        "feed_name": st.session_state.selected_feed_name,
                        "episode_title": st.session_state.selected_episode_title,
                        "episode_url": selected_episode.get("url")
                    }
                    st.success("✅ Embeddings and retrieval index generated! Ready for queries.")

                    if not cached_payload and cache_key:
                        save_cached_transcription(
                            cache_key,
                            st.session_state.transcription_parts,
                            st.session_state.transcript_source
                        )

                with st.expander("📜 Full Transcription", expanded=False):
                    st.text_area(
                        "Full transcription",
                        "\n\n".join(st.session_state.transcription_parts),
                        height=400,
                        key="full_transcription_view"
                    )
        
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
    
        st.divider()

    # Step 4: Query the Podcast
    if st.session_state.retrieval_index and st.session_state.active_llm:
        st.header("💬 Step 4: Ask Questions")

        source = st.session_state.transcript_source
        if source:
            st.caption(
                f"Active transcript source: {source.get('feed_name', 'Unknown feed')} — {source.get('episode_title', 'Unknown episode')}"
            )
    
        # Response length slider
        max_tokens = st.slider(
            "📏 Response Length (tokens)",
            min_value=256,
            max_value=4096,
            value=900,
            step=128,
            help="Higher = longer, more detailed answers. Set to 2048+ for blog posts and emails."
        )
        active_model_name = getattr(st.session_state.active_llm, "model", None)
        if model_needs_reasoning_headroom(active_model_name) and max_tokens < REASONING_MODEL_TOKEN_FLOOR:
            st.caption(
                f"🧠 '{active_model_name}' is a reasoning model — token budget will be "
                f"auto-raised to {REASONING_MODEL_TOKEN_FLOOR} so thinking doesn't crowd out the answer."
            )

        # Predefined query buttons
        st.subheader("⚡ Quick Questions:")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🎙️ Who is the host & what is this about?", use_container_width=True):
                set_current_query("Who is the host of this podcast and what is this episode about?")

        with col2:
            if st.button("📋 Get Summary", use_container_width=True):
                set_current_query("Give me a detailed summary of this podcast?")

        with col3:
            if st.button("🚀 Latest Trends & Innovations", use_container_width=True):
                set_current_query("What are the latest trends, frameworks, innovations, and emerging technologies discussed in this episode?")

        # Discover section — tabbed categories of podcast questions (mirrors Perplexity reference)
        with st.expander("🔍 Discover More Questions", expanded=False):
            st.markdown("**Explore the episode by topic**")

            discover_categories = {
                "🌟 Overview": [
                    "What are the main topics covered in this episode?",
                    "What are the key takeaways from this podcast?",
                    "How does this episode start and how does it end?",
                    "What is the central argument or message of this episode?",
                ],
                "👥 People": [
                    "Who are the guests in this episode?",
                    "What is the background of the guest speaker?",
                    "How does the host introduce themselves?",
                    "Who are the notable people mentioned in this episode?",
                ],
                "💡 Insights": [
                    "What are the most interesting insights shared in this episode?",
                    "What surprising or counterintuitive ideas were discussed?",
                    "What advice or recommendations were given?",
                    "What predictions or future trends were mentioned?",
                ],
                "🔧 Technical": [
                    "What technical concepts or technologies were discussed?",
                    "What tools, frameworks, or platforms were mentioned?",
                    "Were any code examples or technical demos described?",
                    "What technical challenges or problems were explored?",
                ],
                "📊 Details": [
                    "What specific data, numbers, or statistics were cited?",
                    "What real-world examples or case studies were used?",
                    "What companies or products were mentioned?",
                    "What research or studies were referenced?",
                ],
                "🗣️ Conversation": [
                    "What questions did the host ask the guest?",
                    "What was the most debated or controversial topic?",
                    "Were there any funny or memorable moments?",
                    "How did the guest respond to the toughest question?",
                ],
                "✍️ Blog Post": [
                    "Write a blog post about this podcast episode covering the key themes, insights, and takeaways in an engaging narrative style.",
                    "Write a short blog post introduction that would make someone want to listen to this episode.",
                    "Write a listicle blog post: '5 things I learned from this podcast episode'.",
                    "Write a thought leadership blog post inspired by the main ideas discussed in this episode.",
                ],
                "📧 Email": [
                    "Write a professional email newsletter highlighting the most important points from this podcast episode.",
                    "Write a short email to a colleague recommending this podcast episode and summarising why it matters.",
                    "Write an email template announcing this podcast episode to a subscriber list, with subject line and body.",
                    "Write an internal team email summarising the key insights and action points from this episode.",
                ],
            }

            tab_names = list(discover_categories.keys())
            tabs = st.tabs(tab_names)

            for tab, category in zip(tabs, tab_names):
                with tab:
                    questions = discover_categories[category]
                    for i, question in enumerate(questions):
                        if st.button(f"💬 {question}", key=f"discover_{category}_{i}", use_container_width=True):
                            set_current_query(question)

        # Voice query input — attendees can ask by holding a mic instead of typing/clicking
        st.subheader("🎤 Or ask by voice:")
        speak_answer = st.checkbox("🔊 Speak the answer aloud", value=False, key="speak_answer_toggle")
        voice_query_audio = st.audio_input("Record your question", key="voice_query_input")

        if voice_query_audio is not None:
            voice_bytes = voice_query_audio.getvalue()
            voice_hash = hashlib.md5(voice_bytes).hexdigest()
            if st.session_state.last_voice_query_hash != voice_hash:
                st.session_state.last_voice_query_hash = voice_hash
                try:
                    with st.spinner("🎤 Transcribing your question..."):
                        if st.session_state.model is None:
                            st.session_state.model, st.session_state.processor = initialize_audio_models()
                        voice_question = transcribe_voice_query(
                            voice_bytes, st.session_state.model, st.session_state.processor
                        )
                    if voice_question:
                        st.success(f'🎤 Heard: "{voice_question}"')
                        set_current_query(voice_question)
                    else:
                        st.warning("Could not understand the recording. Please try again.")
                except Exception as e:
                    st.error(f"❌ Voice transcription error: {str(e)}")

        # Custom query input
        st.subheader("Or ask your own question:")
        custom_query = st.text_input("Enter your question:", key="custom_query_input")
    
        if st.button("🔍 Get Answer", use_container_width=True) and custom_query:
            set_current_query(custom_query)
    
        # Display answer
        if hasattr(st.session_state, 'current_query') and st.session_state.current_query:
            try:
                source = st.session_state.transcript_source
                current_episode_url = selected_episode.get("url")
                if (
                    not source
                    or source.get("feed_name") != st.session_state.selected_feed_name
                    or source.get("episode_title") != st.session_state.selected_episode_title
                    or source.get("episode_url") not in [None, current_episode_url]
                ):
                    raise ValueError(
                        "Transcript/index does not match the currently selected audio. "
                        "Please run transcription again for this episode."
                    )

                should_generate = (
                    st.session_state.needs_generation
                    or st.session_state.last_answer_query != st.session_state.current_query
                    or not st.session_state.last_answer
                )

                if should_generate:
                    # Use streaming inference — mirrors the Perplexity reference flow
                    answer_stream, retrieved_chunks = get_response(
                        st.session_state.current_query,
                        st.session_state.active_llm,
                        st.session_state.retrieval_index,
                        top_k=8,
                        stream=True,
                        max_tokens=max_tokens,
                    )
                else:
                    answer_stream = st.session_state.last_answer
                    retrieved_chunks = [None] * int(st.session_state.last_retrieved_count)

                st.subheader("Question:")
                st.info(st.session_state.current_query)
                st.caption(f"Retrieved {len(retrieved_chunks)} relevant chunks.")

                st.subheader("Answer:")
                # If the answer is already a plain string (e.g. host extraction shortcut),
                # show it directly; otherwise stream it token-by-token.
                if isinstance(answer_stream, str):
                    answer = answer_stream
                    st.success(answer)
                else:
                    with st.chat_message("assistant"):
                        answer = st.write_stream(answer_stream)
                    logging.info(f" Streamed response complete. Length: {len(answer)}")

                if speak_answer and answer:
                    if (
                        st.session_state.last_answer_audio_query != st.session_state.current_query
                        or not st.session_state.last_answer_audio
                    ):
                        audio_cache_path = get_audio_response_cache_path(answer)
                        if os.path.isfile(audio_cache_path):
                            with open(audio_cache_path, "rb") as cached_audio_file:
                                st.session_state.last_answer_audio = cached_audio_file.read()
                            st.session_state.last_answer_audio_query = st.session_state.current_query
                        else:
                            try:
                                with st.spinner("🔊 Generating audio response..."):
                                    if st.session_state.tts_synthesiser is None:
                                        st.session_state.tts_synthesiser, st.session_state.tts_speaker_embedding = initialize_tts_models()
                                    audio_bytes = synthesize_speech_response(
                                        answer, st.session_state.tts_synthesiser, st.session_state.tts_speaker_embedding
                                    )
                                    with open(audio_cache_path, "wb") as cached_audio_file:
                                        cached_audio_file.write(audio_bytes)
                                    st.session_state.last_answer_audio = audio_bytes
                                    st.session_state.last_answer_audio_query = st.session_state.current_query
                            except Exception as e:
                                st.warning(f"⚠️ Could not generate audio response: {str(e)}")
                                st.session_state.last_answer_audio = None
                    if st.session_state.last_answer_audio:
                        st.audio(st.session_state.last_answer_audio, format="audio/wav")

                st.session_state.last_answer = answer or ""
                st.session_state.last_answer_query = st.session_state.current_query
                st.session_state.last_retrieved_count = len(retrieved_chunks)
                st.session_state.needs_generation = False

                exportable_query = any(
                    phrase in st.session_state.current_query.lower()
                    for phrase in ["blog post", "email"]
                )

                if (
                    exportable_query
                    and st.session_state.last_answer
                    and st.session_state.last_answer_query == st.session_state.current_query
                ):
                    with st.expander("📋 Copy / Download", expanded=False):
                        st.caption("For blog post/email responses only.")

                        safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "_", st.session_state.selected_episode_title or "podcast")[:60]
                        download_name = f"{safe_title}_response.txt"
                        st.download_button(
                            label="⬇️ Download response (.txt)",
                            data=st.session_state.last_answer,
                            file_name=download_name,
                            mime="text/plain",
                            use_container_width=True,
                            key="download_answer_button"
                        )

                        st.text_area(
                            "Copy response",
                            value=st.session_state.last_answer,
                            height=180,
                            key="copy_ready_answer"
                        )

            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                st.session_state.last_answer = ""
                st.session_state.last_answer_query = ""
                st.session_state.last_retrieved_count = 0

# Footer
st.divider()
st.caption("Podcast RAG Application | Powered by Intel® Core™ Ultra Processors with PyTorch XPU")
