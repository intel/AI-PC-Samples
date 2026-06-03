"""
AI Super-Resolution on Intel® Core™ Ultra Processors

This application demonstrates the Neural Processing Unit (NPU) capabilities of 
Intel® Core™ Ultra Processors for AI-powered image upscaling.

Showcases heterogeneous computing across:
- NPU: Dedicated AI acceleration for inference
- GPU: Video decoding and preprocessing
- CPU: System orchestration and baseline comparison

Optimized workload placement for maximum efficiency on Intel AI PC architecture.

OPTIMIZATION: Smart image resizing to avoid NPU recompilation - all images are 
resized to match the first compiled dimension for faster processing.

ENHANCED: Added multiple sample video options including One Piece demo video
"""
import streamlit as st
import cv2
import numpy as np
import torch
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
import time
import tempfile
import os
import subprocess
from nncf import CompressWeightsMode, compress_weights
from pathlib import Path
import matplotlib.pyplot as plt
import hashlib
import threading
from collections import deque

# Model imports
from bsrgan_helper import BSRGAN

# Pre/Post processing imports
from bsrgan_utils import imread_uint
from sample_utils import preprocess, postprocess

# Video imports
from sample_utils import collect_all_frames, write_all_frames, resize_video, download_file, download_file_to_memory

# COCO classes for object detection (matching gpu-device.ipynb)
COCO_CLASSES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
    "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror",
    "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush", "hair brush"
]

def generate_class_colors():
    """Generate consistent colors for each object class using HSV color space"""
    np.random.seed(42)  # Fixed seed for consistent colors
    colors = {}
    num_classes = len(COCO_CLASSES)
    
    for idx, class_name in enumerate(COCO_CLASSES):
        # Generate colors in HSV space for better distribution
        hue = (idx * 137.5) % 360  # Golden angle for even distribution
        saturation = 0.7 + (idx % 3) * 0.1  # Vary saturation
        value = 0.8 + (idx % 2) * 0.15  # Vary brightness
        
        # Convert HSV to RGB
        h_norm = hue / 360.0
        c = value * saturation
        x = c * (1 - abs((h_norm * 6) % 2 - 1))
        m = value - c
        
        if h_norm < 1/6:
            r, g, b = c, x, 0
        elif h_norm < 2/6:
            r, g, b = x, c, 0
        elif h_norm < 3/6:
            r, g, b = 0, c, x
        elif h_norm < 4/6:
            r, g, b = 0, x, c
        elif h_norm < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        # Convert to 0-255 range
        colors[class_name] = (
            int((r + m) * 255),
            int((g + m) * 255),
            int((b + m) * 255)
        )
    
    # Override specific colors for common objects
    colors["apple"] = (220, 20, 60)  # Crimson red
    colors["banana"] = (255, 255, 0)  # Yellow
    colors["orange"] = (255, 140, 0)  # Dark orange
    colors["broccoli"] = (34, 139, 34)  # Forest green
    colors["carrot"] = (255, 140, 0)  # Orange
    colors["person"] = (0, 191, 255)  # Deep sky blue
    colors["dog"] = (139, 69, 19)  # Saddle brown
    colors["cat"] = (255, 105, 180)  # Hot pink
    colors["car"] = (70, 130, 180)  # Steel blue
    colors["bicycle"] = (50, 205, 50)  # Lime green
    
    return colors

CLASS_COLORS = generate_class_colors()

# Page configuration
st.set_page_config(
    page_title="AI Super-Resolution with Intel® Core™ Ultra Processors",
    page_icon="🚀",
    layout="wide"
)

# Title and description
st.title("🚀 AI Super-Resolution with Intel® Core™ Ultra Processors")
st.caption("CPU vs NPU image upscaling comparison · Intel® AI Boost (NPU) · OpenVINO™ · BSRGAN Super-Resolution")

with st.expander("ℹ️ About this demo", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Experience NPU-accelerated AI inference on Intel® Core™ Ultra Processors:**
- 🧠 **NPU**: Dedicated AI acceleration for neural network inference
- 💻 **CPU**: Orchestration & baseline comparison
- 🎨 **GPU**: Preprocessing, color conversions & object detection
""")
    with col_b:
        st.markdown("""
**🖥️ Intel® Core™ Ultra — Heterogeneous Computing:**
- **Image Upscaling**: AI-powered super-resolution on NPU
- **Color Conversions & Detection**: GPU-accelerated preprocessing
- **Baseline Comparison**: CPU-based reference implementation
- **Orchestration**: Intelligent workload distribution
""")

# Create cache directory
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Create compressed models cache directory
COMPRESSED_MODELS_DIR = CACHE_DIR / "compressed_models"
COMPRESSED_MODELS_DIR.mkdir(exist_ok=True)

# Initialize session state for compiled dimensions
if 'compiled_dimensions' not in st.session_state:
    st.session_state.compiled_dimensions = None

# Initialize session state for object detection model
if 'detection_model' not in st.session_state:
    st.session_state.detection_model = None


def check_device_availability():
    """Check NPU and GPU availability"""
    core = ov.Core()
    available_devices = core.available_devices
    
    info = {
        'npu_available': "NPU" in available_devices,
        'gpu_available': "GPU" in available_devices,
        'xpu_available': torch.xpu.is_available(),
        'devices': available_devices
    }
    
    if info['npu_available']:
        try:
            info['npu_name'] = core.get_property("NPU", props.device.full_name)
        except:
            info['npu_name'] = "Unknown NPU"
    
    if info['gpu_available']:
        try:
            info['gpu_name'] = core.get_property("GPU", props.device.full_name)
        except:
            info['gpu_name'] = "Unknown GPU"
    
    return info


# Check device availability at startup
device_info = check_device_availability()

# Sidebar configuration - Configuration section comes first
st.sidebar.header("Configuration")
scaling_factor = st.sidebar.selectbox("Scaling Factor", [2, 4], index=1)

# Resize dimension selector
resize_dimension = st.sidebar.selectbox(
    "Target Resize Dimension",
    [150, 300, 450, 600],
    index=0,
    help="Select the target dimension for resizing images (width x height)"
)

# Display current compiled dimensions and reset button
if st.session_state.compiled_dimensions is not None:
    h, w = st.session_state.compiled_dimensions
    st.sidebar.info(f"📐 Current compiled: {w}×{h}")
    if st.sidebar.button("🔄 Reset Compiled Dimensions"):
        st.session_state.compiled_dimensions = None
        st.sidebar.success("✅ Reset! Next image will compile to selected dimension")
        st.rerun()

# Model selection
model_options = {
    "BSRGAN (Recommended)": "kadirnar/bsrgan",
    "BSRGANx2": "kadirnar/BSRGANx2",
    "RRDB_PSNR_x4": "kadirnar/RRDB_PSNR_x4",
}

selected_model_name = st.sidebar.selectbox("Model", list(model_options.keys()))
selected_model = model_options[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Device Status")

try:
    core = ov.Core()
    available_devices = core.available_devices
    st.sidebar.text(f"Devices: {', '.join(available_devices)}")
    
    if device_info['npu_available']:
        st.sidebar.success(f"✅ NPU: {device_info.get('npu_name', 'Available')}")
        with st.sidebar.expander("🔍 NPU Properties", expanded=False):
            try:
                supported_properties = core.get_property("NPU", props.supported_properties)
                key_props = {
                    "FULL_DEVICE_NAME": "Device Name",
                    "PERFORMANCE_HINT": "Performance Hint",
                    "OPTIMIZATION_CAPABILITIES": "Supported Data Types"
                }
                st.markdown("**Key Properties:**")
                for prop_key, display_name in key_props.items():
                    if prop_key in supported_properties:
                        try:
                            prop_val = core.get_property("NPU", getattr(props, prop_key.lower().replace('_', '.')))
                        except:
                            try:
                                prop_val = core.get_property("NPU", prop_key)
                            except:
                                prop_val = "N/A"
                        st.text(f"{display_name}: {prop_val}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.sidebar.warning("⚠️ NPU not detected")
    
    if device_info['gpu_available']:
        st.sidebar.success(f"✅ GPU: {device_info.get('gpu_name', 'Available')}")
        with st.sidebar.expander("🔍 GPU Properties", expanded=False):
            try:
                supported_properties = core.get_property("GPU", props.supported_properties)
                key_props = {
                    "FULL_DEVICE_NAME": "Device Name",
                    "PERFORMANCE_HINT": "Performance Hint",
                    "OPTIMIZATION_CAPABILITIES": "Supported Data Types"
                }
                st.markdown("**Key Properties:**")
                for prop_key, display_name in key_props.items():
                    if prop_key in supported_properties:
                        try:
                            prop_val = core.get_property("GPU", getattr(props, prop_key.lower().replace('_', '.')))
                        except:
                            try:
                                prop_val = core.get_property("GPU", prop_key)
                            except:
                                prop_val = "N/A"
                        st.text(f"{display_name}: {prop_val}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.sidebar.warning("⚠️ GPU not detected")
    
    if device_info['xpu_available']:
        st.sidebar.success("✅ Intel XPU (PyTorch): Available")
    
except Exception as e:
    st.sidebar.error(f"Error: {str(e)}")

st.sidebar.markdown("---")


def get_model_hash(model_name, height, width):
    """Generate a unique hash for model+dimensions combination"""
    hash_input = f"{model_name}_{height}_{width}"
    return hashlib.md5(hash_input.encode()).hexdigest()


@st.cache_resource
def load_model(model_name, device_type, _device_info):
    """Load the BSRGAN model with proper device detection"""
    status_msgs = []
    
    # Determine device based on device_type
    if device_type == "CPU":
        device = torch.device("cpu")
        status_msgs.append(("info", "✅ Using CPU"))
    elif device_type == "GPU":
        if _device_info['xpu_available']:
            device = torch.device("xpu")
            status_msgs.append(("info", "✅ Using Intel GPU (XPU)"))
        else:
            device = torch.device("cpu")
            status_msgs.append(("warning", "⚠️ Intel GPU (XPU) not available, using CPU instead"))
    else:  # NPU
        if _device_info['xpu_available']:
            device = torch.device("xpu")
            status_msgs.append(("info", "✅ Loading on Intel XPU for faster model compilation"))
        else:
            device = torch.device("cpu")
            status_msgs.append(("info", "ℹ️ Loading on CPU"))
        
        if not _device_info['npu_available']:
            st.error(f"❌ NPU not available on this system")
    
    # Load model
    model = BSRGAN(model_name, device=device, hf_model=True).model
    
    # Log actual device for transparency
    actual_device = next(model.parameters()).device
    status_msgs.append(("info", f"📍 PyTorch model loaded on: {actual_device}"))
    
    return model, status_msgs


@st.cache_resource
def get_or_create_compressed_model(_pytorch_model, height, width, model_name):
    """Get compressed model from cache or create and cache it"""
    model_hash = get_model_hash(model_name, height, width)
    compressed_model_path = COMPRESSED_MODELS_DIR / f"compressed_{model_hash}.xml"
    compressed_weights_path = COMPRESSED_MODELS_DIR / f"compressed_{model_hash}.bin"
    
    core = ov.Core()
    
    if compressed_model_path.exists() and compressed_weights_path.exists():
        compressed_model = core.read_model(compressed_model_path)
        return compressed_model, True
    
    model_device = next(_pytorch_model.parameters()).device
    
    ov_model = ov.convert_model(
        _pytorch_model,
        input=[1, 3, height, width],
        example_input=torch.randn(1, 3, height, width).to(model_device)
    )
    
    compressed_model = compress_weights(ov_model, mode=CompressWeightsMode.INT4_SYM)
    ov.save_model(compressed_model, compressed_model_path)
    
    return compressed_model, False


@st.cache_resource
def get_compiled_model_cached(_pytorch_model, height, width, device_name, model_name, cache_mode="openvino"):
    """Cached model compilation for any device"""
    compressed_model, was_cached = get_or_create_compressed_model(_pytorch_model, height, width, model_name)
    
    core = ov.Core()
    compile_config = {}

    # NPU-specific config disabled - default OpenVINO settings are faster
    # if device_name == "NPU":
    #     compile_config = {"NPU_USE_NPUW": "YES", "NPU_BYPASS_UMD_CACHING": "YES", "NPUW_PARALLEL_COMPILE": "YES", "NPUW_FOLD": "YES"}
    
    if cache_mode == "openvino":
        compile_config[props.cache_dir()] = str(CACHE_DIR)
    
    compile_config[hints.performance_mode()] = hints.PerformanceMode.LATENCY
    
    compiled_model = core.compile_model(
        compressed_model, 
        device_name=device_name,
        config=compile_config
    )
    
    return compiled_model, was_cached


def load_detection_model():
    """Load object detection model on GPU for preprocessing"""
    if st.session_state.detection_model is not None:
        return st.session_state.detection_model
    
    try:
        import huggingface_hub as hf_hub
        
        core = ov.Core()
        
        # Download model from Hugging Face (same as gpu-device.ipynb)
        base_model_dir = CACHE_DIR / "model"
        model_name = "ssdlite_mobilenet_v2_fp16"
        ov_model_path = base_model_dir / model_name / f"{model_name}.xml"
        
        if not ov_model_path.exists():
            with st.spinner("📥 Downloading object detection model from Hugging Face..."):
                hf_hub.snapshot_download(
                    "katuni4ka/ssdlite_mobilenet_v2_fp16", 
                    local_dir=str(base_model_dir / model_name)
                )
        
        # Read and compile model on GPU
        model = core.read_model(str(ov_model_path))
        compiled_model = core.compile_model(
            model=model, 
            device_name="GPU",
            config={hints.performance_mode(): hints.PerformanceMode.THROUGHPUT}
        )
        
        st.session_state.detection_model = compiled_model
        st.success("✅ Object detection model loaded successfully!")
        return compiled_model
        
    except Exception as e:
        error_msg = str(e)
        st.warning(f"⚠️ Could not load detection model: {error_msg[:150]}...")
        st.info("💡 Object detection is optional. Continuing without it.")
        return None

def detect_and_overlay_colors(frame, detection_model=None, min_confidence=0.5):
    """
    Detect objects in frame and apply color coding to entire image based on dominant objects
    Returns: (colored_frame, detected_objects_metadata)
    """
    if detection_model is None:
        return frame, []
    
    try:
        h, w = frame.shape[:2]
        
        # Get model input/output layers
        input_layer = detection_model.input(0)
        output_layer = detection_model.output(0)
        
        # Get input shape (should be [1, 300, 300, 3] for ssdlite_mobilenet_v2)
        num, height_model, width_model, channels = input_layer.shape
        
        # Resize and preprocess for detection
        resized_frame = cv2.resize(frame, (width_model, height_model))
        input_frame = np.expand_dims(resized_frame, axis=0)
        
        # Run detection
        predictions = detection_model([input_frame])[output_layer]
        
        # Create a copy for color transformation
        colored_frame = frame.copy()
        detected_objects = []
        
        # Collect all detected objects with their areas
        detections = []
        
        # Process detections - prediction format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        for prediction in np.squeeze(predictions):
            confidence = prediction[2]
            if confidence > min_confidence:
                class_id = int(prediction[1])
                if class_id < len(COCO_CLASSES):
                    class_name = COCO_CLASSES[class_id]
                    color = CLASS_COLORS.get(class_name, (0, 255, 0))
                    
                    # Get bounding box coordinates (normalized to 0-1)
                    x_min = max(0, int(prediction[3] * w))
                    y_min = max(0, int(prediction[4] * h))
                    x_max = min(w, int(prediction[5] * w))
                    y_max = min(h, int(prediction[6] * h))
                    
                    # Skip if bounding box is invalid
                    if x_max <= x_min or y_max <= y_min:
                        continue
                    
                    area = (x_max - x_min) * (y_max - y_min)
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'color': color,
                        'bbox': (x_min, y_min, x_max, y_max),
                        'area': area,
                        'center': ((x_min + x_max) // 2, (y_min + y_max) // 2)
                    })
        
        if not detections:
            return frame, []
        
        # Create color influence map for the entire image
        color_map = np.zeros((h, w, 3), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        # Apply color influence based on object proximity and size
        for det in detections:
            x_min, y_min, x_max, y_max = det['bbox']
            center_x, center_y = det['center']
            color = np.array(det['color'], dtype=np.float32)
            
            # Create distance map from object center
            y_coords, x_coords = np.ogrid[:h, :w]
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            
            # Calculate influence based on distance with smooth gradient
            max_influence_radius = max(h, w) * 0.7  # Objects influence 70% of image
            influence = np.exp(-distances / (max_influence_radius / 1.5))  # Very smooth falloff
            
            # Much more subtle boost within the object boundaries
            mask = np.zeros((h, w), dtype=np.float32)
            mask[y_min:y_max, x_min:x_max] = 1.2  # Subtle boost inside object (reduced from 3.0)
            
            # Very gentle influence to nearby surroundings
            padding = max(30, min(h, w) // 10)
            y_min_ext = max(0, y_min - padding)
            y_max_ext = min(h, y_max + padding)
            x_min_ext = max(0, x_min - padding)
            x_max_ext = min(w, x_max + padding)
            mask[y_min_ext:y_max_ext, x_min_ext:x_max_ext] = np.maximum(
                mask[y_min_ext:y_max_ext, x_min_ext:x_max_ext], 1.1  # Very subtle (reduced from 1.5)
            )
            
            influence = influence * (1 + mask * 0.3)  # Reduce mask effect
            
            # Add weighted color to map
            for c in range(3):
                color_map[:, :, c] += color[c] * influence * det['confidence']
            weight_map += influence * det['confidence']
        
        # Normalize color map
        weight_map = np.maximum(weight_map, 1e-6)  # Avoid division by zero
        for c in range(3):
            color_map[:, :, c] /= weight_map
        
        # Apply subtle color tint to entire image with strong gradient
        color_map = color_map.astype(np.uint8)
        color_intensity = np.clip(weight_map / (weight_map.max() + 1e-6), 0, 1)
        
        # Apply non-linear transformation for smoother gradient
        color_intensity = np.power(color_intensity, 2)  # Square for smoother falloff
        color_intensity = np.stack([color_intensity] * 3, axis=2) * 0.25  # 25% max intensity (reduced from 50%)
        
        # Blend colors across entire image with subtle effect
        colored_frame = (colored_frame * (1 - color_intensity) + color_map * color_intensity).astype(np.uint8)
        
        # Store metadata
        for det in detections:
            detected_objects.append({
                'class': det['class'],
                'confidence': det['confidence'],
                'color': det['color']
            })
        
        return colored_frame, detected_objects
        
    except Exception as e:
        st.warning(f"⚠️ Error in object detection: {str(e)[:100]}...")
        return frame, []

def preprocess_on_gpu(frame, target_height, target_width, apply_detection=False):
    """
    GPU preprocessing with optional object detection and color overlay
    In production, this would use Intel Media SDK or GPU-accelerated CV operations
    Returns: (preprocessed_tensor, gpu_time, detected_objects_metadata)
    """
    start_time = time.time()
    detected_objects = []
    
    # Only do color space conversion if detection is enabled
    if apply_detection:
        # Color space conversion for detection (simulating GPU work)
        if len(frame.shape) == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Apply object detection and color overlay
        detection_model = load_detection_model()
        if detection_model is not None:
            frame_rgb, detected_objects = detect_and_overlay_colors(frame_rgb, detection_model)
        
        preprocessed = preprocess(frame_rgb)
    else:
        # No color conversion - use original frame as-is
        preprocessed = preprocess(frame)
    
    gpu_time = time.time() - start_time
    
    return preprocessed, gpu_time, detected_objects


def resize_to_compiled_dimensions(image, target_height, target_width):
    """Resize image to match compiled model dimensions"""
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def upscale_image_multidevice(image, model, height, width, model_name, use_gpu_preprocessing=True, apply_detection=False):
    """
    Multi-device upscaling pipeline
    - GPU: Preprocessing, object detection, and color overlays
    - NPU: AI inference
    Returns: (output, timings, preprocessing_device, detected_objects)
    """
    timings = {
        'gpu_preprocessing': 0,
        'npu_inference': 0,
        'total': 0
    }
    detected_objects = []
    
    start_total = time.time()
    
    # Stage 1: GPU Preprocessing (with optional object detection)
    if use_gpu_preprocessing and device_info['gpu_available']:
        tensor_img, gpu_time, detected_objects = preprocess_on_gpu(image, height, width, apply_detection=apply_detection)
        timings['gpu_preprocessing'] = gpu_time
        preprocessing_device = "GPU"
    else:
        start_preprocess = time.time()
        tensor_img = preprocess(image)
        timings['gpu_preprocessing'] = time.time() - start_preprocess
        preprocessing_device = "CPU"
    
    # Stage 2: NPU Inference
    compiled_model, _ = get_compiled_model_cached(model, height, width, "NPU", model_name, cache_mode="openvino")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    start_inference = time.time()
    result = compiled_model([tensor_img])[output_layer]
    timings['npu_inference'] = time.time() - start_inference
    
    # Postprocess
    output = postprocess(result)
    
    timings['total'] = time.time() - start_total
    
    return output, timings, preprocessing_device, detected_objects


def upscale_image_cpu_baseline(image, model, height, width, use_gpu_preprocessing=True, apply_detection=False):
    """CPU-only baseline for comparison with optional GPU preprocessing for consistency"""
    model_device = next(model.parameters()).device
    
    start_time = time.time()
    
    # Use same preprocessing as multi-device pipeline for fair comparison
    if use_gpu_preprocessing and device_info['gpu_available']:
        tensor_img, _, _ = preprocess_on_gpu(image, height, width, apply_detection=apply_detection)
    else:
        # Standard preprocessing
        tensor_img = preprocess(image)
    
    # Move to model device and run inference
    if isinstance(tensor_img, torch.Tensor):
        tensor_img = tensor_img.to(model_device)
    
    with torch.no_grad():
        result = model(tensor_img)
    
    # Postprocess
    output = postprocess(result)
    
    total_time = time.time() - start_time
    
    return output, total_time


def benchmark_upscale_openvino(image, compiled_model, device_name, num_runs=10, warmup_runs=3, use_gpu_preprocess=False):
    """Benchmark image upscaling with OpenVINO compiled model"""
    times = []
    
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    # Get image dimensions for GPU preprocessing
    height, width = image.shape[0], image.shape[1]
    
    # Warmup runs
    for _ in range(warmup_runs):
        if use_gpu_preprocess:
            warmup_tensor, _, _ = preprocess_on_gpu(image, height, width)
        else:
            warmup_tensor = preprocess(image)
        _ = compiled_model([warmup_tensor])[output_layer]
    
    # Actual benchmark runs
    for i in range(num_runs):
        if use_gpu_preprocess:
            tensor_img, _, _ = preprocess_on_gpu(image, height, width)
        else:
            tensor_img = preprocess(image)
        
        start_time = time.time()
        result = compiled_model([tensor_img])[output_layer]
        inference_time = time.time() - start_time
        
        times.append(inference_time)
    
    return times


def process_video_multidevice(video_path, model, model_name, status_text, progress_bar, apply_detection=False):
    """
    Multi-device video processing pipeline:
    1. GPU: Video decode, frame preprocessing, color conversions, object detection
    2. NPU: AI upscaling inference
    """
    # Open video
    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    st.info(f"📹 Video: {frame_width}x{frame_height} @ {fps:.1f} fps, {num_frames} frames")
    
    # Compile NPU model
    status_text.text("🧠 Preparing NPU for AI inference...")
    compiled_model, _ = get_compiled_model_cached(
        model, frame_height, frame_width, "NPU", model_name, cache_mode="openvino"
    )
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    # Performance tracking
    gpu_preprocess_times = []
    npu_inference_times = []
    
    # Stage 1: GPU-accelerated video decode and preprocessing
    status_text.text("🎬 Stage 1: GPU decoding and preprocessing frames...")
    frames = []
    preprocessed_frames = []
    all_detected_objects = []  # Collect unique objects across all frames
    
    frame_idx = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        frames.append(frame)
        
        # GPU preprocessing with optional object detection
        start_gpu = time.time()
        preprocessed, _, detected_objects = preprocess_on_gpu(frame, frame_height, frame_width, apply_detection=apply_detection)
        gpu_time = time.time() - start_gpu
        gpu_preprocess_times.append(gpu_time)
        
        # Collect detected objects
        if detected_objects:
            all_detected_objects.extend(detected_objects)
        
        preprocessed_frames.append(preprocessed)
        
        frame_idx += 1
        progress_bar.progress(min(frame_idx / num_frames * 0.5, 0.49))
        if frame_idx % 10 == 0:
            status_text.text(f"🎬 GPU preprocessing: {frame_idx}/{num_frames} frames...")
    
    video.release()
    status_text.text(f"✅ GPU preprocessing complete: {len(frames)} frames")
    time.sleep(0.5)
    
    # Stage 2: NPU Async Inference
    status_text.text("🧠 Stage 2: NPU executing AI upscaling inference...")
    postprocessed_frames = [None] * len(frames)
    
    class ProgressTracker:
        def __init__(self):
            self.count = 0
            self.lock = threading.Lock()
            self.times = []
    
    tracker = ProgressTracker()
    
    def callback(infer_request, userdata):
        postprocessed_frames, frame_idx, tracker, start_time = userdata
        
        inference_time = time.time() - start_time
        
        res = infer_request.get_output_tensor(0).data[0]
        frame = postprocess(res)
        postprocessed_frames[frame_idx] = frame
        
        with tracker.lock:
            tracker.count += 1
            tracker.times.append(inference_time)
    
    # Create async inference queue
    infer_queue = ov.AsyncInferQueue(compiled_model)
    infer_queue.set_callback(callback)
    
    # Submit all preprocessed frames
    for idx, preprocessed in enumerate(preprocessed_frames):
        start_infer = time.time()
        infer_queue.start_async(
            inputs={input_layer.any_name: preprocessed},
            userdata=(postprocessed_frames, idx, tracker, start_infer)
        )
    
    # Monitor progress
    start_npu_time = time.time()
    last_count = 0
    while not infer_queue.is_ready():
        current_count = tracker.count
        if current_count != last_count:
            progress_pct = 0.5 + (current_count / len(frames)) * 0.5
            progress_bar.progress(min(progress_pct, 0.99))
            elapsed = time.time() - start_npu_time
            status_text.text(f"🧠 NPU inference: {current_count}/{len(frames)} frames - {elapsed:.1f}s")
            last_count = current_count
        time.sleep(0.3)
    
    infer_queue.wait_all()
    total_npu_time = time.time() - start_npu_time
    npu_inference_times = tracker.times
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ NPU inference complete: {len(frames)} frames in {total_npu_time:.1f}s")
    
    # Write output video
    status_text.text("💾 Writing output video...")
    temp_output = tempfile.mktemp(suffix=".avi")
    upscaled_video = cv2.VideoWriter(
        temp_output,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (frame_width * scaling_factor, frame_height * scaling_factor),
    )
    
    write_all_frames(postprocessed_frames, upscaled_video)
    upscaled_video.release()
    
    # Convert to H.264
    output_path = tempfile.mktemp(suffix=".mp4")
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_output,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-pix_fmt', 'yuv420p', output_path
        ], check=True, capture_output=True)
        try:
            os.unlink(temp_output)
        except:
            pass
    except:
        output_path = temp_output
    
    # Performance summary
    avg_gpu_time = np.mean(gpu_preprocess_times)  # seconds
    avg_npu_time = np.mean(npu_inference_times)  # seconds
    total_pipeline_time = sum(gpu_preprocess_times) + sum(npu_inference_times)
    pipeline_fps = len(frames) / total_pipeline_time
    
    # Get unique detected objects across all frames
    unique_objects = {}
    for obj in all_detected_objects:
        class_name = obj['class']
        if class_name not in unique_objects or obj['confidence'] > unique_objects[class_name]['confidence']:
            unique_objects[class_name] = obj
    
    perf_summary = {
        'gpu_avg_s': avg_gpu_time,
        'npu_avg_s': avg_npu_time,
        'total_time': total_pipeline_time,
        'fps': pipeline_fps,
        'num_frames': len(frames),
        'detected_objects': list(unique_objects.values())
    }
    
    return output_path, perf_summary


# Main content area
tab1, tab2 = st.tabs([
    "📷 Image Upscaling (Multi-Device)",
    "📊 Performance Analysis"
])

# Tab 1: Image Upscaling with Multi-Device Architecture
with tab1:
    st.header("Image Upscaling with NPU")
    st.markdown("""
    GPU preprocessing with NPU inference compared to CPU baseline.
    """)
    
    # Display compiled dimensions info if available
    if st.session_state.compiled_dimensions:
        h, w = st.session_state.compiled_dimensions
        st.info(f"💡 NPU compiled for {w}x{h}. All images will be resized to this dimension to avoid recompilation.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image to upscale"
        )
        
        # Track uploaded file and auto-uncheck sample image checkbox
        if 'last_uploaded_file' not in st.session_state:
            st.session_state.last_uploaded_file = None
        if 'use_sample_image' not in st.session_state:
            st.session_state.use_sample_image = True
        
        # If a new file is uploaded, uncheck sample image
        if uploaded_file is not None and uploaded_file != st.session_state.last_uploaded_file:
            st.session_state.last_uploaded_file = uploaded_file
            st.session_state.use_sample_image = False
        
        use_sample_image = st.checkbox(
            "Use sample image", 
            value=st.session_state.use_sample_image,
            key="use_sample_image_cb"
        )
        
        # Sync checkbox state back to session state
        st.session_state.use_sample_image = use_sample_image
        
        if use_sample_image:
            sample_url = "https://storage.openvinotoolkit.org/data/test_data/images/dog.jpg"
            with st.spinner("Downloading sample image..."):
                file_bytes = download_file_to_memory(sample_url)
                if file_bytes is not None:
                    image = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        st.error("Failed to decode the sample image.")
                        image = None
                else:
                    st.error("Failed to download the sample image. Check your network connection.")
                    image = None
        elif uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = None
        
        if image is not None:
            resize_option = st.checkbox(
                "Resize to compiled dimension (avoid recompilation)", 
                value=True,
                help="Resize image to match the first compiled dimension. This avoids NPU recompilation for each new image size."
            )
            
            if resize_option:
                st.info(f"📐 Images will be resized to {resize_dimension}x{resize_dimension} (configured in sidebar)")
            
            # Object detection color coding option
            enable_detection = st.checkbox(
                "Enable Object Detection Color Coding",
                value=False,
                help="Apply color overlays based on detected objects (apple=red, banana=yellow, etc.) during GPU preprocessing"
            )
            
            # CPU comparison option
            enable_cpu_comparison = st.checkbox(
                "Compare with CPU baseline",
                value=False,
                help="Run CPU baseline for performance comparison (slower)"
            )
            
            # If we don't have compiled dimensions yet, or resize is off, resize to selected dimension
            if resize_option:
                if st.session_state.compiled_dimensions is None:
                    # First run - resize to selected dimension and store these dimensions
                    image = cv2.resize(image, (resize_dimension, resize_dimension))
                    st.session_state.compiled_dimensions = (image.shape[0], image.shape[1])
                else:
                    # Subsequent runs - resize to match compiled dimensions
                    h, w = st.session_state.compiled_dimensions
                    if image.shape[0] != h or image.shape[1] != w:
                        image = resize_to_compiled_dimensions(image, h, w)
                        st.success(f"✅ Image resized to {w}x{h} to match compiled model")
            
            height, width = image.shape[0], image.shape[1]
            st.image(image, caption=f"Input Image ({width}x{height})", width="stretch")
            
            # Multi-device upscale button
            if st.button("🚀 Run Multi-Device Pipeline", type="primary"):
                with col2:
                    st.subheader("Multi-Device Results")
                    
                    # Show detection status
                    if enable_detection:
                        st.info("🔍 Object Detection: **ENABLED** - Color coding will be applied")
                    else:
                        st.info("🔍 Object Detection: **DISABLED** - No color coding")
                    
                    # Load model
                    model, _ = load_model(selected_model, "NPU", device_info)
                    
                    # Multi-device pipeline
                    st.markdown("### GPU→NPU Pipeline")
                    with st.spinner("Running multi-device pipeline..."):
                        upscaled_gpu_npu, timings, prep_device, detected_objects = upscale_image_multidevice(
                            image, model, height, width, selected_model, 
                            use_gpu_preprocessing=True, apply_detection=enable_detection
                        )
                    
                    st.image(upscaled_gpu_npu, caption="Multi-Device Output", width="stretch")
                    
                    # Display detected objects metadata
                    if detected_objects:
                        st.markdown("**🔍 Detected Objects:**")
                        cols = st.columns(min(len(detected_objects), 4))
                        for idx, obj in enumerate(detected_objects):
                            with cols[idx % 4]:
                                color_hex = "#{:02x}{:02x}{:02x}".format(obj['color'][0], obj['color'][1], obj['color'][2])
                                st.markdown(f"<div style='padding:8px; background-color:{color_hex}20; border-left:4px solid {color_hex}; margin:4px 0;'>"
                                          f"<strong>{obj['class']}</strong><br/>"
                                          f"<small>Confidence: {obj['confidence']:.1%}</small></div>", 
                                          unsafe_allow_html=True)
                    
                    # Show pipeline breakdown
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("GPU Preprocessing", f"{timings['gpu_preprocessing']:.3f}s")
                    with col_b:
                        st.metric("NPU Inference", f"{timings['npu_inference']:.3f}s")
                    with col_c:
                        st.metric("Total Pipeline", f"{timings['total']:.3f}s")
                    
                    # CPU baseline comparison (only if enabled)
                    if enable_cpu_comparison:
                        st.markdown("### CPU Baseline (for comparison)")
                        cpu_model, _ = load_model(selected_model, "CPU", device_info)
                        with st.spinner("Running CPU baseline..."):
                            upscaled_cpu, cpu_time = upscale_image_cpu_baseline(
                                image, cpu_model, height, width, 
                                use_gpu_preprocessing=True, apply_detection=enable_detection
                            )
                        
                        st.image(upscaled_cpu, caption="CPU Baseline Output", width="stretch")
                        st.metric("CPU Total Time", f"{cpu_time:.3f}s")
                        
                        # Speedup calculation
                        speedup = cpu_time / timings['total']
                        st.success(f"✅ Multi-Device Pipeline is **{speedup:.2f}x faster** than CPU baseline!")
                        
                        # Architecture visualization
                        st.markdown("### Performance Summary")
                        st.code(f"""
GPU→NPU Pipeline:
  Preprocessing: {timings['gpu_preprocessing']:.3f}s
  Inference:     {timings['npu_inference']:.3f}s
  Total:         {timings['total']:.3f}s

CPU Baseline:    {cpu_time:.3f}s
  
Speedup: {speedup:.2f}x
                        """)
                    else:
                        # Show NPU-only performance summary
                        st.markdown("### Performance Summary")
                        st.code(f"""
GPU→NPU Pipeline:
  Preprocessing: {timings['gpu_preprocessing']:.3f}s
  Inference:     {timings['npu_inference']:.3f}s
  Total:         {timings['total']:.3f}s
                        """)
                    
                    st.balloons()
    
    with col2:
        if 'image' not in locals() or image is None:
            st.info("👈 Upload an image or use the sample to get started!")
    
    # Batch Processing Section
    st.markdown("---")
    st.header("📁 Batch Image Processing")
    st.markdown("Upload multiple images to process them all together with the same pipeline.")
    
    batch_files = st.file_uploader(
        "Upload multiple images for batch processing",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        help="Select multiple images to upscale in batch"
    )
    
    if batch_files:
        st.success(f"✅ {len(batch_files)} images uploaded")
        
        batch_resize_option = st.checkbox(
            "Resize all images to compiled dimension (avoid recompilation)", 
            value=True,
            help="Resize all images to match the first compiled dimension for consistent processing.",
            key="batch_resize"
        )
        
        batch_enable_detection = st.checkbox(
            "Enable Object Detection Color Coding for Batch",
            value=False,
            help="Apply color overlays based on detected objects during batch processing",
            key="batch_detection"
        )
        
        batch_enable_cpu_comparison = st.checkbox(
            "Compare with CPU baseline for Batch",
            value=False,
            help="Run CPU baseline for performance comparison (slower)",
            key="batch_cpu_comparison"
        )
        
        if st.button("🚀 Run Batch Processing", type="primary"):
            # Show detection status
            if batch_enable_detection:
                st.info("🔍 Object Detection: **ENABLED** - Color coding will be applied to all images")
            else:
                st.info("🔍 Object Detection: **DISABLED** - No color coding")
            
            # Load model once
            model, _ = load_model(selected_model, "NPU", device_info)
            if batch_enable_cpu_comparison:
                cpu_model, _ = load_model(selected_model, "CPU", device_info)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Store results
            batch_results = []
            total_npu_time = 0
            total_cpu_time = 0
            
            for idx, uploaded_file in enumerate(batch_files):
                status_text.text(f"Processing image {idx + 1}/{len(batch_files)}: {uploaded_file.name}")
                
                # Load image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                batch_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                batch_image = cv2.cvtColor(batch_image, cv2.COLOR_BGR2RGB)
                
                # Apply resize logic
                if batch_resize_option:
                    if st.session_state.compiled_dimensions is None:
                        # First image - resize to selected dimension and store
                        batch_image = cv2.resize(batch_image, (resize_dimension, resize_dimension))
                        st.session_state.compiled_dimensions = (batch_image.shape[0], batch_image.shape[1])
                    else:
                        # Subsequent images - resize to match
                        h, w = st.session_state.compiled_dimensions
                        if batch_image.shape[0] != h or batch_image.shape[1] != w:
                            batch_image = resize_to_compiled_dimensions(batch_image, h, w)
                
                height, width = batch_image.shape[0], batch_image.shape[1]
                
                # Process with NPU
                upscaled_npu, timings, _, detected_objects = upscale_image_multidevice(
                    batch_image, model, height, width, selected_model, 
                    use_gpu_preprocessing=True, apply_detection=batch_enable_detection
                )
                
                # Process with CPU for comparison (only if enabled)
                if batch_enable_cpu_comparison:
                    upscaled_cpu, cpu_time = upscale_image_cpu_baseline(
                        batch_image, cpu_model, height, width, 
                        use_gpu_preprocessing=True, apply_detection=batch_enable_detection
                    )
                else:
                    upscaled_cpu = None
                    cpu_time = 0
                
                # Store results
                result_data = {
                    'name': uploaded_file.name,
                    'input': batch_image,
                    'npu_output': upscaled_npu,
                    'npu_time': timings['total'],
                    'detected_objects': detected_objects
                }
                
                if batch_enable_cpu_comparison:
                    result_data['cpu_output'] = upscaled_cpu
                    result_data['cpu_time'] = cpu_time
                    result_data['speedup'] = cpu_time / timings['total']
                
                batch_results.append(result_data)
                
                total_npu_time += timings['total']
                if batch_enable_cpu_comparison:
                    total_cpu_time += cpu_time
                
                # Update progress
                progress_bar.progress((idx + 1) / len(batch_files))
            
            status_text.text("✅ Batch processing complete!")
            
            # Display summary
            st.markdown("### Batch Processing Summary")
            if batch_enable_cpu_comparison:
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Total NPU Time", f"{total_npu_time:.2f}s")
                with col_s2:
                    st.metric("Total CPU Time", f"{total_cpu_time:.2f}s")
                with col_s3:
                    avg_speedup = total_cpu_time / total_npu_time
                    st.metric("Average Speedup", f"{avg_speedup:.2f}x")
            else:
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric("Total NPU Time", f"{total_npu_time:.2f}s")
                with col_s2:
                    st.metric("Images Processed", len(batch_files))
            
            # Display each result
            st.markdown("### Individual Results")
            for result in batch_results:
                if batch_enable_cpu_comparison:
                    expander_title = f"📷 {result['name']} - Speedup: {result['speedup']:.2f}x"
                else:
                    expander_title = f"📷 {result['name']} - NPU Time: {result['npu_time']:.3f}s"
                
                with st.expander(expander_title):
                    if batch_enable_cpu_comparison:
                        col_r1, col_r2, col_r3 = st.columns(3)
                        
                        with col_r1:
                            st.image(result['input'], caption="Input", use_container_width=True)
                        with col_r2:
                            st.image(result['npu_output'], caption=f"NPU Output ({result['npu_time']:.3f}s)", use_container_width=True)
                        with col_r3:
                            st.image(result['cpu_output'], caption=f"CPU Output ({result['cpu_time']:.3f}s)", use_container_width=True)
                    else:
                        col_r1, col_r2 = st.columns(2)
                        
                        with col_r1:
                            st.image(result['input'], caption="Input", use_container_width=True)
                        with col_r2:
                            st.image(result['npu_output'], caption=f"NPU Output ({result['npu_time']:.3f}s)", use_container_width=True)
                    
                    # Display detected objects metadata if available
                    if result.get('detected_objects'):
                        st.markdown("**🔍 Detected Objects:**")
                        det_cols = st.columns(min(len(result['detected_objects']), 4))
                        for idx, obj in enumerate(result['detected_objects']):
                            with det_cols[idx % 4]:
                                color_hex = "#{:02x}{:02x}{:02x}".format(obj['color'][0], obj['color'][1], obj['color'][2])
                                st.markdown(f"<div style='padding:8px; background-color:{color_hex}20; border-left:4px solid {color_hex}; margin:4px 0;'>"
                                          f"<strong>{obj['class']}</strong><br/>"
                                          f"<small>Confidence: {obj['confidence']:.1%}</small></div>", 
                                          unsafe_allow_html=True)
            
            st.balloons()


# Tab 2: Performance Analysis
with tab2:
    st.header("NPU Performance")
    st.markdown("""
    Compare NPU performance against CPU and GPU for AI inference workloads.
    """)
    
    # Display compiled dimensions info if available
    if st.session_state.compiled_dimensions:
        h, w = st.session_state.compiled_dimensions
        st.info(f"💡 NPU compiled for {w}x{h}. Images will be resized to this dimension.")
    
    # Caching options
    st.subheader("⚡ NPU Caching Strategy")
    
    caching_mode = st.radio(
        "Choose NPU caching mode:",
        ["UMD Caching", "OpenVINO Caching (Recommended)", "No Caching"],
        index=1,  # OpenVINO is default
        help="""
        **OpenVINO Caching (Recommended)**: Application-level caching with fast cache hits. 
        Best performance for dynamic model conversion workflows.
        
        **UMD Caching**: Automatic driver-level caching. 
        Best suited for static IR model files.
        
        **No Caching**: Disable all caching. Useful for measuring true compilation time.
        """
    )
    
    use_ov_cache = caching_mode == "OpenVINO Caching (Recommended)"
    use_no_cache = caching_mode == "No Caching"
    
    # Configuration
    st.subheader("Configuration")
    col_settings1, col_settings2 = st.columns(2)
    
    with col_settings1:
        num_runs = st.slider("Number of runs per device", min_value=5, max_value=20, value=5)
    
    with col_settings2:
        include_gpu_bench = st.checkbox("Include GPU", value=False, 
                                       help="Enable to include GPU comparison")
    
    # GPU Preprocessing toggle
    use_gpu_preprocessing = st.checkbox(
        "Enable GPU Preprocessing", 
        value=device_info['gpu_available'],
        help="Toggle GPU preprocessing with color space conversion."
    )
    
    # Load image
    benchmark_col1, benchmark_col2 = st.columns([1, 2])
    
    with benchmark_col1:
        st.subheader("Image")
        
        # Option to use sample or upload
        use_sample_bench = st.checkbox("Use sample image", value=True, key="bench_sample")
        
        if use_sample_bench:
            sample_url = "https://storage.openvinotoolkit.org/data/test_data/images/dog.jpg"
            with st.spinner("Downloading sample image..."):
                file_bytes = download_file_to_memory(sample_url)
                if file_bytes is not None:
                    bench_image = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if bench_image is not None:
                        bench_image = cv2.cvtColor(bench_image, cv2.COLOR_BGR2RGB)
                    else:
                        st.error("Failed to decode the sample image.")
                        bench_image = None
                else:
                    st.error("Failed to download the sample image. Check your network connection.")
                    bench_image = None
        else:
            uploaded_bench = st.file_uploader(
                "Upload image",
                type=["jpg", "jpeg", "png", "bmp"],
                key="bench_upload"
            )
            if uploaded_bench is not None:
                file_bytes = np.asarray(bytearray(uploaded_bench.read()), dtype=np.uint8)
                bench_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                bench_image = cv2.cvtColor(bench_image, cv2.COLOR_BGR2RGB)
            else:
                bench_image = None
        
        if bench_image is not None:
            # Resize to compiled dimensions if available
            if st.session_state.compiled_dimensions:
                h, w = st.session_state.compiled_dimensions
                if bench_image.shape[0] != h or bench_image.shape[1] != w:
                    bench_image = resize_to_compiled_dimensions(bench_image, h, w)
                    st.success(f"✅ Image resized to {w}x{h}")
            else:
                # First run - resize to selected dimension
                bench_image = cv2.resize(bench_image, (resize_dimension, resize_dimension))
                st.session_state.compiled_dimensions = (bench_image.shape[0], bench_image.shape[1])
            
            height, width = bench_image.shape[0], bench_image.shape[1]
            st.image(bench_image, caption=f"Image ({width}x{height})", width="stretch")
    
    with benchmark_col2:
        if bench_image is not None:
            if st.button("🚀 Run Comprehensive Analysis", type="primary", key="run_benchmark"):
                st.subheader("Results")
                
                # Create placeholders
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                try:
                    # Collect all status messages
                    all_status_msgs = []
                    
                    # Load model (for NPU path, will be loaded on XPU/CPU for conversion)
                    progress_text.text("Loading model...")
                    npu_model, npu_msgs = load_model(selected_model, "NPU", device_info)
                    all_status_msgs.extend(npu_msgs)
                    progress_bar.progress(0.05)
                    
                    # Determine caching mode based on user selection
                    if use_no_cache:
                        cache_mode = None
                    elif use_ov_cache:
                        cache_mode = "openvino"
                    else:  # UMD caching
                        cache_mode = "umd"
                    
                    # Compile for all devices with INT4
                    devices_to_test = ["CPU"]
                    device_labels = ["CPU"]
                    
                    if include_gpu_bench and device_info['gpu_available']:
                        devices_to_test.append("GPU")
                        device_labels.append("GPU")
                    
                    if device_info['npu_available']:
                        devices_to_test.append("NPU")
                        device_labels.append("NPU")
                    
                    compiled_models = {}
                    compile_times = {}
                    
                    progress_step = 0.3 / len(devices_to_test)
                    
                    for idx, device_name in enumerate(devices_to_test):
                        progress_text.text(f"Preparing model for {device_name}...")
                        compile_start = time.time()
                        compiled_model, was_cached = get_compiled_model_cached(
                            npu_model, height, width, device_name, selected_model, cache_mode
                        )
                        actual_compile_time = time.time() - compile_start
                        
                        compiled_models[device_name] = compiled_model
                        compile_times[device_name] = actual_compile_time if actual_compile_time > 0.5 else 0  # 0 means cached
                        
                        progress_bar.progress(0.05 + (idx + 1) * progress_step)
                    
                    # Add compilation info to status messages
                    if any(t > 0 for t in compile_times.values()):
                        all_status_msgs.append(("info", f"📦 Models Prepared | Cache Mode: **{caching_mode}**"))
                    else:
                        all_status_msgs.append(("info", f"📦 Models Loaded from Cache | Cache Mode: **{caching_mode}**"))
                    
                    # Show compilation info
                    with st.expander("🔧 Model Loading & Compilation Details", expanded=False):
                        st.text(f"Available Devices: {', '.join(device_info['devices'])}")
                        st.text(f"NPU Available: {'✅ Yes' if device_info['npu_available'] else '❌ No'}")
                        if device_info['npu_available']:
                            st.text(f"NPU: {device_info.get('npu_name', 'Unknown')}")
                        st.text(f"Intel GPU Available: {'✅ Yes' if device_info['gpu_available'] else '❌ No'}")
                        if device_info['gpu_available']:
                            st.text(f"GPU: {device_info.get('gpu_name', 'Unknown')}")
                        st.text(f"Intel XPU (PyTorch) Available: {'✅ Yes' if device_info['xpu_available'] else '❌ No'}")
                        st.markdown("---")
                        for msg_type, msg in all_status_msgs:
                            if msg_type == "info":
                                st.info(msg)
                            elif msg_type == "warning":
                                st.warning(msg)
                    
                    # Generate sample outputs
                    progress_text.text("Generating sample outputs from all devices...")
                    outputs = {}
                    
                    # Use GPU preprocessing based on user setting
                    if use_gpu_preprocessing and device_info['gpu_available']:
                        preprocessed_tensor, _, _ = preprocess_on_gpu(bench_image, height, width)
                        prep_method = "GPU"
                    else:
                        preprocessed_tensor = preprocess(bench_image)
                        prep_method = "CPU"
                    
                    for device_name in devices_to_test:
                        result = compiled_models[device_name]([preprocessed_tensor])[compiled_models[device_name].output(0)]
                        outputs[device_name] = postprocess(result)
                    progress_bar.progress(0.4)
                    
                    # Run benchmarks
                    all_times = {}
                    
                    # Determine if we should use GPU preprocessing based on user setting
                    use_gpu_prep = use_gpu_preprocessing and device_info['gpu_available']
                    
                    for idx, device_name in enumerate(devices_to_test):
                        progress_text.text(f"Running {device_name} ({num_runs} iterations)...")
                        times = benchmark_upscale_openvino(
                            bench_image, compiled_models[device_name], device_name, 
                            num_runs, warmup_runs=3, use_gpu_preprocess=use_gpu_prep
                        )
                        all_times[device_name] = times
                        progress_bar.progress(0.4 + ((idx + 1) / len(devices_to_test)) * 0.6)
                    
                    progress_text.text("✅ Complete!")
                    progress_bar.progress(1.0)
                    
                    # Calculate statistics
                    stats = {}
                    for device_name in devices_to_test:
                        stats[device_name] = {
                            'mean': np.mean(all_times[device_name]),
                            'std': np.std(all_times[device_name]),
                            'times': all_times[device_name]
                        }
                    
                    # Display statistics
                    st.markdown("### Performance Statistics")
                    
                    # Metrics
                    display_devices = devices_to_test.copy()
                    
                    cols = st.columns(len(display_devices))
                    for idx, device_name in enumerate(display_devices):
                        with cols[idx]:
                            st.metric(
                                f"{device_name} Average",
                                f"{stats[device_name]['mean']:.3f}s",
                                f"±{stats[device_name]['std']:.3f}s"
                            )
                    
                    # Speedup comparisons
                    if len(display_devices) > 1:
                        st.markdown("### Speedup Comparisons")
                        speedup_cols = st.columns(min(len(display_devices), 3))
                        
                        col_idx = 0
                        baseline = "CPU"
                        for target in display_devices:
                            if target != baseline:
                                speedup = stats[baseline]['mean'] / stats[target]['mean']
                                
                                with speedup_cols[col_idx % len(speedup_cols)]:
                                    st.metric(
                                        f"{target} vs {baseline}",
                                        f"{speedup:.2f}x",
                                        delta=f"{((speedup-1)*100):.1f}% faster" if speedup > 1 else f"{((1-speedup)*100):.1f}% slower"
                                    )
                                col_idx += 1
                    
                    # Create comparison chart
                    st.markdown("### Performance Comparison")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Line plot
                    runs = list(range(1, num_runs + 1))
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    markers = ['o', 's', '^', 'D', 'v']
                    
                    for idx, device_name in enumerate(display_devices):
                        ax1.plot(runs, stats[device_name]['times'], f'{markers[idx % len(markers)]}-', 
                                label=device_name, linewidth=2, markersize=6, alpha=0.7, color=colors[idx % len(colors)])
                    
                    ax1.set_xlabel('Run Number', fontsize=11)
                    ax1.set_ylabel('Time (seconds)', fontsize=11)
                    ax1.set_title('Inference Time per Run', fontsize=12, fontweight='bold')
                    ax1.legend(fontsize=9, loc='best')
                    ax1.grid(True, alpha=0.3)
                    
                    # Bar chart
                    means = [stats[d]['mean'] for d in display_devices]
                    stds = [stats[d]['std'] for d in display_devices]
                    display_labels = [d.replace('→', '\n→\n') if '→' in d else d for d in display_devices]
                    
                    bars = ax2.bar(display_labels, means, yerr=stds, capsize=8, 
                                  color=colors[:len(display_devices)], alpha=0.7, 
                                  edgecolor='black', linewidth=1.5)
                    ax2.set_ylabel('Time (seconds)', fontsize=11)
                    ax2.set_title('Average Inference Time', fontsize=12, fontweight='bold')
                    ax2.grid(True, alpha=0.3, axis='y')
                    ax2.tick_params(axis='x', rotation=0, labelsize=9)
                    
                    # Add value labels
                    for bar, mean, std in zip(bars, means, stds):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + std + max(means)*0.02,
                                f'{mean:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display upscaled images
                    st.markdown("### Upscaled Images Comparison")
                    
                    img_cols = st.columns(len(devices_to_test))
                    
                    for idx, device_name in enumerate(devices_to_test):
                        with img_cols[idx]:
                            st.image(outputs[device_name], caption=f"{device_name} Output", width="stretch")
                    
                    # Detailed results table
                    st.markdown("### Detailed Results (Per Run)")
                    
                    results_data = {'Run': list(range(1, num_runs + 1))}
                    
                    for device_name in display_devices:
                        results_data[f'{device_name} (s)'] = [f"{t:.4f}" for t in stats[device_name]['times']]
                    
                    # Add speedup columns
                    if len(display_devices) > 1:
                        baseline = "CPU"
                        for device_name in display_devices:
                            if device_name != baseline:
                                speedups = [stats[baseline]['times'][i] / stats[device_name]['times'][i] for i in range(num_runs)]
                                results_data[f'{device_name}/{baseline}'] = [f"{s:.2f}x" for s in speedups]
                    
                    st.dataframe(results_data, use_container_width=True)
                    
                    # Summary with multi-device insights
                    st.markdown("### Summary & Insights")
                    
                    if device_info['npu_available'] and "NPU" in devices_to_test:
                        speedup_vs_cpu = stats['CPU']['mean'] / stats['NPU']['mean']
                        st.success(f"✅ NPU is {speedup_vs_cpu:.2f}x faster than CPU")
                        
                        if "GPU" in devices_to_test and device_info['gpu_available']:
                            speedup_vs_gpu = stats['GPU']['mean'] / stats['NPU']['mean']
                            st.info(f"ℹ️ NPU is {speedup_vs_gpu:.2f}x vs GPU for AI inference")
                    else:
                        st.success("✅ Analysis Complete!")
                    
                    # Recommendations
                    st.markdown("### Workload Placement Recommendations")
                    
                    if device_info['npu_available']:
                        st.info("""
**Optimal Device Selection:**

**NPU:** Best for AI inference workloads with high throughput

**GPU:** Ideal for preprocessing and parallel operations

**CPU:** Universal fallback, good for single operations and portability
                        """)
                    else:
                        st.info("""
**Device Selection Guidelines:**
- **NPU**: Best for AI inference workloads (when available)
- **GPU**: Good for preprocessing and parallel operations  
- **CPU**: Universal fallback, good for single operations
                        """)
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("👈 Select or upload an image to run the analysis")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About This Demo

Showcases **NPU capabilities** in Intel® Core™ Ultra Processors:

**Smart Resizing:**
- Images resized to first compiled dimension
- Avoids NPU recompilation overhead
- Faster processing for subsequent images

**Multi-Device Pipeline:**
- NPU: AI inference acceleration
- GPU: Video decode, preprocessing & color conversions
- Optimal workload placement

**Key Features:**
- Heterogeneous computing
- Device-specific optimization
- Image upscaling with super-resolution
- GPU-accelerated preprocessing
- Object detection & color coding
- **Multiple sample videos** (Coco, One Piece)
- Performance comparison
- Real-world AI workloads

**Technologies:**
- Intel® Core™ Ultra Processors
- Intel® AI Boost (NPU)
- Intel® Arc™ Graphics (GPU)
- OpenVINO™ Toolkit
- BSRGAN Super-Resolution
""")
