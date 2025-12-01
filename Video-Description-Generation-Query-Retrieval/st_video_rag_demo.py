import os
import base64
import shutil
import logging
import chromadb
import warnings
import ollama
import cv2
import streamlit as st
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DATABASE_PATH = "./Video_descriptions_database_ollama"
COLLECTION_NAME = "Video_descriptions_ollama"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Page config
st.set_page_config(
    page_title="Video RAG with Ollama on Intel GPUs", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #0071c5 0%, #00c7fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .intel-badge {
        background: linear-gradient(90deg, #0071c5 0%, #00c7fd 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #0071c5;
    }
    .search-card-1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .search-card-2 {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .search-card-3 {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎥 Video RAG: Semantic Video Search</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Ollama on Intel® Arc™ & Core™ Ultra Processors</div>', unsafe_allow_html=True)

# Initialize session state
if "collection" not in st.session_state:
    st.session_state.collection = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "video_files" not in st.session_state:
    st.session_state.video_files = []
if "database_loaded" not in st.session_state:
    st.session_state.database_loaded = False

# Sidebar configuration
with st.sidebar:
    st.markdown("### 🎥 Video RAG")
    st.markdown("---")
    
    st.header("⚙️ Configuration")
    
    # Model selection - show all models
    try:
        response = ollama.list()
        all_models = [model.model for model in response.models]
        
        if not all_models:
            st.error("⚠️ No models found! Please pull a model:")
            st.code("ollama pull llava", language="bash")
            all_models = ["llava"]
        
        selected_model = st.selectbox(
            "🤖 Model", 
            all_models,
            key="vision_model"
        )
        
        model_lower = selected_model.lower()
        vision_patterns = ['llava', 'llama3.2-vision', 'minicpm-v', 'qwen', 'cogvlm', 'bakllava']
        
        if any(pattern in model_lower for pattern in vision_patterns):
            if 'llava' in model_lower:
                st.success("✅ Vision model")
            elif 'llama3.2-vision' in model_lower:
                st.info("💡 Vision model")
            elif 'qwen' in model_lower:
                st.success("✅ Vision model")
            else:
                st.info("💡 Vision model detected")
        else:
            st.warning("⚠️ This model may not support vision/images")
            
    except Exception as e:
        st.warning(f"⚠️ Ollama not accessible: {e}")
        st.info("Please ensure Ollama is running: `ollama serve`")
        selected_model = "llava"
    
    max_tokens = 100
    temperature = 0.7
    
    st.markdown("---")
    
    # Dataset configuration
    st.subheader("📁 Dataset")
    dataset_folder = st.text_input("Video Folder", ".")
    max_videos = st.slider("Max Videos", 1, 128, 20)
    
    st.markdown("---")
    
    # Database info
    st.subheader("🗄️ Database")
    if os.path.exists(DATABASE_PATH):
        st.success("✅ Database Ready")
        if st.button("🔄 Reset Database", type="secondary"):
            try:
                shutil.rmtree(DATABASE_PATH)
                st.success("Database reset!")
                st.session_state.collection = None
                st.session_state.database_loaded = False
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("💾 No database found")
    
    st.markdown("---")
    st.markdown("### 🚀 Intel Optimizations")
    st.markdown("✓ Intel Arc™ Graphics")
    st.markdown("✓ Intel Core™ Ultra")
    st.markdown("✓ Hardware Acceleration")


def encode_video_frame(video_path, frame_time=2.0):
    """Extract and encode a frame from video."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(frame_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64
    except Exception as e:
        logging.error(f"Frame encoding error: {e}")
        return None


def generate_video_description_ollama(video_path, model, max_tokens=100, temperature=0.7):
    """Generate video description using Ollama vision model."""
    try:
        frame_base64 = encode_video_frame(video_path, frame_time=2.0)
        
        if not frame_base64:
            return "Unable to process video"
        
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': 'Describe this video frame concisely for search. Include main subjects, actions, setting, and key visual details in 2-3 sentences.',
                'images': [frame_base64]
            }],
            options={
                'num_predict': max_tokens,
                'temperature': temperature,
                'num_ctx': 2048,
                'num_gpu': 99
            }
        )
        
        message = response['message']
        description = message.get('content', '')
        
        if not description or len(description.strip()) == 0:
            description = message.get('thinking', '')
        
        if not description or len(description.strip()) == 0:
            return "Empty response from model"
        
        return description
    except Exception as e:
        logging.error(f"Description generation error: {e}")
        return "Error generating description"


def get_video_paths(folder, max_count):
    """Get video file paths from folder."""
    try:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_files = []
        
        folder_path = os.path.abspath(folder)
        
        for root, dirs, files in os.walk(folder_path):
            video_files.extend([
                os.path.join(root, f) for f in files 
                if any(f.lower().endswith(ext) for ext in video_extensions)
            ])
        
        if len(video_files) > max_count:
            video_files = video_files[:max_count]
        
        return video_files
    except Exception as e:
        logging.error(f"Error finding videos: {e}")
        return []


def initialize_database():
    """Initialize ChromaDB and embedding model."""
    try:
        client = chromadb.PersistentClient(path=DATABASE_PATH)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        return collection, embedding_model
    except Exception as e:
        error_msg = str(e)
        
        # Handle tenant connection issues
        if "Could not connect to tenant" in error_msg or "default_tenant" in error_msg:
            st.error(f"Database initialization error: {e}")
            st.warning("⚠️ ChromaDB database is corrupted or incompatible.")
            
            if st.button("🔄 Reset Database", type="primary"):
                try:
                    # Remove corrupted database
                    if os.path.exists(DATABASE_PATH):
                        shutil.rmtree(DATABASE_PATH)
                        st.success("✅ Database removed. Click 'Start Processing Videos' again.")
                        st.rerun()
                except Exception as reset_error:
                    st.error(f"Failed to reset database: {reset_error}")
            
            st.info("💡 **To fix manually:**\n\n1. Close this app\n2. Delete folder: `Video_descriptions_database_ollama`\n3. Restart the app")
            return None, None
        else:
            st.error(f"Database initialization error: {e}")
            return None, None


def get_existing_descriptions(collection):
    """Get existing descriptions from database."""
    try:
        all_items = collection.get(include=["metadatas"])
        existing = {meta['video_filename']: True for meta in all_items['metadatas']}
        return existing
    except:
        return {}


# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎬 Process Videos", "🔍 Search Videos", "📊 Architecture", "ℹ️ About"])

with tab1:
    st.header("Process Videos and Build Knowledge Base")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>🤖 AI Model</h3><p>Qwen Vision-Language</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>💾 Storage</h3><p>ChromaDB Vector Store</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🔍 Search</h3><p>Semantic Similarity</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🚀 Start Processing Videos", type="primary", use_container_width=True):
        with st.spinner("Initializing AI models on Intel GPU..."):
            collection, embedding_model = initialize_database()
            
            if collection is None:
                st.error("Failed to initialize database")
                st.stop()
            
            st.session_state.collection = collection
            st.session_state.embedding_model = embedding_model
        
        with st.spinner("Scanning for video files..."):
            video_files = get_video_paths(dataset_folder, max_videos)
            
            if not video_files:
                st.error(f"❌ No video files found in '{os.path.abspath(dataset_folder)}'")
                st.info("💡 Supported: .mp4, .avi, .mov, .mkv, .flv, .wmv")
                st.stop()
            
            st.session_state.video_files = video_files
            st.success(f"✅ Found {len(video_files)} videos")
        
        existing = get_existing_descriptions(collection)
        st.info(f"📚 Database contains {len(existing)} existing descriptions")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed = 0
        skipped = 0
        
        for idx, video_file in enumerate(video_files):
            video_filename = os.path.basename(video_file)
            progress = (idx + 1) / len(video_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {idx + 1}/{len(video_files)}: {video_filename}")
            
            if video_filename in existing:
                skipped += 1
                continue
            
            try:
                description = generate_video_description_ollama(video_file, selected_model, max_tokens, temperature)
                embedding = embedding_model.encode(description).tolist()
                
                collection.add(
                    embeddings=[embedding],
                    documents=[description],
                    metadatas=[{"video_filename": video_filename}],
                    ids=[video_file]
                )
                
                processed += 1
                
            except Exception as e:
                st.warning(f"Error processing {video_filename}: {e}")
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        st.success(f"✅ Processing Complete!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Processed", processed, delta=processed)
        col2.metric("Skipped", skipped)
        col3.metric("Total in Database", collection.count())
        
        st.session_state.database_loaded = True
        st.balloons()

with tab2:
    st.header("Semantic Video Search")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="search-card-1"><h3>🔍 AI-Powered</h3><p>Semantic Understanding</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="search-card-2"><h3>⚡ Lightning Fast</h3><p>Vector Search</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="search-card-3"><h3>🎯 Accurate Results</h3><p>Ranked by Relevance</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if not st.session_state.database_loaded:
        if os.path.exists(DATABASE_PATH):
            with st.spinner("Loading knowledge base..."):
                collection, embedding_model = initialize_database()
                if collection:
                    st.session_state.collection = collection
                    st.session_state.embedding_model = embedding_model
                    st.session_state.database_loaded = True
                    st.success(f"✅ Loaded {collection.count()} video descriptions")
    
    if not st.session_state.database_loaded:
        st.warning("⚠️ Please process videos first")
    else:
        with st.expander("📋 View All Video Descriptions", expanded=False):
            try:
                collection = st.session_state.collection
                all_data = collection.get(include=["documents", "metadatas"])
                
                st.info(f"📊 Total videos: {len(all_data['documents'])}")
                
                for i, (doc, metadata) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
                    st.markdown(f"**{i+1}. {metadata['video_filename']}**")
                    st.text(doc)
                    if i < len(all_data['documents']) - 1:
                        st.markdown("---")
            except Exception as e:
                st.error(f"Error loading descriptions: {e}")
        
        st.markdown("### 🔍 Enter Your Search Query")
        query = st.text_input("", placeholder="e.g., person playing basketball, sunset over ocean, dog running in park", label_visibility="collapsed")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("💡 Try: 'person', 'animal', 'outdoor scene', 'sports activity', or keywords from descriptions above")
        with col2:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if search_button and query:
            with st.spinner("Searching with AI..."):
                try:
                    collection = st.session_state.collection
                    embedding_model = st.session_state.embedding_model
                    
                    query_embedding = embedding_model.encode(query).tolist()
                    
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=3,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    st.success("✅ Search Complete!")
                    st.balloons()
                    st.markdown(f"**Query:** *{query}*")
                    st.markdown("---")
                    
                    for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                    )):
                        similarity_score = 1 - distance
                        video_path = results['ids'][0][i]
                        
                        if similarity_score > 0.7:
                            score_emoji = "🟢"
                            score_label = "Excellent Match"
                        elif similarity_score > 0.5:
                            score_emoji = "🟡"
                            score_label = "Good Match"
                        else:
                            score_emoji = "🟠"
                            score_label = "Moderate Match"
                        
                        with st.expander(f"{score_emoji} Result {i+1}: {metadata['video_filename']} - {score_label} ({similarity_score:.2%})", expanded=(i==0)):
                            col1, col2 = st.columns([3, 2])
                            
                            with col1:
                                st.markdown("#### 📝 Video Description")
                                st.info(doc)
                                st.markdown("#### 📊 Similarity Metrics")
                                st.progress(similarity_score)
                                st.caption(f"Similarity Score: {similarity_score:.2%} | Distance: {distance:.3f}")
                            
                            with col2:
                                if os.path.exists(video_path):
                                    st.markdown("#### 🎬 Video Preview")
                                    st.video(video_path)
                                else:
                                    st.warning("Video file not found")
                    
                except Exception as e:
                    st.error(f"Search error: {e}")

with tab3:
    st.header("System Architecture")
    
    st.markdown("""
    ### Video RAG Pipeline on Intel Hardware
    
    This system demonstrates the power of Intel GPUs for AI workloads, using Ollama for efficient model inference.
    """)
    
    # Clean architecture diagram using Streamlit columns
    st.markdown("### 🔄 System Architecture Flow")
    
    st.info("**Video Processing Pipeline** - How videos are converted to searchable descriptions")
    
    cols = st.columns(7)
    with cols[0]:
        st.markdown("### 📹\n**Video Files**\nInput videos")
    with cols[1]:
        st.markdown("### ➡️")
    with cols[2]:
        st.markdown("### 🎞️\n**Extract Frame**\nOpenCV")
    with cols[3]:
        st.markdown("### ➡️")
    with cols[4]:
        st.markdown("### 🤖\n**Ollama Vision**\n:blue[Intel GPU]")
    with cols[5]:
        st.markdown("### ➡️")
    with cols[6]:
        st.markdown("### 📝\n**Description**\nText output")
    
    st.markdown("")
    
    cols2 = st.columns(5)
    with cols2[0]:
        st.markdown("### 📝\n**Description**")
    with cols2[1]:
        st.markdown("### ➡️")
    with cols2[2]:
        st.markdown("### 🔢\n**Embeddings**\n:blue[Transformers]")
    with cols2[3]:
        st.markdown("### ➡️")
    with cols2[4]:
        st.markdown("### 💾\n**ChromaDB**\nVector store")
    
    st.markdown("---")
    
    st.info("**Search Pipeline** - How queries find relevant videos")
    
    cols3 = st.columns(7)
    with cols3[0]:
        st.markdown("### 👤\n**User Query**\nNatural text")
    with cols3[1]:
        st.markdown("### ➡️")
    with cols3[2]:
        st.markdown("### 🔢\n**Embedding**\nVector form")
    with cols3[3]:
        st.markdown("### ➡️")
    with cols3[4]:
        st.markdown("### 🔍\n**Similarity**\n:blue[Cosine dist]")
    with cols3[5]:
        st.markdown("### ➡️")
    with cols3[6]:
        st.markdown("### 🎬\n**Results**\nMatched videos")
    
    st.markdown("---")
    
    # Architecture diagram
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Key Components")
        st.markdown("""
        **1. Video Processing**
        - Frame extraction with OpenCV
        - Vision model inference on Intel GPU
        - Description generation
        
        **2. Embedding Generation**
        - Sentence Transformers
        - 384-dimensional vectors
        - Semantic representation
        
        **3. Vector Storage**
        - ChromaDB persistent storage
        - Cosine similarity metric
        - Efficient retrieval
        """)
    
    with col2:
        st.markdown("### 🚀 Intel Optimizations")
        st.markdown("""
        **Hardware Acceleration**
        - Intel Arc™ Graphics
        - Intel Core™ Ultra Processors
        - Intel Iris® Xe Graphics
        
        **Performance Benefits**
        - Fast inference times
        - Efficient memory usage
        - Local processing
        
        **Ollama Integration**
        - Optimized for Intel hardware
        - Easy model management
        - Production-ready
        """)
    
    st.markdown("---")
    
    # Technical details
    st.markdown("### 🔧 Technical Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **AI Models**
        - Qwen 2.5 VL
        - Llama 3.2 Vision
        - MiniLM Embeddings
        """)
    
    with col2:
        st.markdown("""
        **Infrastructure**
        - Ollama Runtime
        - ChromaDB
        - Streamlit UI
        """)
    
    with col3:
        st.markdown("""
        **Intel Hardware**
        - Arc™ GPUs
        - Core™ Ultra CPUs
        - Iris® Xe Graphics
        """)

with tab4:
    st.header("About This Demo")
    
    st.markdown("""
    ### 🎯 Video RAG with Ollama on Intel GPUs
    
    This application demonstrates **semantic video search** powered by **Ollama on Intel GPUs**. 
    It combines vision-language models, vector embeddings, and similarity search to enable 
    natural language queries over video content and detailed image understanding.
    
    ### 💡 Use Cases
    
    - **Video Libraries**: Quickly find specific content in large video collections
    - **Content Management**: Search videos by describing what you're looking for
    - **Surveillance**: Locate specific events or activities in footage
    - **Education**: Find relevant video segments for learning materials
    - **Media Production**: Search stock footage by description
    
    ### 🏗️ How It Works
    
    1. **Extract**: Take representative frames from videos
    2. **Describe**: Use AI vision models to generate detailed descriptions
    3. **Embed**: Convert descriptions to semantic vectors
    4. **Store**: Save vectors in a searchable database
    5. **Query**: Search using natural language
    6. **Retrieve**: Find most similar videos by semantic meaning
    
    ### 🚀 Why Intel GPUs?
    
    - **Performance**: Fast inference for vision models
    - **Efficiency**: Optimized power consumption
    - **Accessibility**: Available on mainstream devices
    - **Local Processing**: No cloud dependency
    - **Cost Effective**: Use existing hardware
    """)
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Technology Stack")
    
    st.code("""
    # Core Technologies
    - Ollama: Local LLM runtime optimized for Intel GPUs
    - Qwen 2.5 VL: State-of-the-art vision-language model
    - ChromaDB: Efficient vector database
    - Sentence Transformers: Text embedding generation
    - Streamlit: Interactive web interface
    
    # Intel Optimizations
    - Hardware acceleration on Arc and Iris Xe
    - Optimized inference on Core Ultra processors
    - Local processing without cloud dependency
    """, language="python")
    
    st.markdown("---")
    
    # System status
    st.markdown("### 📡 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            ollama.list()
            st.success("✅ Ollama Connected")
        except:
            st.error("❌ Ollama Offline")
    
    with col2:
        if os.path.exists(DATABASE_PATH):
            st.success(f"✅ Database Active")
        else:
            st.info("💾 Database Not Created")
    
    with col3:
        if st.session_state.database_loaded:
            st.success(f"✅ {st.session_state.collection.count()} Videos Indexed")
        else:
            st.info("⏳ Awaiting Processing")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666;">'
    '<p>🎥 <strong>Video RAG Demo</strong> | Powered by <strong>Ollama</strong> on '
    '<strong>Intel Arc Graphics</strong> and <strong>Intel Core Ultra Processors</strong></p>'
    '</div>',
    unsafe_allow_html=True
)
