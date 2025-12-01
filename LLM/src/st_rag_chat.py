from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community import document_loaders, embeddings, vectorstores, llms
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
import streamlit as st
import time
import os
import warnings
import ollama
import hashlib

warnings.filterwarnings("ignore")

OLLAMA_BASE_URL = "http://localhost:11434"

st.header("LLM RAG Chat Interface 🐻‍❄️💬")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_document" not in st.session_state:
    st.session_state.current_document = None

response = ollama.list()
models = [model.model for model in response.models]
model = st.selectbox("Choose a model from the list", models)

# Select source type
source_type = st.selectbox(
    "Select document source:",
    ("URL", "Local File from Data Folder"),
    key="source_type"
)

if source_type == "URL":
    # Input text to load the document from URL
    source_path = st.text_input("Enter the URL to load for RAG:", key="url_path")
else:
    # Show available files in data folder
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to LLM folder
    data_dir = os.path.join(current_dir, "data")
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith(('.txt', '.pdf', '.md'))]
        if files:
            st.write("Available files in data folder:")
            source_path = st.selectbox("Select a file:", files, key="file_selection")
            st.info(f"Selected file: {source_path}")
        else:
            st.warning("No supported files (.txt, .pdf, .md) found in data folder")
            source_path = None
    else:
        st.error("Data folder not found. Please create a 'data' folder in the parent directory and add your files.")
        source_path = None

# Select embedding type
embedding_type = st.selectbox(
    "Please select an embedding type",
    ("ollama",
     "huggingface",
     "nomic",
     "fastembed"),
    index=1)


def load_document(source_path, source_type="URL"):
    """
    Load the document from the specified URL or local file.

    Args:
        source_path (str): The URL or filename of the document to load.
        source_type (str): Either "URL" or "Local File from Data Folder"

    Returns:
        Document: The loaded document.
    """
    if source_type == "URL":
        print("Loading document from URL...")
        st.markdown(''' :green[Loading document from URL...] ''')
        loader = document_loaders.WebBaseLoader(source_path)
        return loader.load()
    else:
        # Load from local file in data folder
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to LLM folder
        data_dir = os.path.join(current_dir, "data")
        full_path = os.path.join(data_dir, source_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        
        print(f"Loading document from: {full_path}")
        st.markdown(f''' :green[Loading document from: {full_path}] ''')
        
        if source_path.endswith('.pdf'):
            loader = PyPDFLoader(full_path)
        else:
            loader = TextLoader(full_path, encoding='utf-8')
        
        return loader.load()


def split_document(text, chunk_size=3000, overlap=200):
    """
    Split the document into multiple chunks.

    Args:
        text (str): The text of the document to split.
        chunk_size (int): The size of each chunk.
        overlap (int): The overlap between chunks.

    Returns:
        list: A list of document chunks.
    """
    print("Splitting document into chunks...")
    st.markdown(''' :green[Splitting document into chunks...] ''')
    text_splitter_instance = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter_instance.split_documents(text)


def initialize_embedding_fn(
        embedding_type="huggingface",
        model_name="sentence-transformers/all-MiniLM-l6-v2"):
    """
    Initialize the embedding function based on the specified type.

    Args:
        embedding_type (str): The type of embedding to use.
        model_name (str): The name of the model to use for embeddings.

    Returns:
        Embeddings: The initialized embedding function.
    """
    print(f"Initializing {embedding_type} model with {model_name}...")
    st.write(f"Initializing {embedding_type} model with {model_name}...")
    if embedding_type == "ollama":
        model_name = chat_model
        return embeddings.OllamaEmbeddings(
            model=model_name, base_url=OLLAMA_BASE_URL)
    elif embedding_type == "huggingface":
        model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
        return embeddings.HuggingFaceEmbeddings(model_name=model_name)
    elif embedding_type == "nomic":
        return embeddings.NomicEmbeddings(model_name=model_name)
    elif embedding_type == "fastembed":
        return FastEmbedEmbeddings(threads=16)
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")


def get_or_create_embeddings(document_url, source_type, embedding_fn):
    """
    Create embeddings for the document chunks and store them in a vector database.
    Uses persistent storage with improved caching that considers embedding type.

    Args:
        document_url (str): The URL of the document.
        source_type (str): The type of source (URL or local file).
        embedding_fn (Embeddings): The embedding function to use.

    Returns:
        VectorStore: The created or loaded vector store.
    """
    # Create a more specific hash for caching that includes embedding type
    embedding_type_name = embedding_fn.__class__.__name__
    cache_key = f"{document_url}_{source_type}_{embedding_type_name}"
    source_hash = hashlib.md5(cache_key.encode()).hexdigest()
    persist_directory = f"./chroma_db/{source_hash}"
    
    print(f"Cache key: {cache_key}")
    print(f"Cache directory: {persist_directory}")
    
    # Check if embeddings already exist and are for the same document
    if os.path.exists(persist_directory):
        # Check if there's a metadata file to verify this is the right document
        metadata_file = os.path.join(persist_directory, "document_info.txt")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                cached_info = f.read().strip()
                if cached_info == cache_key:
                    print(f"Loading existing embeddings from: {persist_directory}")
                    st.markdown(f''' :orange[Loading cached embeddings for: {source_type}: {document_url}] ''')
                    try:
                        vector_store = vectorstores.Chroma(
                            persist_directory=persist_directory,
                            embedding_function=embedding_fn
                        )
                        return vector_store
                    except Exception as e:
                        print(f"Error loading cached embeddings: {e}")
                        st.warning("Error loading cached embeddings, creating new ones...")
                else:
                    print(f"Cache mismatch. Expected: {cache_key}, Found: {cached_info}")
                    st.warning("Cache mismatch detected, creating new embeddings...")
        else:
            print("No metadata file found, creating new embeddings...")
            st.warning("Cache metadata missing, creating new embeddings...")
    
    # Create new embeddings
    start_time = time.time()
    print(f"Creating new embeddings for: {document_url}")
    st.markdown(f''' :green[Creating new embeddings for: {source_type}: {document_url}] ''')
    
    # Clean up the directory if it exists but has issues
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
    
    document = load_document(document_url, source_type)
    documents = split_document(document)
    
    vector_store = vectorstores.Chroma.from_documents(
        documents=documents,
        embedding=embedding_fn,
        persist_directory=persist_directory
    )
    
    # Save metadata to verify cache validity
    os.makedirs(persist_directory, exist_ok=True)
    metadata_file = os.path.join(persist_directory, "document_info.txt")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(cache_key)
    
    print(f"Embedding time: {time.time() - start_time:.2f} seconds")
    st.write(f"Embedding time: {time.time() - start_time:.2f} seconds")
    return vector_store


def get_chat_context():
    """
    Build context from previous messages for continuity
    """
    if not st.session_state.messages:
        return ""
    
    # Get last few messages for context (limit to avoid token overflow)
    recent_messages = st.session_state.messages[-6:]  # Last 3 Q&A pairs
    context_parts = []
    
    for msg in recent_messages:
        if msg["role"] == "user":
            context_parts.append(f"Previous Question: {msg['content']}")
        else:
            context_parts.append(f"Previous Answer: {msg['content']}")
    
    return "\n".join(context_parts)


def handle_chat_query(vector_store, chat_model, question):
    """
    Handle chat query with context awareness
    """
    # Get conversation context
    chat_context = get_chat_context()
    
    # Create enhanced question with chat context if available
    if chat_context:
        enhanced_question = f"""
Previous conversation context:
{chat_context}

Current question: {question}
"""
    else:
        enhanced_question = question
    
    # Simple prompt template that only uses context and question
    prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you do not know the answer, answer 'I don't know', limit your response to the answer and nothing more.

    {context}

    Question: {question}
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain_type_kwargs = {"prompt": prompt}
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    qachain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs=chain_type_kwargs
    )
    
    start_time = time.time()
    answer = qachain.invoke({"query": enhanced_question})
    print(f"Response time: {time.time() - start_time:.2f} seconds")
    
    return answer['result']


# Load document section
st.write("### 📄 Document Loading")
load_button = st.button("Load Document", type="primary")

if load_button:
    if not source_path or not source_path.strip():
        st.error("Please select/enter a valid source.")
    else:
        with st.spinner("Loading document and creating embeddings..."):
            try:
                embedding_fn = initialize_embedding_fn(embedding_type)
                vector_store = get_or_create_embeddings(source_path, source_type, embedding_fn)
                
                # Proper warmup: initialize the model and retriever chain
                st.markdown(''' :green[Initializing RAG system...] ''')
                chat_model_instance = llms.Ollama(base_url=OLLAMA_BASE_URL, model=model)
                
                # Warmup the retriever
                retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                warmup_docs = retriever.get_relevant_documents("document content summary")
                
                # Warmup the QA chain with actual query
                prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you do not know the answer, answer 'I don't know', limit your response to the answer and nothing more.

    {context}

    Question: {question}
    """
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                chain_type_kwargs = {"prompt": prompt}
                
                warmup_chain = RetrievalQA.from_chain_type(
                    llm=chat_model_instance,
                    retriever=retriever,
                    chain_type="stuff",
                    chain_type_kwargs=chain_type_kwargs
                )
                
                # Execute warmup query
                warmup_chain.invoke({"query": "What is this document about?"})
                
                st.session_state.vector_store = vector_store
                st.session_state.current_document = f"{source_type}: {source_path}"
                
                # Clear previous chat and add initial summary request
                st.session_state.messages = []
                st.session_state.messages.append({
                    "role": "user", 
                    "content": "Summarize this document"
                })
                
                # Generate the summary automatically
                st.markdown(''' :green[Generating document summary...] ''')
                summary = handle_chat_query(
                    st.session_state.vector_store,
                    chat_model_instance,
                    "Summarize this document"
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": summary
                })
                
                st.success(f"✅ Document loaded successfully!")
                st.info(f"Loaded: {st.session_state.current_document}")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading document: {e}")

# Display current document status
if st.session_state.current_document:
    st.write(f"📄 **Current Document**: {st.session_state.current_document}")
else:
    st.warning("⚠️ No document loaded. Please load a document first.")

# Chat Interface
st.write("### 💬 Chat Interface")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("Ask a question about the document..."):
    if not st.session_state.vector_store:
        st.error("Please load a document first!")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chat_model_instance = llms.Ollama(
                        base_url=OLLAMA_BASE_URL, model=model)
                    response = handle_chat_query(
                        st.session_state.vector_store, 
                        chat_model_instance, 
                        question
                    )
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button
if st.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Display chat statistics
if st.session_state.messages:
    st.write(f"💬 **Chat History**: {len(st.session_state.messages)} messages")