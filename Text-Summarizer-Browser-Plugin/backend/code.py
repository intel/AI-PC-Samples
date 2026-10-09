# Importing necessary libraries
import ipaddress
import socket
from urllib.parse import urlparse

from transformers import AutoTokenizer, pipeline
from optimum.intel import OVModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

# Prompt Templates for Summarization & QA Bot
summary_template = """Write a concise summary of the following: "{context}" CONCISE SUMMARY: """
query_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use 10 words maximum and keep the answer as concise as possible in one sentence.
    Always say "thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""


def _validate_external_url(url: str) -> str:
    """
    Validate a user-supplied URL before the server fetches it, to mitigate
    Server-Side Request Forgery (SSRF). Only public http/https endpoints are
    allowed; requests to localhost, link-local, private, and other
    non-routable addresses are rejected so the backend cannot be tricked into
    reaching internal services (e.g. cloud metadata endpoints, LAN hosts).
    """
    parsed = urlparse((url or "").strip())

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs are allowed.")

    if not parsed.hostname:
        raise ValueError("URL must include a valid hostname.")

    host = parsed.hostname.strip().lower()
    if host == "localhost" or host.endswith(".local"):
        raise ValueError("Local/private hosts are not allowed.")

    try:
        ip_obj = ipaddress.ip_address(host)
        if not ip_obj.is_global or ip_obj.is_multicast:
            raise ValueError("Private or non-routable IP addresses are not allowed.")
    except ValueError:
        # Not an IP literal; resolve hostname and ensure all resolved IPs are public.
        try:
            addrinfos = socket.getaddrinfo(host, None)
        except socket.gaierror:
            raise ValueError("Unable to resolve URL hostname.")

        resolved_ips = {ai[4][0] for ai in addrinfos if ai and ai[4]}
        if not resolved_ips:
            raise ValueError("Unable to resolve URL hostname.")

        for ip in resolved_ips:
            ip_obj = ipaddress.ip_address(ip)
            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
                or ip_obj.is_reserved
                or ip_obj.is_unspecified
            ):
                raise ValueError("URL host resolves to a private/non-routable address.")

    return url


def pre_processing(loader):
    """
        This is a helper function which does the below steps in a sequential order:
        1. Loads page content from the URL/PDF
        2. Splits the page data using Recursive Character Text Splitter & creates embeddings using HuggingFace Embeddings
        3. This is further stored into ChromaDB then after for retrieval
        input: Fetched RAW content from the input(URL/PDF).
        output: returns a vectorstore
    """
    try:
        page_data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20)
        all_splits = text_splitter.split_documents(page_data)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        global vectorstore
        vectorstore = Chroma.from_documents(
            documents=all_splits, embedding=embeddings)
        return vectorstore
    except Exception as e:
        print("Error while processing Webpage/PDF page content\n")
        raise e


def load_llm(model_id):
    """
        Meta Llama2 & Qwen 7B models are converted to OpenVINO IR Format. This function compiles those converted models on GPU.
        input: user selected model_id from plugin
        output: compiled model with openvino
    """
    if model_id:
        try:
            if model_id == "Meta LLama 2":
                model_path = '../models/ov_llama_2'
            elif model_id == "Qwen 7B Instruct":
                model_path = '../models/ov_qwen7b'
            model = OVModelForCausalLM.from_pretrained(
                model_path, device='GPU')
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=4096,
                device=model.device
            )
            global llm_model
            llm_model = HuggingFacePipeline(pipeline=pipe)
            return llm_model
        except Exception as e:
            print(
                "Failed to load the model. Please check whether the model_path is correct.")
            raise e


def pre_process_url_data(urls):
    """
        When an end user pastes a URL into the plugin, The RAW data is passed onto the RetrievalQA chain,
        and the output is returned back to the plugin.
        input: Webpage URL(str).
        output: Glance Summary of the fetched URL.
    """
    try:
        validated_urls = [_validate_external_url(u) for u in urls]
        loader = WebBaseLoader(validated_urls)
        global summ_vectorstore
        # Common Helper function for processing data.
        summ_vectorstore = pre_processing(loader)
        prompt = PromptTemplate(
            template=summary_template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            retriever=summ_vectorstore.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False,
        )

        question = "Please summarize the entire book in one paragraph of 100 words"
        summary = qa_chain(question)
        response = summary['result']
        summary_start = response.find("CONCISE SUMMARY:")
        concise_summary = response[summary_start +
                                   len("CONCISE SUMMARY:"):].strip()
        return concise_summary
    except Exception as e:
        print("Failed to summarize webpage \n")
        raise e


def qa_on_url_summarized_text(query):
    """
        This function fetches the query asked by the users post summarization from the URL, searches an answer from the vectorstore & returns answer in less than 10 words.
        input: user's follow-up question(str)
        output: Answer to the conversations.
    """
    try:
        prompt = PromptTemplate(
            template=query_template,
            input_variables=["context", "question"]
        )
        reduce_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            retriever=summ_vectorstore.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )
        summary = reduce_chain({'query': query})
        summ_vectorstore.delete
        response = summary['result']
        summary_start = response.find("Helpful Answer:")
        concise_summary = response[summary_start +
                                   len("Helpful Answer:"):].strip()
        return concise_summary
    except Exception as e:
        print("Error in Webpage Summarizer QA BoT")
        raise e


def pre_process_pdf_data(pdf):
    """
        When an end user uploads a PDF into the plugin, The RAW data is passed onto the RetrievalQA chain,
        and the output is returned back to the plugin.
        input: PDF path(str).
        output: Glance Summary of the uploaded PDF.
    """
    try:
        loader = PyPDFLoader(pdf, extract_images=False)
        global pdf_vectorstore
        pdf_vectorstore = pre_processing(loader)

        prompt = PromptTemplate(
            template=summary_template,
            input_variables=["context", "question"]
        )
        reduce_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            retriever=pdf_vectorstore.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False,
        )
        question = "Please summarize the entire book in 100 words."
        summary = reduce_chain({'query': question})

        response = summary['result']
        summary_start = response.find("CONCISE SUMMARY:")
        concise_summary = response[summary_start +
                                   len("CONCISE SUMMARY:"):].strip()
        return concise_summary
    except Exception as e:
        print("Failed to summarize PDF \n")
        raise e


def qa_on_pdf_summarized_text(query):
    """
        This function fetches the query asked by the users post summarization from the PDF, then after it searches an answer from the vectorstore & returns answer in less than 10 words.
        input: user's follow-up question(str)
        output: Answer to the conversations.
    """
    try:
        prompt = PromptTemplate(
            template=query_template,
            input_variables=["context", "question"]
        )
        reduce_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            retriever=pdf_vectorstore.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )
        summary = reduce_chain({'query': query})
        response = summary['result']
        summary_start = response.find("Helpful Answer:")
        concise_summary = response[summary_start +
                                   len("Helpful Answer:"):].strip()
        return concise_summary
    except Exception as e:
        print("Error in PDF Summarizer QA BoT")
        raise e
