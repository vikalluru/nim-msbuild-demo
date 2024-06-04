import os
from PyPDF2 import PdfReader
from typing import List, Dict
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from .nv_embedding_client import NVEmbeddings

config = {
    "openai_model": "text-embedding-3-large",
    "nv_model": "NV-Embed-QA-Mistral-7B",
    "maximum_chunk_size": 800,
    "chunk_overlap": 100,
    "nv_embed_url": "http://51.124.97.12:8080/v1/embeddings",
    "chunks_to_retrieve": 1
}

def chunk_text(pdf_text) -> List[str]:
    text_splitter = CharacterTextSplitter(
        separator="\n",  # Split on newlines
        chunk_size=config["maximum_chunk_size"],  # Maximum chunk size
        chunk_overlap=config["chunk_overlap"],  # Overlap between chunks
    )
    chunks = text_splitter.split_text(pdf_text)
    return chunks

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_db(texts, embed_tool) -> FAISS:
    text_embeddings = embed_tool.embed_documents(texts)
    text_embeddg_pairs = zip(texts, text_embeddings)
    return FAISS.from_embeddings(text_embeddg_pairs, embed_tool)

def convert_query_to_embeddings(query, embed_tool) -> List[float]:
    return embed_tool.embed_query(query)

def retrieve_context(db, query_embedded) -> str:
    docs = db.similarity_search_by_vector(query_embedded, k=config["chunks_to_retrieve"])
    return "\n".join([f"{doc.page_content[:300]}" for doc in docs])

def get_embedding_type(nim_on=False):
    if nim_on:
        return NVEmbeddings(api_url=config["nv_embed_url"], model=config["nv_model"])
    return OpenAIEmbeddings(model="text-embedding-3-small")

def save_db(pdf_path, nim_on=False) -> bool:
    try:
        pdf_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(pdf_text)
        faiss_db = generate_db(chunks, get_embedding_type(nim_on))
        prefix = "nim_on" if nim_on else "nim_off"
        faiss_db.save_local(f"{prefix}_db")
    except Exception as e:
        # This block will execute if any other exception is raised
        print(f"An unexpected error occurred: {e}")
        return False
    return True

def get_context(user_prompt, nim_on=False) -> str:
    prefix = "nim_on" if nim_on else "nim_off"    
    faiss_db = FAISS.load_local(f"{prefix}_db", get_embedding_type(nim_on), allow_dangerous_deserialization=True)
    query_embedding = convert_query_to_embeddings(user_prompt, get_embedding_type(nim_on))
    context = retrieve_context(faiss_db, query_embedding)
    return context