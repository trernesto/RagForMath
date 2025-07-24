import os
import uuid

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
#from langchain_community.embeddings import HuggingFaceEmbeddings

# Глобальные переменные для хранения состояния
vector_store = None
current_doc_id = None

def init_vector_store():
    """Инициализация векторного хранилища"""
    global vector_store
    if vector_store is None:
        model = SentenceTransformer("cointegrated/rubert-tiny2")
        embeddings = model.encode([''])
        vector_store = FAISS.from_texts([""], embeddings)
    return vector_store

def process_pdf(file_path):
    """Обработка PDF файла"""
    global vector_store, current_doc_id
    
    # Инициализация хранилища
    vector_store = init_vector_store()
    
    # Извлечение текста
    with pdfplumber.open(file_path) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    
    # Разделение на чанки
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    
    # Генерация ID документа
    doc_id = str(uuid.uuid4())
    current_doc_id = doc_id
    
    # Добавление в векторную БД
    for chunk in chunks:
        vector_store.add_texts([chunk], metadatas=[{"doc_id": doc_id}])
    
    return doc_id