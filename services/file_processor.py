import os
import uuid
import faiss
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
#from langchain_community.embeddings import HuggingFaceEmbeddings

class PDFProccessor():
    def __init__(self):
        self.model_name = "cointegrated/rubert-tiny2"
        #self.raw_data = []
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)


        index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def process_pdf(self, file_path) -> list:
        print(file_path)
        raw_data = []
        ids = []
        doc_id = str(uuid.uuid4())
        # Извлечение текста
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        
        # Разделение на чанки
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        
        # Добавление в векторную БД
        for chunk in chunks:
            #vector_store.add_texts([chunk], metadatas=[{"doc_id": doc_id}])
            raw_data.append(Document(
                page_content=chunk,
                metadata={"source": file_path,
                          "doc_id": doc_id}
            ))
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)

        self.vector_store.add_documents(documents=raw_data, ids=ids)
        return ids
        