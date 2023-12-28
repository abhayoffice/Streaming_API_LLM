from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
import os

def create_faiss_db(documents, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # embeddings = HuggingFaceBgeEmbeddings(
    #     model_name="BAAI/bge-large-en",
    #     model_kwargs={'device': 'cpu'},
    # )

    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss")
    return db


def load_faiss_db(embeddings):
    db = FAISS.load_local("faiss", embeddings)
    return db