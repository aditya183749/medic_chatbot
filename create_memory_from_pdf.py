from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# 1. Load PDF
def load_documents():
    loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

# 2. Split into Chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# 3. Convert to Embeddings + Store in FAISS
def store_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("vectorstore")

if __name__ == "__main__":
    print("🚀 Loading and processing documents...")
    docs = load_documents()
    chunks = split_documents(docs)
    store_vector_db(chunks)
    print("✅ FAISS vector store created!")

