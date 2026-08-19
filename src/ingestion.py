from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ==========================================
# 1. Project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "raw" / "college_admission.txt"
VECTOR_DB_DIR = BASE_DIR / "chroma_db"


# ==========================================
# 2. Load college admission document
# ==========================================

print("Loading college admission document...")

loader = TextLoader(
    str(DATA_FILE),
    encoding="utf-8"
)

documents = loader.load()

print(f"Loaded documents: {len(documents)}")


# ==========================================
# 3. Split document into chunks
# ==========================================

print("Splitting document into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# ==========================================
# 4. Create embeddings
# ==========================================

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# 5. Store embeddings in ChromaDB
# ==========================================

print("Creating ChromaDB vector store...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(VECTOR_DB_DIR),
    collection_name="college_admission"
)


print("===================================")
print("Ingestion completed successfully!")
print(f"Chunks created: {len(chunks)}")
print(f"Vector database: {VECTOR_DB_DIR}")
print("===================================")