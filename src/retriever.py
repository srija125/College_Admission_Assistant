from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever


# ==========================================
# 1. Project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DB_DIR = BASE_DIR / "chroma_db"
DATA_FILE = BASE_DIR / "data" /"raw"/ "college_admission.txt"


# ==========================================
# 2. Load embedding model
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# 3. Load ChromaDB
# ==========================================

vectorstore = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings,
    collection_name="college_admission"
)


# ==========================================
# 4. Create Semantic Retriever
# ==========================================

semantic_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# ==========================================
# 5. Load documents for BM25
# ==========================================

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = TextLoader(
    str(DATA_FILE),
    encoding="utf-8"
)

documents = loader.load()


# Use the same chunking configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)


# ==========================================
# 6. Create BM25 Retriever
# ==========================================

bm25_retriever = BM25Retriever.from_documents(
    chunks
)

bm25_retriever.k = 4


# ==========================================
# 7. Hybrid Retrieval Function
# ==========================================

def hybrid_search(query, k=4):

    semantic_docs = semantic_retriever.invoke(query)

    keyword_docs = bm25_retriever.invoke(query)

    # Combine results
    combined = semantic_docs + keyword_docs

    # Remove duplicate chunks
    unique_docs = []

    seen = set()

    for doc in combined:

        content = doc.page_content

        if content not in seen:
            unique_docs.append(doc)
            seen.add(content)

    return unique_docs[:k]


# ==========================================
# 8. Test Retriever
# ==========================================

if __name__ == "__main__":

    query = input("\nEnter your question: ")

    results = hybrid_search(query)

    print("\n==============================")
    print("HYBRID RETRIEVAL RESULTS")
    print("==============================")

    for i, doc in enumerate(results, start=1):

        print(f"\n--- Result {i} ---")
        print(doc.page_content)