from sentence_transformers import CrossEncoder

from retriever import hybrid_search


# ==========================================
# 1. Load Cross-Encoder Reranker
# ==========================================

print("Loading reranker model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker loaded successfully!")


# ==========================================
# 2. Rerank retrieved documents
# ==========================================

def rerank_documents(query, documents, top_k=3):

    if not documents:
        return []

    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_documents[:top_k]


# ==========================================
# 3. Complete retrieval + reranking
# ==========================================

def retrieve_and_rerank(query, top_k=3):

    # First: Hybrid retrieval
    documents = hybrid_search(query, k=6)

    # Second: Reranking
    ranked_documents = rerank_documents(
        query,
        documents,
        top_k=top_k
    )

    return ranked_documents


# ==========================================
# 4. Test
# ==========================================

if __name__ == "__main__":

    query = input("\nEnter your question: ")

    results = retrieve_and_rerank(query)

    print("\n================================")
    print("RERANKED RESULTS")
    print("================================")

    for i, (document, score) in enumerate(
        results,
        start=1
    ):

        print(f"\n--- Result {i} ---")
        print(f"Score: {score:.4f}")
        print(document.page_content)