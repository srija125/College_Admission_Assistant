from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from reranker import retrieve_and_rerank


# ==========================================
# 1. Ollama model
# ==========================================

OLLAMA_MODEL = "llama3.2"

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)


# ==========================================
# 2. RAG Prompt
# ==========================================

prompt = ChatPromptTemplate.from_template(
    """
You are a College Admission Assistant.

Answer the user's question ONLY using the provided context.

Rules:
1. Do not make up information.
2. If the answer is not present in the context,
   clearly say that the information is not available.
3. Give a clear and concise answer.
4. Use bullet points when appropriate.
5. Mention the relevant source when possible.

Context:
{context}

User Question:
{question}

Answer:
"""
)


# ==========================================
# 3. Generate RAG Answer
# ==========================================

def generate_answer(question):

    # Retrieve + rerank
    results = retrieve_and_rerank(
        question,
        top_k=3
    )

    if not results:
        return {
            "answer": "I could not find relevant information.",
            "sources": []
        }

    # Build context
    context_parts = []

    sources = []

    for document, score in results:

        context_parts.append(
            document.page_content
        )

        source = document.metadata.get(
            "source",
            "college_admission.txt"
        )

        if source not in sources:
            sources.append(source)

    context = "\n\n---\n\n".join(
        context_parts
    )

    # Create prompt
    messages = prompt.format_messages(
        context=context,
        question=question
    )

    # Generate answer
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": sources
    }


# ==========================================
# 4. Test RAG Chatbot
# ==========================================

if __name__ == "__main__":

    question = input(
        "\nEnter your question: "
    )

    result = generate_answer(question)

    print("\n================================")
    print("COLLEGE ADMISSION RAG ANSWER")
    print("================================")

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")

    for source in result["sources"]:
        print(f"- {source}")