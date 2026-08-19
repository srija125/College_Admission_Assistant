# 🎓 College Admission RAG Chatbot

## 1. Project Overview

The College Admission RAG Chatbot is an AI-powered question-answering
system designed to provide accurate information about college admission.

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant
information from college admission documents and generate grounded answers.

---

## 2. Objectives

- Provide admission-related information through a chatbot.
- Retrieve relevant information from college documents.
- Reduce hallucination by grounding answers in retrieved context.
- Use semantic and keyword-based retrieval.
- Improve retrieval using a reranking model.
- Provide an interactive Streamlit interface.

---

## 3. Technologies Used

- Python
- LangChain
- LangChain Community
- LangChain Ollama
- ChromaDB
- BM25
- Sentence Transformers
- Cross-Encoder
- Ollama
- Streamlit
- Scikit-learn
- NumPy
- Pandas

---

## 4. System Architecture

The system follows the following RAG pipeline:

User Question
        ↓
Hybrid Retrieval
        ↓
ChromaDB + BM25
        ↓
Cross-Encoder Reranking
        ↓
Top Relevant Documents
        ↓
LangChain Prompt
        ↓
Ollama LLM
        ↓
Final Answer

---

## 5. Project Structure

college_admission_assistant/

├── data/
│   └── college_admission.txt
│
├── src/
│   ├── ingestion.py
│   ├── retriever.py
│   ├── reranker.py
│   └── rag_chain.py
│
├── evaluation/
│   ├── evaluation.py
│   └── questions.json
│
├── chroma_db/
├── streamlit_app.py
├── requirements.txt
└── README.md

---

## 6. RAG Pipeline

### Document Ingestion

The admission document is loaded and divided into smaller chunks.

### Embedding Generation

The text chunks are converted into numerical vector representations.

### Vector Database

The embeddings are stored in ChromaDB for semantic similarity search.

### BM25 Retrieval

BM25 is used for keyword-based retrieval.

### Hybrid Retrieval

Semantic retrieval and BM25 retrieval are combined to improve search quality.

### Reranking

A Cross-Encoder model reranks the retrieved documents according to
their relevance to the user's question.

### Answer Generation

The top relevant documents are provided as context to an Ollama LLM
through LangChain.

The LLM generates the final answer using the retrieved context.

---

## 7. Installation

Create a virtual environment:

```bash
python -m venv venv