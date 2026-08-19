# 🎓 College Admission Assistant

An AI-powered College Admission RAG Chatbot that provides context-aware answers to admission-related questions using Retrieval-Augmented Generation (RAG).

## 🚀 Features

- 🤖 AI-powered admission assistant
- 🔎 Semantic document search
- 📚 Retrieval-Augmented Generation (RAG)
- 🧠 Hugging Face embeddings
- 🗄️ ChromaDB vector database
- 🔄 Document chunking and indexing
- 🎯 Reranking for improved retrieval
- 💬 ChatGPT-style Streamlit interface
- 📖 Source/context display
- 📊 Retrieval evaluation
- 🔐 Environment variable support for API secrets

## 🏗️ Architecture

```text
College Admission Documents
            ↓
      Text Loader
            ↓
    Text Chunking
            ↓
     Embeddings
            ↓
       ChromaDB
            ↓
     Query Retrieval
            ↓
       Reranking
            ↓
     RAG Generation
            ↓
     Streamlit Chat UI

     📁 Project Structure

     college_admission_assistant/
│
├── data/
│   └── raw/
│       └── college_admission.txt
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
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md

🛠️ Technologies Used
Python
LangChain
Hugging Face
ChromaDB
Streamlit
Sentence Transformers
RAG
Vector Embeddings
Semantic Search
⚙️ Installation

Clone the repository:

git clone https://github.com/srija125/College_Admission_Assistant.git
cd College_Admission_Assistant

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
📌 Create Vector Database

Run the ingestion pipeline:

python src/ingestion.py

This loads the college admission document, creates chunks, generates embeddings, and stores them in ChromaDB.

▶️ Run the Application

Start the Streamlit application:

streamlit run streamlit_app.py

The application will open in your browser.

🧪 Evaluation

The project includes an evaluation module to test retrieval performance using predefined admission-related questions.

python evaluation/evaluation.py
🔐 Environment Variables

Store API keys and secrets in environment variables or Streamlit secrets.

Example:

.env
.streamlit/secrets.toml

Do not upload API keys or passwords to GitHub.

🔮 Future Improvements
Multi-college support
College comparison
Admission deadline tracking
Course recommendation
Scholarship information
Multilingual chatbot
Voice-based interaction
Cloud deployment
Authentication and user profiles
🎯 Use Cases

The same RAG architecture can be adapted for:

Healthcare document assistants
Banking and financial assistants
Legal document assistants
Education assistants
HR assistants
Customer support systems

Only the domain-specific documents, prompts, and business logic need to be changed.

👩‍💻 Author

Srija Tanti

B.Tech CSE (AI & ML)

⭐ If you find this project useful, consider giving it a star!