import json
import sys
import uuid
from pathlib import Path

import streamlit as st


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
HISTORY_DIR = BASE_DIR / "chat_history"

HISTORY_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from rag_chain import generate_answer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="College Admission Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.25);
    }

    .app-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-top: 25px;
    }

    .app-subtitle {
        text-align: center;
        opacity: 0.65;
        margin-bottom: 30px;
    }

    .chat-title {
        font-weight: 600;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHAT FILE FUNCTIONS
# ============================================================

def chat_file(chat_id):
    return HISTORY_DIR / f"{chat_id}.json"


def save_chat(chat):
    with open(
        chat_file(chat["id"]),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            chat,
            file,
            ensure_ascii=False,
            indent=2
        )


def load_chat(chat_id):

    path = chat_file(chat_id)

    if not path.exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_all_chats():

    chats = []

    for path in HISTORY_DIR.glob("*.json"):

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:

                chats.append(
                    json.load(file)
                )

        except Exception:
            pass

    chats.sort(
        key=lambda x: x.get("updated_at", ""),
        reverse=True
    )

    return chats


def delete_chat(chat_id):

    path = chat_file(chat_id)

    if path.exists():
        path.unlink()


# ============================================================
# SESSION STATE
# ============================================================

if "chats" not in st.session_state:

    st.session_state.chats = load_all_chats()


if "current_chat_id" not in st.session_state:

    if st.session_state.chats:

        st.session_state.current_chat_id = (
            st.session_state.chats[0]["id"]
        )

    else:

        st.session_state.current_chat_id = None


# ============================================================
# CREATE NEW CHAT
# ============================================================

def create_new_chat():

    chat_id = str(uuid.uuid4())

    chat = {
        "id": chat_id,
        "title": "New Chat",
        "messages": [],
        "updated_at": str(uuid.uuid1().time)
    }

    save_chat(chat)

    st.session_state.chats = load_all_chats()

    st.session_state.current_chat_id = chat_id


# ============================================================
# GET CURRENT CHAT
# ============================================================

def get_current_chat():

    chat_id = st.session_state.current_chat_id

    if not chat_id:
        return None

    return load_chat(chat_id)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 College Assistant")

    st.caption(
        "AI-powered College Admission RAG Chatbot"
    )

    st.divider()


    # New Chat
    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        create_new_chat()

        st.rerun()


    st.divider()


    # Chat history
    st.markdown("### 💬 Chats")

    chats = load_all_chats()

    if not chats:

        st.caption("No conversations yet.")

    else:

        for chat in chats:

            title = chat.get(
                "title",
                "New Chat"
            )

            if len(title) > 28:
                title = title[:28] + "..."

            col1, col2 = st.columns(
                [5, 1]
            )


            with col1:

                if st.button(
                    f"💬 {title}",
                    key=f"open_{chat['id']}",
                    use_container_width=True
                ):

                    st.session_state.current_chat_id = (
                        chat["id"]
                    )

                    st.rerun()


            with col2:

                if st.button(
                    "⋮",
                    key=f"menu_{chat['id']}"
                ):

                    st.session_state[
                        f"show_menu_{chat['id']}"
                    ] = not st.session_state.get(
                        f"show_menu_{chat['id']}",
                        False
                    )


            # Chat options
            if st.session_state.get(
                f"show_menu_{chat['id']}",
                False
            ):

                new_title = st.text_input(
                    "Rename",
                    value=chat.get(
                        "title",
                        "New Chat"
                    ),
                    key=f"rename_{chat['id']}"
                )

                if st.button(
                    "Rename",
                    key=f"rename_button_{chat['id']}"
                ):

                    chat["title"] = new_title

                    save_chat(chat)

                    st.rerun()


                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{chat['id']}"
                ):

                    delete_chat(
                        chat["id"]
                    )

                    remaining = load_all_chats()

                    st.session_state.chats = remaining

                    if remaining:

                        st.session_state.current_chat_id = (
                            remaining[0]["id"]
                        )

                    else:

                        st.session_state.current_chat_id = None

                    st.rerun()


    st.divider()


    # Settings
    with st.expander("⚙️ Settings"):

        st.write("Model: Ollama")

        st.write("Framework: LangChain")

        st.write(
            "Retrieval: ChromaDB + BM25"
        )

        st.write(
            "Reranker: Cross-Encoder"
        )


    # About
    with st.expander("ℹ️ About"):

        st.write(
            """
            College Admission Assistant uses
            Retrieval-Augmented Generation (RAG)
            to answer questions using college
            admission documents.
            """
        )


# ============================================================
# MAIN AREA
# ============================================================

current_chat = get_current_chat()


if current_chat is None:

    st.markdown(
        """
        <div class="app-title">
            🎓 College Admission Assistant
        </div>

        <div class="app-subtitle">
            Ask questions about college admission
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### How can I help you?"
    )

    st.info(
        "Click **＋ New Chat** in the sidebar to start."
    )


else:

    # --------------------------------------------------------
    # Chat title
    # --------------------------------------------------------

    st.markdown(
        f"### {current_chat['title']}"
    )


    # --------------------------------------------------------
    # Messages
    # --------------------------------------------------------

    for message in current_chat["messages"]:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                with st.expander(
                    "📚 Sources"
                ):

                    for source in message["sources"]:

                        st.write(
                            f"• {source}"
                        )


    # --------------------------------------------------------
    # Chat input
    # --------------------------------------------------------

    question = st.chat_input(
        "Message College Admission Assistant..."
    )


    if question:

        # User message
        user_message = {
            "role": "user",
            "content": question
        }

        current_chat["messages"].append(
            user_message
        )


        # Automatically title chat
        if current_chat["title"] == "New Chat":

            current_chat["title"] = (
                question[:30]
            )

            if len(question) > 30:
                current_chat["title"] += "..."


        save_chat(current_chat)


        # Display user
        with st.chat_message("user"):

            st.markdown(question)


        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching admission information..."
            ):

                try:

                    result = generate_answer(
                        question
                    )

                    answer = result["answer"]

                    sources = result["sources"]


                    st.markdown(answer)


                    if sources:

                        with st.expander(
                            "📚 Sources"
                        ):

                            for source in sources:

                                st.write(
                                    f"• {source}"
                                )


                    assistant_message = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }


                    current_chat["messages"].append(
                        assistant_message
                    )

                    save_chat(current_chat)


                except Exception as error:

                    error_message = (
                        "Sorry, something went wrong.\n\n"
                        f"`{error}`"
                    )

                    st.error(
                        error_message
                    )

                    current_chat["messages"].append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": []
                        }
                    )

                    save_chat(current_chat)