import streamlit as st
from chatbot_rag.RAG import RAG
from chatbot_rag.chat import *
from chatbot_rag.preprocessing import PyMuPDFPreprocessing, BasePreprocessing
import os
import shutil
import ollama
import time
import asyncio


class ChatApp:
    def __init__(self):
        st.set_page_config(page_title="Interactive Chatbot", layout="centered")
        st.title("Interactive Chatbot")
        with st.expander("Model configuration"):
            host = st.selectbox(label="Host", options=["Ollama", "Hugginface"], index=0)
            if host == "Ollama":
                model_name = st.selectbox(
                    "Model:", options=[model.model for model in ollama.list()["models"]]
                )

                if not hasattr(self, "name") or model_name != self.name:
                    self.name = model_name
                    self.load_bot(host, model_name=self.name)
            elif host == "Hugginface":
                model_name = st.text_input(
                    label="Model Name", value="deepseek-ai/DeepSeek-V3-0324"
                )
                token = st.text_input(label="Token", value="")
                provider = st.text_input(label="Provider", value="hyperbolic")
                st.button(
                    "Load Model",
                    on_click=self.load_bot,
                    args=(host, model_name, token, provider),
                )

        if "bot" not in st.session_state:
            st.session_state.bot = None
        if "rag" not in st.session_state:
            st.session_state.rag = None

        if os.path.exists("./data/"):
            shutil.rmtree("./data/")
        if os.path.exists("./info/"):
            shutil.rmtree("./info/")

        self.sidebar_options()

        self.init_session()

        self.retry_done = False  # to handle timeout retries

    async def call_bot_with_timeout(self, bot, context, question, timeout=120):
        try:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: bot(context=context, question=question),
                ),
                timeout,
            )
        except asyncio.TimeoutError:
            return None  # indicates timeout

    def load_rag(self, preprocessing=None, **kwargs):
        if preprocessing == "PyMuPDFPreprocessing":
            preprocessing = PyMuPDFPreprocessing
            st.session_state.rag = RAG(
                path="./data/", preprocessing=preprocessing, **kwargs
            )
        elif preprocessing == "BasePreprocessing":
            st.session_state.rag = RAG(path="./data/")
        st.session_state.rag()

    def load_bot(self, host, model_name, token=None, provider=None):
        if host == "Ollama":
            st.session_state.bot = OllamaChatbot(name=model_name)
        elif host == "Hugginface":
            st.session_state.bot = HuggingFaceChatbot(
                model_name=model_name, token=token, provider=provider
            )

    def sidebar_options(self):
        with st.sidebar:
            st.header("Options")

            self.rag_action = st.selectbox(
                "RAG Actions:",
                options=["BasePreprocessing", "PyMuPDFPreprocessing"],
                index=0,
            )
            if self.rag_action == "PyMuPDFPreprocessing":
                with st.expander("PyMuPDF Configuration"):
                    st.write(
                        "You can configure image extraction and other parameters. (Only text by default)"
                    )
                    self.extract_images = st.checkbox("Extract images", value=False)
                    if self.extract_images:
                        st.warning("You must have Tesseract installed and configured.")
                        self.tesseract_path = st.text_input(label="Tesseract Path")
                    self.extract_tables = st.checkbox("Extract tables", value=False)

            self.uploaded_file = st.file_uploader(
                "Upload a document (PDF)", type=["pdf"]
            )
            if st.button("Process document"):
                if self.uploaded_file:
                    os.makedirs("./data", exist_ok=True)
                    if not os.path.exists(f"./data/{self.uploaded_file.name}"):
                        with open(f"./data/{self.uploaded_file.name}", "wb") as f:
                            f.write(self.uploaded_file.getbuffer())
                    if self.rag_action == "PyMuPDFPreprocessing":
                        kwargs = {
                            "extract_images": self.extract_images,
                            "extract_tables": self.extract_tables,
                        }
                        if self.extract_images:
                            kwargs["tesseract_path"] = self.tesseract_path

                        self.load_rag(preprocessing=self.rag_action, **kwargs)
                    else:
                        self.load_rag(preprocessing=self.rag_action)
                else:
                    if os.path.exists("./data/"):
                        shutil.rmtree("./data/")
                    if os.path.exists("./info/"):
                        shutil.rmtree("./info/")

                    st.session_state.rag = None

    def init_session(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def display_css(self):
        st.markdown(
            """
            <style>
            .chat-row {
                display: flex;
                margin-bottom: 1rem;
            }
            .chat-left {
                justify-content: flex-start;
            }
            .chat-right {
                justify-content: flex-end;
            }
            .chat-bubble {
                max-width: 70%;
                padding: 0.5rem;
                border-radius: 0.5rem;
            }
            .bot-bubble {
                background-color: #0e1117ff;
            }
            .user-bubble {
                background-color: #1a1c24ff;
                text-align: right;
            }
            .avatar {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                background-color: #ccc;
                display: inline-block;
                margin: 0 0.5rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def display_history(self):
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                self.display_user_message(msg["content"])
            else:
                self.display_bot_message(msg["content"])

    def display_user_message(self, content):
        st.markdown(
            f"""
            <div class="chat-row chat-right">
                <div class="chat-bubble user-bubble">{content}</div>
                <div class="avatar"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def display_bot_message(self, content):
        st.markdown(
            f"""
            <div class="chat-row chat-left">
                <div class="avatar"></div>
                <div class="chat-bubble bot-bubble">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def handle_input(self):
        user_input = st.chat_input("Write your message...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            self.display_user_message(user_input)
            if st.session_state.rag is None:
                context = None
            else:
                context = st.session_state.rag._search_context(user_input)

            respuesta = asyncio.run(
                self.call_bot_with_timeout(st.session_state.bot, context, user_input)
            )
            if respuesta is None:
                # first timeout, retry
                if not self.retry_done:
                    self.retry_done = True
                    respuesta = asyncio.run(
                        self.call_bot_with_timeout(
                            st.session_state.bot, context, user_input
                        )
                    )
                    if respuesta is None:
                        respuesta = self.call_bot_with_timeout(
                            st.session_state.bot,
                            """Say "I'm sorry, I haven't been able to generate an answer yet. Is there anything else I can assist you with?" In the same language as the user's question""",
                            user_input,
                        )
                else:
                    respuesta = self.call_bot_with_timeout(
                        st.session_state.bot,
                        """Say "I'm sorry, I haven't been able to generate an answer yet. Is there anything else I can assist you with?" In the same language as the user's question""",
                        user_input,
                    )
            else:
                self.retry_done = False  # reset if successful
            st.session_state.messages.append(
                {"role": "assistant", "content": respuesta}
            )
            self.display_bot_message(respuesta)

    def run(self):
        self.display_css()
        self.display_history()
        self.handle_input()


# Run app
if __name__ == "__main__":
    app = ChatApp()
    app.run()
