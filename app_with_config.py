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

        if "view" not in st.session_state:
            st.session_state.view = "chat"  # Options: "chat" or "config"

        if "bot" not in st.session_state:
            st.session_state.bot = None
        if "rag" not in st.session_state:
            st.session_state.rag = None
        if "rag_config" not in st.session_state:
            st.session_state.rag_config = {
                "rag_action": "BasePreprocessing",
                "extract_images": False,
                "extract_tables": False,
                "tesseract_path": None,
            }

        if os.path.exists("./data/"):
            shutil.rmtree("./data/")
        if os.path.exists("./info/"):
            shutil.rmtree("./info/")

        self.init_session()

        self.retry_done = False  # to handle timeout retries

        # Display the appropriate view
        if st.session_state.view == "chat":
            self.display_chat_view()
        else:
            self.display_config_view()

    def display_chat_view(self):
        st.title("Interactive Chatbot")
        self.sidebar_options()
        self.display_css()
        self.display_history()
        self.handle_input()

    def display_config_view(self):
        st.title("Configuration")
        self.show_configuration()

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

    def sidebar_options(self):
        with st.sidebar:
            self.uploaded_file = st.file_uploader(
                "Upload a document (PDF)", type=["pdf"]
            )

            if st.button("Process document", type="primary"):
                if self.uploaded_file:
                    os.makedirs("./data", exist_ok=True)
                    if not os.path.exists(f"./data/{self.uploaded_file.name}"):
                        with open(f"./data/{self.uploaded_file.name}", "wb") as f:
                            f.write(self.uploaded_file.getbuffer())

                    rag_action = st.session_state.rag_config["rag_action"]

                    if rag_action == "PyMuPDFPreprocessing":
                        kwargs = {
                            "extract_images": st.session_state.rag_config[
                                "extract_images"
                            ],
                            "extract_tables": st.session_state.rag_config[
                                "extract_tables"
                            ],
                        }
                        if st.session_state.rag_config["extract_images"]:
                            kwargs["tesseract_path"] = st.session_state.rag_config[
                                "tesseract_path"
                            ]

                        self.load_rag(preprocessing=rag_action, **kwargs)
                    else:
                        self.load_rag(preprocessing=rag_action)
                else:
                    if os.path.exists("./data/"):
                        shutil.rmtree("./data/")
                    if os.path.exists("./info/"):
                        shutil.rmtree("./info/")

                    st.session_state.rag = None

            if st.button("Configuration ⚙️"):
                st.session_state.view = "config"
                st.rerun()

    def show_configuration(self):
        st.markdown("## Host Configuration")
        host_list = ["Ollama", "Hugginface"]
        host = st.selectbox(
            label="Host",
            options=host_list,
            index=host_list.index(st.session_state.get("host", "Hugginface")),
        )

        if host == "Ollama":
            try:
                model_options = [model.model for model in ollama.list()["models"]]
                model_name = st.selectbox("Model:", options=model_options)
            except:
                st.error(
                    "Could not connect to Ollama. Make sure it's installed and running."
                )
                model_name = st.text_input(label="Model Name", value="llama3.1:8b")
            token = None
            provider = None
        elif host == "Hugginface":
            model_name = st.text_input(
                label="Model Name", value="deepseek-ai/DeepSeek-V3-0324"
            )
            token = st.text_input(label="Token", value="")
            provider = st.text_input(label="Provider", value="hyperbolic")

        st.markdown("## RAG configuration")
        rag_list = ["BasePreprocessing", "PyMuPDFPreprocessing"]
        rag_action = st.selectbox(
            "RAG Actions:",
            rag_list,
            index=rag_list.index(
                st.session_state.rag_config.get("rag_action", "BasePreprocessing")
            ),
        )

        extract_images = False
        extract_tables = False
        tesseract_path = None

        if rag_action == "PyMuPDFPreprocessing":
            with st.expander("PyMuPDF Configuration"):

                st.write("You can configure image extraction and other parameters.")
                extract_images = st.checkbox(
                    "Extract images",
                    value=st.session_state.rag_config.get("extract_images", False),
                )
                if extract_images:
                    st.warning("You must have Tesseract installed and configured.")
                    tesseract_path = st.text_input("Tesseract Path")
                extract_tables = st.checkbox(
                    "Extract tables",
                    value=bool(
                        st.session_state.rag_config.get("extract_tables", False)
                    ),
                )

        if st.button("Save Configuration"):
            if extract_images and not tesseract_path:
                st.error("Please provide the Tesseract path if extracting images.")
                return
            # Save RAG configuration
            st.session_state.host = host
            st.session_state.rag_config = {
                "rag_action": rag_action,
                "extract_images": extract_images,
                "extract_tables": extract_tables,
                "tesseract_path": tesseract_path,
            }

            # Load the bot based on configuration
            self.load_bot(
                host=host, model_name=model_name, token=token, provider=provider
            )

            st.success("Configuration saved!")
            time.sleep(2)

            st.session_state.view = "chat"
            st.rerun()

    def load_bot(self, host, model_name, token=None, provider=None):
        if host == "Ollama":
            st.session_state.bot = OllamaChatbot(name=model_name)
        elif host == "Hugginface":
            st.session_state.bot = HuggingFaceChatbot(
                model_name=model_name, token=token, provider=provider
            )

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

            if st.session_state.bot is None:
                st.error(
                    "Please configure a chatbot model first in the Configuration page"
                )
                return
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
                        respuesta = "I'm sorry, I haven't been able to generate an answer yet. Is there anything else I can assist you with?"
                else:
                    respuesta = "I'm sorry, I haven't been able to generate an answer yet. Is there anything else I can assist you with?"
            else:
                self.retry_done = False  # reset if successful

            st.session_state.messages.append(
                {"role": "assistant", "content": respuesta}
            )
            self.display_bot_message(respuesta)

    def run(self):
        # The main logic is now handled in __init__ and the respective view methods
        pass


# Run app
if __name__ == "__main__":
    app = ChatApp()
