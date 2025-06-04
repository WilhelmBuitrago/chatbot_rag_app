import streamlit as st
from chatbot_rag.RAG import RAG
from chatbot_rag.chat import Chatbot
from chatbot_rag.preprocessing import PyMuPDFPreprocessing, BasePreprocessing
import os
import shutil
import ollama
import time
import asyncio


class ChatApp:
    def __init__(self):
        st.set_page_config(page_title="Chatbot Interactivo", layout="centered")
        st.title("Chatbot Interactivo")
        if "bot" not in st.session_state:
            st.session_state.bot = None
        if "rag" not in st.session_state:
            st.session_state.rag = None

        if os.path.exists("./data/"):
            shutil.rmtree("./data/")
        if os.path.exists("./info/"):
            shutil.rmtree("./info/")

        self.opciones_sidebar()

        self.init_session()

        self.reintento_hecho = False  # para manejar reintentos de timeout

    async def llamar_bot_con_timeout(self, bot, context, question, timeout=120):
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
            return None  # indica que se agotó el tiempo

    def cargar_rag(self, preprocessing=None, **kwargs):
        if preprocessing == "PyMuPDFPreprocessing":
            kwargs = {"extract_images": False}
            preprocessing = PyMuPDFPreprocessing
            st.session_state.rag = RAG(
                path="./data/", preprocessing=preprocessing, **kwargs
            )
        elif preprocessing == "BasePreprocessing":
            st.session_state.rag = RAG(path="./data/")
        st.session_state.rag()

    def cargar_bot(self, model_name):
        st.session_state.bot = Chatbot(name=model_name)

    def opciones_sidebar(self):
        with st.sidebar:
            st.header("Opciones")
            nombre_modelo = st.selectbox(
                "Modelo:", options=[model.model for model in ollama.list()["models"]]
            )

            if not hasattr(self, "name") or nombre_modelo != self.name:
                self.name = nombre_modelo
                self.cargar_bot(self.name)
            self.accion_rag = st.selectbox(
                "Acciones sobre RAG:",
                options=["BasePreprocessing", "PyMuPDFPreprocessing"],
                index=0,
            )
            self.archivo_subido = st.file_uploader(
                "Sube un documento (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"]
            )
            if st.button("Procesar documento"):
                if self.archivo_subido:
                    os.makedirs("./data", exist_ok=True)
                    if not os.path.exists(f"./data/{self.archivo_subido.name}"):
                        with open(f"./data/{self.archivo_subido.name}", "wb") as f:
                            f.write(self.archivo_subido.getbuffer())
                    self.cargar_rag(preprocessing=self.accion_rag)
                else:
                    if os.path.exists("./data/"):
                        shutil.rmtree("./data/")
                    if os.path.exists("./info/"):
                        shutil.rmtree("./info/")

                    st.session_state.rag = None

    def init_session(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def mostrar_css(self):
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

    def mostrar_historial(self):
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                self.mostrar_mensaje_usuario(msg["content"])
            else:
                self.mostrar_mensaje_bot(msg["content"])

    def mostrar_mensaje_usuario(self, contenido):
        st.markdown(
            f"""
            <div class="chat-row chat-right">
                <div class="chat-bubble user-bubble">{contenido}</div>
                <div class="avatar"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def mostrar_mensaje_bot(self, contenido):
        st.markdown(
            f"""
            <div class="chat-row chat-left">
                <div class="avatar"></div>
                <div class="chat-bubble bot-bubble">{contenido}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def manejar_entrada(self):
        user_input = st.chat_input("Escribe tu mensaje...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            self.mostrar_mensaje_usuario(user_input)
            if st.session_state.rag is None:
                context = """None"""
            else:
                context = st.session_state.rag._search_context(user_input)

            respuesta = asyncio.run(
                self.llamar_bot_con_timeout(st.session_state.bot, context, user_input)
            )
            if respuesta is None:
                # primer timeout, reintenta
                if not self.reintento_hecho:
                    self.reintento_hecho = True
                    respuesta = asyncio.run(
                        self.llamar_bot_con_timeout(
                            st.session_state.bot, context, user_input
                        )
                    )
                    if respuesta is None:
                        respuesta = "Lo siento, la respuesta está tomando demasiado tiempo. ¿Hay otra cosa en la que pueda ayudarte?"
                else:
                    respuesta = "Lo siento, aún no he podido generar una respuesta. ¿Hay algo más en lo que pueda asistirte?"
            else:
                self.reintento_hecho = False  # reinicia si fue exitoso
            st.session_state.messages.append(
                {"role": "assistant", "content": respuesta}
            )
            self.mostrar_mensaje_bot(respuesta)

    def ejecutar(self):
        self.mostrar_css()
        self.mostrar_historial()
        self.manejar_entrada()


# Ejecutar app
if __name__ == "__main__":
    app = ChatApp()
    app.ejecutar()
