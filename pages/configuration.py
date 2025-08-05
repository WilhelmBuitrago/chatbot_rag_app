import streamlit as st
import time
import ollama
from chatbot_rag.chat import OllamaChatbot, HuggingFaceChatbot

st.set_page_config(page_title="Configuration")
st.title("Configuration")
st.markdown("## Host Configuration")
host = st.selectbox(label="Host", options=["Ollama", "Hugginface"], index=1)
if host == "Ollama":
    model_name = st.selectbox(
        "Model:", options=[model.model for model in ollama.list()["models"]]
    )
    token = None
    provider = None

    # load_bot(host, model_name=self.name)
elif host == "Hugginface":
    model_name = st.text_input(label="Model Name", value="deepseek-ai/DeepSeek-V3-0324")
    token = st.text_input(label="Token", value="")
    provider = st.text_input(label="Provider", value="hyperbolic")

st.markdown("## RAG configuration")
rag_action = st.selectbox(
    "RAG Actions:", ["BasePreprocessing", "PyMuPDFPreprocessing"], index=0
)

if rag_action == "PyMuPDFPreprocessing":
    with st.expander("PyMuPDF Configuration"):
        st.write("You can configure image extraction and other parameters.")
        extract_images = st.checkbox("Extract images", value=False)
        if extract_images:
            st.warning("You must have Tesseract installed and configured.")
            tesseract_path = st.text_input("Tesseract Path")
        extract_tables = st.checkbox("Extract tables", value=False)

st.session_state.rag_config = {
    "rag_action": rag_action,
    "extract_images": extract_images if rag_action == "PyMuPDFPreprocessing" else False,
    "extract_tables": extract_tables if rag_action == "PyMuPDFPreprocessing" else False,
    "tesseract_path": tesseract_path if rag_action == "PyMuPDFPreprocessing" else None,
}


def load_bot(host, model_name, token=None, provider=None):
    if host == "Ollama":
        st.session_state.bot = OllamaChatbot(name=model_name)
    elif host == "Hugginface":
        st.session_state.bot = HuggingFaceChatbot(
            model_name=model_name, token=token, provider=provider
        )


if st.button("Save Configuration"):
    load_bot(host=host, model_name=model_name, token=token, provider=provider)

    st.success("Configuration saved!")
    time.sleep(2)
    st.switch_page("app.py")
