import os
import time
import uuid
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

# ---------- LangChain 0.2+ compatible imports ----------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec

# ---------- Config ----------
load_dotenv(override=True)

KB_DIR = os.environ.get("KB_DIR", "knowledge-base")
# Use HF Spaces persistent storage if enabled
PERSIST_DIR = os.environ.get("CHROMA_DIR", "/data/chroma-db" if os.path.exists("/data") else "chroma-db")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.environ.get("CHAT_MODEL",  "gpt-4o")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))

Path(KB_DIR).mkdir(parents=True, exist_ok=True)
Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

# ---------- Utilities ----------
def load_docs(data_dir: str):
    """Load PDF and Markdown files from a folder into LangChain Documents."""
    documents = []
    if not os.path.isdir(data_dir):
        return documents
    for file in os.listdir(data_dir):
        fp = os.path.join(data_dir, file)
        low = file.lower()
        if os.path.isfile(fp) and low.endswith(".pdf"):
            documents.extend(PyMuPDFLoader(fp).load())
        elif os.path.isfile(fp) and low.endswith(".md"):
            documents.extend(TextLoader(fp, encoding="utf-8").load())
    return documents

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# ---------- Build or load vector DB ----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY. Add it in your Space Settings → Secrets.")

embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=api_key)

def initialize_db():
    """If no persisted DB exists, build it from KB_DIR; otherwise load."""
    has_any = any(Path(PERSIST_DIR).iterdir())
    if not has_any:
        docs = load_docs(KB_DIR)
        chunks = chunk_docs(docs) if docs else []
        db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
        db.persist()
    else:
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    return db

vectordb = initialize_db()

# ---------- LLM + Retrieval Chain ----------
llm = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE, api_key=api_key)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Tri's representative. Answer ONLY from the provided context. "
            "If you don't know, say \"I don't know\" and ask the user to clarify. "
            "Reply in the same language as the question.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Question: {input}"),
    ]
)

documents_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectordb.as_retriever(search_kwargs={"k": 8})

retrieval_chain = create_retrieval_chain(retriever, documents_chain)
retrieval_chain = retrieval_chain | RunnableLambda(lambda x: x.get("answer", "")) | StrOutputParser()

# ---------- Per-session chat history ----------
store = {}

def get_session_history(user_id: str, conversation_id: str) -> ChatMessageHistory:
    key = (user_id, conversation_id)
    if key not in store:
        store[key] = ChatMessageHistory()
    return store[key]

chain_with_history = RunnableWithMessageHistory(
    retrieval_chain,
    get_session_history=get_session_history,
    history_messages_key="history",
    input_messages_key="input",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

# ---------- Chat handler (token-by-token generator) ----------
def respond(message, history, session_id):
    cfg = {"configurable": {"user_id": session_id, "conversation_id": session_id}}
    out = chain_with_history.invoke({"input": message}, config=cfg)
    answer = out if isinstance(out, str) else (out or "")
    if not answer:
        answer = "I don't know based on the provided knowledge base."
    partial = []
    for token in answer.split():
        partial.append(token)
        yield " ".join(partial)
        time.sleep(0.01)  # typing speed

def init_session():
    return str(uuid.uuid4())

# ---------- Gradio UI ----------
demo = gr.ChatInterface(
    fn=respond,
    type="messages",  # function receives (message, history, ...)
    chatbot=gr.Chatbot(
        type="messages",
        value=[{"role": "assistant", "content": "Hi, I'm Tri Imam's Digital Representative. Ask me anything!"}],
    ),
    title="Tri Imam Digital Twin",
    description="Answers are grounded in a pre-embedded knowledge base only.",
)

demo.launch()