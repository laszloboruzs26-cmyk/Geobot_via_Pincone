# -*- coding: utf-8 -*-
"""Streamlit Geography RAG chatbot using Pinecone.

Deploy this file as app.py on Streamlit.
Before running, make sure your Pinecone index has already been populated
by running index_to_pinecone.py once locally or in Colab.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ============================================================
# CONFIG
# ============================================================

DEFAULT_INDEX_NAME = "geography-kb"
DEFAULT_NAMESPACE = "default"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    return value or os.getenv(name, default)


def configure_environment() -> None:
    """Set required API keys into os.environ for LangChain integrations."""
    openai_key = get_secret("OPENAI_API_KEY")
    pinecone_key = get_secret("PINECONE_API_KEY")

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if pinecone_key:
        os.environ["PINECONE_API_KEY"] = pinecone_key


@st.cache_resource(show_spinner=False)
def load_llm_and_vectorstore(index_name: str, namespace: str):
    """Connect to an existing Pinecone index and initialise the LLM."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return llm, vectorstore


rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user question into a keyword-rich search query for retrieving relevant "
        "Wikipedia-style geography passages from a vector database. Keep the main entities, "
        "places, scientific terms, and topic words. Remove filler words and conversational phrasing. "
        "Do not answer the question. Return only the rewritten query.\n\n"
        "Rules:\n"
        "- For definition or 'what is' questions, keep the main concept exactly.\n"
        "- For location questions, keep the place/entity and location-related terms.\n"
        "- For 'why' questions, preserve the causal intent explicitly and include words such as "
        "causes, factors, reasons, influences, mechanisms, or effects when useful.\n"
        "- Prefer compact keyword phrases over full sentences.\n"
        "- Keep important original terms; do not replace them with vaguer synonyms.\n\n"
        "Examples:\n"
        "Question: What is geography?\n"
        "Query: geography definition study of places human environment\n\n"
        "Question: What continent is France in?\n"
        "Query: France continent Europe location geography\n\n"
        "Question: Why are deserts dry?\n"
        "Query: deserts dry causes factors low precipitation atmospheric circulation geographic barriers aridity",
    ),
    ("human", "{question}"),
])

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful geography assistant. Answer ONLY using the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Provide a complete but concise answer and cite source names when possible.",
    ),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
])


def format_context(docs: list[Document]) -> str:
    lines: list[str] = []
    for doc in docs:
        src = Path(str(doc.metadata.get("source", "Unknown source"))).name
        page = doc.metadata.get("page", None)
        loc = f"{src}" + (f" p.{page}" if page is not None else "")
        lines.append(f"[Source: {loc}]\n{doc.page_content}")
    return "\n\n".join(lines)


def rewrite_query(llm: ChatOpenAI, question: str) -> str:
    response = llm.invoke(rewrite_prompt.format_messages(question=question))
    return response.content.strip()


def retrieve_documents(vectorstore: PineconeVectorStore, query: str, top_k: int, use_mmr: bool, fetch_k: int, lambda_mult: float):
    if use_mmr:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever.invoke(query)


def answer_question(
    question: str,
    llm: ChatOpenAI,
    vectorstore: PineconeVectorStore,
    top_k: int,
    use_mmr: bool,
    fetch_k: int,
    lambda_mult: float,
    use_query_rewrite: bool,
):
    retrieval_query = rewrite_query(llm, question) if use_query_rewrite else question
    docs = retrieve_documents(vectorstore, retrieval_query, top_k, use_mmr, fetch_k, lambda_mult)
    context = format_context(docs)
    response = llm.invoke(answer_prompt.format_messages(context=context, question=question))
    return response.content.strip(), docs, retrieval_query


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Geography RAG Chatbot", page_icon="🌍", layout="wide")
configure_environment()

st.title("🌍 Geography RAG Chatbot")
st.caption("RAG chatbot using OpenAI + Pinecone. The corpus should already be indexed in Pinecone.")

missing = [name for name in ["OPENAI_API_KEY", "PINECONE_API_KEY"] if not os.getenv(name)]
if missing:
    st.error(
        "Missing required secrets: " + ", ".join(missing) +
        ". Add them in Streamlit Cloud → App settings → Secrets."
    )
    st.stop()

with st.sidebar:
    st.header("Retrieval settings")
    index_name = st.text_input("Pinecone index name", value=get_secret("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME))
    namespace = st.text_input("Pinecone namespace", value=get_secret("PINECONE_NAMESPACE", DEFAULT_NAMESPACE))
    top_k = st.slider("TOP_K", min_value=1, max_value=20, value=8)
    use_mmr = st.toggle("Use MMR retrieval", value=True)
    fetch_k = st.slider("FETCH_K", min_value=top_k, max_value=60, value=max(20, top_k * 3))
    lambda_mult = st.slider("MMR lambda", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    use_query_rewrite = st.toggle("Rewrite query before retrieval", value=True)

try:
    llm, vectorstore = load_llm_and_vectorstore(index_name=index_name, namespace=namespace)
except Exception as exc:
    st.error(f"Could not connect to Pinecone/OpenAI: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a geography question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving from Pinecone and generating answer..."):
            try:
                answer, docs, retrieval_query = answer_question(
                    question=question,
                    llm=llm,
                    vectorstore=vectorstore,
                    top_k=top_k,
                    use_mmr=use_mmr,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    use_query_rewrite=use_query_rewrite,
                )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                with st.expander("Retrieved context"):
                    st.write("Retrieval query:", retrieval_query)
                    for i, doc in enumerate(docs, start=1):
                        source = doc.metadata.get("source", "Unknown source")
                        st.markdown(f"**Chunk {i} — Source: {source}**")
                        st.write(doc.page_content[:1200])
                        st.divider()
            except Exception as exc:
                st.error(f"Error while answering: {exc}")
