import os
import json
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ─────────────────────────── Page config ───────────────────────────
st.set_page_config(
    page_title="Schola · Chat with your books",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_DIR = "chroma_library"
COLLECTION = "library"
REGISTRY_PATH = Path(DB_DIR) / "library.json"
Path(DB_DIR).mkdir(exist_ok=True)

# ─────────────────────────── Styling ───────────────────────────
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {
    padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, rgba(99,102,241,.18), rgba(236,72,153,.14));
    border: 1px solid rgba(128,128,128,.25);
}
.hero h1 {
    margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.5px;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p {margin: .35rem 0 0 0; opacity: .8; font-size: 1rem;}
.chip {
    display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0;
    border-radius: 999px; font-size: .75rem; font-weight: 600;
    background: rgba(99,102,241,.15); border: 1px solid rgba(99,102,241,.35);
}
.src {
    border-left: 3px solid #6366f1; padding: .4rem .8rem; margin: .5rem 0;
    background: rgba(128,128,128,.08); border-radius: 0 8px 8px 0;
    font-size: .85rem;
}
.empty {
    text-align: center; padding: 3rem 1rem; border-radius: 18px;
    border: 2px dashed rgba(128,128,128,.35); opacity: .9;
}
[data-testid="stSidebar"] .stButton>button {width: 100%;}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────── Library registry ───────────────────────────
def load_library() -> dict:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_library(lib: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(lib, indent=2), encoding="utf-8")


# ─────────────────────────── Cached resources ───────────────────────────
@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)…")
def get_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=DB_DIR,
    )


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, temperature: float):
    return init_chat_model(
        model=model_name,
        model_provider="groq",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise, helpful study assistant. Answer using ONLY the "
            "provided context. If the answer is not in the context, say you could "
            "not find it in the selected books. When useful, mention the book "
            "title and page in brackets, e.g. [Book Title, p.4]. Be clear and "
            "well-structured.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

# ─────────────────────────── Book operations ───────────────────────────
def add_book(uploaded_file, chunk_size: int, chunk_overlap: int, progress=None):
    """Index one PDF. Returns (ok, message)."""
    raw = uploaded_file.getvalue()
    book_id = hashlib.sha1(raw).hexdigest()[:12]
    lib = load_library()

    if book_id in lib:
        return False, f"“{lib[book_id]['title']}” is already in your library."

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        pages = PyPDFLoader(tmp_path).load()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        return False, f"“{uploaded_file.name}” has no extractable text (scanned PDF?)."

    title = Path(uploaded_file.name).stem.replace("_", " ").replace("-", " ").strip()
    ids = []
    for i, c in enumerate(chunks):
        c.metadata = {
            "book_id": book_id,
            "book_title": title,
            "page": int(c.metadata.get("page", 0)) + 1,
        }
        ids.append(f"{book_id}-{i}")

    vs = get_vectorstore()
    batch = 64
    for start in range(0, len(chunks), batch):
        vs.add_documents(chunks[start : start + batch], ids=ids[start : start + batch])
        if progress:
            progress(min(1.0, (start + batch) / len(chunks)))

    lib[book_id] = {
        "title": title,
        "filename": uploaded_file.name,
        "pages": len(pages),
        "chunks": len(chunks),
        "added": datetime.now().strftime("%d %b %Y, %H:%M"),
    }
    save_library(lib)
    return True, f"Added “{title}” ({len(pages)} pages, {len(chunks)} chunks)."


def remove_book(book_id: str):
    lib = load_library()
    meta = lib.pop(book_id, None)
    if meta:
        get_vectorstore().delete(ids=[f"{book_id}-{i}" for i in range(meta["chunks"])])
        save_library(lib)
        st.session_state.pop(f"active_{book_id}", None)
        flash("success", f"Removed “{meta['title']}”.")


def flash(kind: str, msg: str):
    st.session_state.setdefault("flash", []).append((kind, msg))


# ─────────────────────────── Session state ───────────────────────────
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("pending_prompt", None)

for kind, msg in st.session_state.pop("flash", []):
    st.toast(msg, icon="✅" if kind == "success" else "⚠️")

library = load_library()

# ─────────────────────────── Sidebar ───────────────────────────
with st.sidebar:
    st.markdown("## 📚 Library")

    total_pages = sum(b["pages"] for b in library.values())
    total_chunks = sum(b["chunks"] for b in library.values())
    m1, m2, m3 = st.columns(3)
    m1.metric("Books", len(library))
    m2.metric("Pages", total_pages)
    m3.metric("Chunks", total_chunks)

    with st.expander("⚙️ Settings"):
        chunk_size = st.slider("Chunk size", 300, 2000, 1000, 100,
                               help="Applies to books added from now on.")
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)
        st.divider()
        k = st.slider("Passages to use (k)", 1, 12, 4)
        fetch_k = st.slider("Candidates to fetch (fetch_k)", 4, 40, 12)
        lam = st.slider("MMR diversity (λ)", 0.0, 1.0, 0.5, 0.05,
                        help="1 = pure relevance, 0 = max diversity.")
        st.divider()
        model_name = st.text_input("Groq model", "openai/gpt-oss-120b")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.markdown("### ➕ Add books")
    files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )
    if st.button("Index selected books", type="primary", disabled=not files):
        bar = st.progress(0.0, text="Starting…")
        for idx, f in enumerate(files):
            bar.progress(idx / len(files), text=f"Indexing {f.name}…")
            try:
                ok, msg = add_book(
                    f, chunk_size, chunk_overlap,
                    progress=lambda p, i=idx: bar.progress(
                        (i + p) / len(files), text=f"Indexing {f.name}…"
                    ),
                )
            except Exception as e:
                ok, msg = False, f"Failed on {f.name}: {e}"
            flash("success" if ok else "warn", msg)
        bar.empty()
        st.session_state.uploader_key += 1
        st.rerun()

    st.markdown("### 🗂️ Your books")
    if not library:
        st.caption("No books yet. Upload a PDF above to get started.")

    for bid, meta in library.items():
        with st.container(border=True):
            st.markdown(f"**{meta['title']}**")
            st.caption(f"{meta['pages']} pages · {meta['chunks']} chunks · {meta['added']}")
            c1, c2 = st.columns([3, 2])
            c1.toggle("Use in chat", value=True, key=f"active_{bid}")
            with c2.popover("🗑️ Remove"):
                st.warning("Remove this book from the library?")
                if st.button("Yes, delete", key=f"del_{bid}", type="primary"):
                    remove_book(bid)
                    st.rerun()

    st.divider()
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

active_ids = [
    bid for bid in library if st.session_state.get(f"active_{bid}", True)
]

# ─────────────────────────── Main area ───────────────────────────
st.markdown(
    """
<div class="hero">
  <h1>Schola</h1>
  <p>Ask questions across your books. Every answer is grounded in the pages you choose, with sources.</p>
</div>
""",
    unsafe_allow_html=True,
)

if active_ids:
    chips = "".join(f'<span class="chip">📖 {library[b]["title"]}</span>' for b in active_ids)
    st.markdown(f"**Searching in:** {chips}", unsafe_allow_html=True)

if not library:
    st.markdown(
        """
<div class="empty">
  <h3>Your library is empty</h3>
  <p>Add one or more PDFs from the sidebar, then ask anything about them.</p>
</div>
""",
        unsafe_allow_html=True,
    )
elif not active_ids:
    st.info("All books are switched off. Turn on at least one in the sidebar to chat.")

# Suggestion buttons on a fresh chat
if active_ids and not st.session_state.messages:
    st.markdown("##### Try asking")
    cols = st.columns(3)
    suggestions = [
        "Summarize the main idea of this book",
        "What are the key concepts I should know?",
        "Explain the most important topic in simple words",
    ]
    for col, text in zip(cols, suggestions):
        if col.button(text, use_container_width=True):
            st.session_state.pending_prompt = text
            st.rerun()


def render_sources(sources):
    with st.expander(f"📎 Sources ({len(sources)})"):
        for s in sources:
            snippet = s["text"].replace("\n", " ")
            snippet = snippet[:380] + ("…" if len(snippet) > 380 else "")
            st.markdown(
                f'<div class="src"><b>{s["title"]}</b> · page {s["page"]}<br>{snippet}</div>',
                unsafe_allow_html=True,
            )


# Replay history
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🤖"):
        st.markdown(m["content"])
        if m.get("sources"):
            render_sources(m["sources"])

# Input
typed = st.chat_input(
    "Ask something about your books…",
    disabled=not active_ids,
)
query = typed or st.session_state.pop("pending_prompt", None)

if query and active_ids:
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found. Add it to your .env file and restart.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching your books…"):
            retriever = get_vectorstore().as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": max(fetch_k, k),
                    "lambda_mult": lam,
                    "filter": {"book_id": {"$in": active_ids}},
                },
            )
            docs = retriever.invoke(query)

        if not docs:
            answer = "I couldn't find anything relevant in the selected books."
            st.markdown(answer)
            sources = []
        else:
            context = "\n\n".join(
                f"[{d.metadata['book_title']}, p.{d.metadata['page']}]\n{d.page_content}"
                for d in docs
            )
            final_prompt = PROMPT.invoke({"context": context, "question": query})
            llm = get_llm(model_name, temperature)

            def token_stream():
                for chunk in llm.stream(final_prompt):
                    if chunk.content:
                        yield chunk.content

            answer = st.write_stream(token_stream())
            sources = [
                {
                    "title": d.metadata["book_title"],
                    "page": d.metadata["page"],
                    "text": d.page_content,
                }
                for d in docs
            ]
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
