# pdf-reader-app.
A simple Python-based PDF reader that extracts and displays text from PDF files.
"""
Streamlit + LangChain RAG Resume Search Chatbot (Robust version)
File: streamlit_langchain_rag_resume_bot.py

This version is updated to be *robust* in environments where Streamlit or other optional
libraries are not available (for example: sandboxed execution). It provides:

- A Streamlit-based UI when `streamlit` is installed (the original behaviour).
- A CLI fallback that runs in the terminal if Streamlit is not installed.
- Graceful fallbacks for missing LangChain/FAISS by using sentence-transformers and an
  in-memory similarity search.
- Small unit tests for helper functions (run with `--run-tests`).

How to use:
- To run with Streamlit UI (recommended):
    pip install -U streamlit langchain openai faiss-cpu PyPDF2 sentence_transformers tiktoken
    export OPENAI_API_KEY=sk-...
    streamlit run streamlit_langchain_rag_resume_bot.py

- To run in CLI mode (no Streamlit needed):
    pip install -U sentence_transformers numpy PyPDF2
    python streamlit_langchain_rag_resume_bot.py --pdf resume1.pdf resume2.pdf

Notes:
- If LangChain or FAISS is not available, the script will still work using a simple
  in-memory index built with sentence-transformers.
- If you want LLM-generated answers (concise summaries), set OPENAI_API_KEY and have
  the `openai`/`langchain` packages installed; otherwise the tool returns matching snippets.

"""

from __future__ import annotations

import os
import sys
import tempfile
import argparse
import logging
from typing import List, Dict, Any, Tuple

# Optional imports: streamlit, langchain, faiss. We'll import them when available and
# otherwise provide graceful fallbacks.

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# LangChain & FAISS optional components
try:
    from langchain.document_loaders import PyPDFLoader
except Exception:
    PyPDFLoader = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

try:
    from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
except Exception:
    OpenAIEmbeddings = None
    HuggingFaceEmbeddings = None

try:
    from langchain.vectorstores import FAISS
except Exception:
    FAISS = None

try:
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.llms import OpenAI as LangChainOpenAI
except Exception:
    RetrievalQA = None
    ChatOpenAI = None
    LangChainOpenAI = None

# Lightweight PDF reader fallback
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# Sentence-transformers (used as fallback for embeddings)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None
    np = None

import re
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "auto")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ---------------------------
# Helper: candidate-name extraction
# ---------------------------

def extract_candidate_names(text: str) -> List[str]:
    """Simple heuristics to extract candidate names from resume text.

    This function is intentionally conservative: it looks for lines beginning with
    common labels (Name, Full Name, Candidate) and also tries a heuristic on the
    first few lines.
    """
    names: List[str] = []
    if not text:
        return names
    lines = text.splitlines()
    # Look for explicit labels
    for i, line in enumerate(lines[:30]):
        m = re.search(r"^\s*(Name|Full Name|Candidate)[:\-\s]+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})", line)
        if m:
            candidate = m.group(2).strip()
            if candidate not in names:
                names.append(candidate)
    # Fallback: check top lines for capitalized name-like patterns
    for line in lines[:10]:
        s = line.strip()
        if s and len(s.split()) <= 4 and re.match(r"^[A-Z][a-z]+(\s+[A-Z][a-z]+){0,3}$", s):
            if s not in names:
                names.append(s)
    return names

# ---------------------------
# Text chunking
# ---------------------------

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks of characters, preserving words.

    This is a simple fallback that mimics what LangChain's RecursiveCharacterTextSplitter
    does for the purposes of building a semantic index.
    """
    if not text:
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        if end >= length:
            chunk = text[start: length]
            chunks.append(chunk)
            break
        # try to break at newline or space to avoid cutting words
        cut_at = text.rfind("\n", start, end)
        if cut_at == -1:
            cut_at = text.rfind(" ", start, end)
        if cut_at <= start:
            cut_at = end
        chunk = text[start:cut_at]
        chunks.append(chunk)
        start = cut_at - chunk_overlap if (cut_at - chunk_overlap) > 0 else cut_at
    return chunks

# ---------------------------
# PDF loading (LangChain or fallback)
# ---------------------------

def load_pdf_pages(file_path: str) -> List[Dict[str, Any]]:
    """Return a list of dicts with 'page_content' and 'metadata'.

    Uses LangChain's PyPDFLoader if available, otherwise falls back to PyPDF2.
    """
    pages: List[Dict[str, Any]] = []
    if PyPDFLoader is not None:
        try:
            loader = PyPDFLoader(file_path)
            lc_pages = loader.load()
            for i, p in enumerate(lc_pages):
                pages.append({
                    "page_content": p.page_content,
                    "metadata": {"source": os.path.basename(file_path), "page": i + 1},
                })
            return pages
        except Exception as e:
            logger.warning("PyPDFLoader failed (%s), falling back to PyPDF2: %s", file_path, e)
    # Fallback: PyPDF2
    if PdfReader is None:
        raise RuntimeError("Neither langchain.PyPDFLoader nor PyPDF2 is available. Install PyPDF2 or langchain to read PDFs.")
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages.append({
                "page_content": text,
                "metadata": {"source": os.path.basename(file_path), "page": i + 1},
            })
        return pages
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF {file_path}: {e}")

# ---------------------------
# Embeddings: try multiple options
# ---------------------------

def get_embedding_function() -> Tuple[callable, str]:
    """Return (embed_fn, provider_name).

    embed_fn(texts: List[str]) -> List[List[float]]
    """
    # 1) Prefer OpenAIEmbeddings via LangChain if available and API key set
    if EMBEDDINGS_PROVIDER == "openai" or (EMBEDDINGS_PROVIDER == "auto" and OPENAI_API_KEY and OpenAIEmbeddings is not None):
        if OpenAIEmbeddings is not None and OPENAI_API_KEY:
            emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            return lambda texts: emb.embed_documents(texts), "openai"
    # 2) Use LangChain's HuggingFaceEmbeddings if available
    if HuggingFaceEmbeddings is not None:
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return lambda texts: emb.embed_documents(texts), "hf-langchain"
    # 3) Use sentence-transformers directly if available
    if SentenceTransformer is not None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        def hf_embed(texts: List[str]) -> List[List[float]]:
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
        return hf_embed, "hf-sentence-transformers"
    raise RuntimeError("No embeddings provider available. Install langchain (with HuggingFaceEmbeddings) or sentence_transformers.")
try:
    from langchain.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("No embeddings provider available...")

# ---------------------------
# In-memory simple vector index
# ---------------------------
class SimpleInMemoryIndex:
    def __init__(self, docs: List[Dict[str, Any]], embedding_fn: callable):
        self.docs = docs
        self.embedding_fn = embedding_fn
        self.embeddings = None
        self._build()

    def _build(self):
        texts = [d["page_content"] for d in self.docs]
        embs = self.embedding_fn(texts)
        # ensure numpy array
        if np is not None:
            self.embeddings = np.array(embs, dtype=float)
        else:
            # fallback: keep list
            self.embeddings = embs

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.embedding_fn([query])[0]
        if np is not None and isinstance(self.embeddings, np.ndarray):
            qv = np.array(q_emb, dtype=float)
            # cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1) * (np.linalg.norm(qv) + 1e-12)
            sims = np.dot(self.embeddings, qv) / norms
            # safety for NaN
            sims = np.nan_to_num(sims)
            topk_idx = list(np.argsort(-sims)[:k])
            return [self.docs[i] for i in topk_idx]
        else:
            # naive dot-product/list fallback
            scores = []
            for i, emb in enumerate(self.embeddings):
                # compute simplistic similarity
                score = sum(a * b for a, b in zip(emb, q_emb))
                scores.append((score, i))
            scores.sort(reverse=True)
            topk = [self.docs[i] for (_, i) in scores[:k]]
            return topk

# ---------------------------
# Build index: try FAISS then fallback to SimpleInMemoryIndex
# ---------------------------

def build_index(docs: List[Dict[str, Any]], embedding_fn: callable):
    """Try to build a FAISS index using LangChain if possible; otherwise build in-memory index."""
    if FAISS is not None and (HuggingFaceEmbeddings is not None or OpenAIEmbeddings is not None):
        try:
            # LangChain expects Document objects; create them if needed
            from langchain.schema import Document as LCDoc
            lc_docs = [LCDoc(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in docs]
            index = FAISS.from_documents(lc_docs, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
            return index, "faiss"
        except Exception as e:
            logger.warning("FAISS index build failed, falling back to in-memory index: %s", e)
    # fallback
    return SimpleInMemoryIndex(docs, embedding_fn), "inmemory"

# ---------------------------
# Querying
# ---------------------------

def retrieve_top_docs(index, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Return list of docs. Handles both FAISS index and SimpleInMemoryIndex."""
    # FAISS-like object (LangChain) has as_retriever or similarity_search
    try:
        if hasattr(index, "as_retriever"):
            # LangChain vectorstore
            retriever = index.as_retriever(search_type="similarity", search_kwargs={"k": k})
            docs = retriever.get_relevant_documents(query)
            # convert to our simple dict form
            out = []
            for d in docs:
                out.append({"page_content": d.page_content, "metadata": d.metadata})
            return out
    except Exception:
        pass
    # Fallback: our SimpleInMemoryIndex
    if hasattr(index, "similarity_search"):
        return index.similarity_search(query, k=k)
    raise RuntimeError("Index object does not support retrieval")

def generate_answer(query: str, index, use_llm: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (answer_text, top_docs).

    If use_llm is True and a LangChain-compatible LLM is available with OPENAI_API_KEY,
    attempt to use RetrievalQA. Otherwise, return concatenated snippets as the answer.
    """
    top_docs = retrieve_top_docs(index, query, k=6)
    if use_llm and RetrievalQA is not None and OPENAI_API_KEY:
        try:
            # Use LangChain's RetrievalQA with ChatOpenAI if possible
            if ChatOpenAI is not None:
                llm = ChatOpenAI(temperature=0)
            elif LangChainOpenAI is not None:
                llm = LangChainOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
            else:
                llm = None
            if llm is not None:
                # build a retriever-like wrapper for FAISS or in-memory using our function
                class _TmpRetriever:
                    def __init__(self, docs):
                        self.docs = docs
                    def get_relevant_documents(self, query):
                        from langchain.schema import Document as LCDoc
                        return [LCDoc(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in self.docs]
                retriever = _TmpRetriever(top_docs)
                qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
                result = qa.run(query)
                return result, top_docs
        except Exception as e:
            logger.warning("LLM retrieval failed, falling back to snippet return: %s", e)
    # Default: return concatenated snippets
    snippets = []
    for d in top_docs:
        src = d.get("metadata", {}).get("source", "unknown")
        page = d.get("metadata", {}).get("page", "?")
        content = d.get("page_content", "")
        snippet = (content.strip().replace("\n", " ")[:500] + ("..." if len(content) > 500 else ""))
        snippets.append(f"Source: {src} (page {page})\n{snippet}")
    answer = "\n\n".join(snippets) or "No relevant content found."
    return answer, top_docs

# ---------------------------
# CLI mode
# ---------------------------

def run_cli_mode(pdf_paths: List[str]):
    embed_fn, provider = get_embedding_function()
    # load pages from PDFs
    docs: List[Dict[str, Any]] = []
    if not pdf_paths:
        print("No PDF paths provided. Running demo with synthetic document.")
        docs = [{"page_content": "John Doe\nExperience: Python, AWS, 5 years\nEducation: B.Tech", "metadata": {"source": "demo1.pdf", "page": 1}},
                {"page_content": "Jane Smith\nExperience: Java, Spring, 3 years\nEducation: B.Sc", "metadata": {"source": "demo2.pdf", "page": 1}}]
    else:
        for p in pdf_paths:
            p = os.path.abspath(p)
            if not os.path.exists(p):
                print(f"Warning: file not found: {p}")
                continue
            pages = load_pdf_pages(p)
            # create chunks for each page
            for pg in pages:
                chunks = split_text_into_chunks(pg.get("page_content", ""))
                for c in chunks:
                    docs.append({"page_content": c, "metadata": pg.get("metadata", {})})
    if not docs:
        print("No documents to index. Exiting.")
        return
    index, idx_type = build_index(docs, embed_fn)
    print(f"Index built using: {idx_type}. {len(docs)} chunks indexed.")

    print("Enter your queries (type 'exit' to quit):")
    while True:
        q = input("Query> ")
        if not q or q.strip().lower() in ("exit", "quit"):
            break
        use_llm = bool(OPENAI_API_KEY)
        answer, top_docs = generate_answer(q, index, use_llm=use_llm)
        print("\n--- ANSWER ---")
        print(answer)
        print("\n--- TOP MATCHES ---")
        for d in top_docs:
            print(f"- {d.get('metadata', {}).get('source', 'unknown')} page {d.get('metadata', {}).get('page', '?')}")
        print("\n")

# ---------------------------
# Streamlit UI (if available)
# ---------------------------

def run_streamlit_app():
    import streamlit as st  # local import
    st.set_page_config(page_title="Resume RAG Bot", layout="wide")
    st.title("🔎 Resume Search — RAG Chatbot (LangChain + Streamlit)")
    st.caption("Upload bulk resume PDFs, then ask to find particular candidates or skills. The bot will search and answer using RAG.")

    with st.sidebar:
        st.header("Settings")
        st.markdown("- Upload one or more PDF resume files\n- Wait for indexing to finish, then ask questions in the chat box below.")
        st.markdown("\n**Embedding provider** (auto-detect):")
        st.write(EMBEDDINGS_PROVIDER)
        st.markdown("If you want to force provider, set EMBEDDINGS_PROVIDER env var to 'openai' or 'hf' before running.")

    uploaded_files = st.file_uploader("Upload resume PDF(s)", type=["pdf"], accept_multiple_files=True)

    if "index" not in st.session_state:
        st.session_state.index = None
        st.session_state.docs = []

    if uploaded_files:
        with st.spinner("Extracting text and building vectorstore — this may take a moment..."):
            embed_fn, provider = get_embedding_function()
            docs = []
            for uploaded_file in uploaded_files:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tfile.write(uploaded_file.read())
                tfile.flush()
                pages = load_pdf_pages(tfile.name)
                for pg in pages:
                    chunks = split_text_into_chunks(pg.get("page_content", ""))
                    for c in chunks:
                        docs.append({"page_content": c, "metadata": pg.get("metadata", {})})
            if not docs:
                st.error("No text extracted from uploaded PDFs.")
            else:
                index, idx_type = build_index(docs, embed_fn)
                st.session_state.index = index
                st.session_state.docs = docs
                st.success(f"Indexed {len(docs)} chunks using {idx_type}.")

    st.markdown("---")
    st.header("Ask the Resume Bot")
    query = st.text_input("Type your question (e.g. 'Find candidate John Doe' or 'Who has experience in React and Python?')")

    if st.button("Show indexed sources"):
        if not st.session_state.docs:
            st.info("No resumes indexed yet. Upload PDF(s) first.")
        else:
            sources = {}
            for d in st.session_state.docs:
                src = d.get("metadata", {}).get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
            st.write("Indexed chunks overview:")
            for s, cnt in sources.items():
                st.write(f"**{s}** — {cnt} chunks")

    if query:
        if not st.session_state.index:
            st.error("Please upload and index PDF resumes first.")
        else:
            use_llm = bool(OPENAI_API_KEY)
            answer, top_docs = generate_answer(query, st.session_state.index, use_llm=use_llm)
            st.markdown("**Answer:**")
            st.write(answer)
            if top_docs:
                st.markdown("**Top matching resume snippets (source — page):**")
                for s in top_docs:
                    src = s.get("metadata", {}).get("source", "unknown")
                    page = s.get("metadata", {}).get("page", "?")
                    snippet = s.get("page_content", "").replace("\n", " ")[:500]
                    st.write(f"- **{src}** — page {page}: {snippet}...")

    st.markdown("---")
    st.info("Tips: For better results, ask specific questions like 'Show resumes that mention Python and AWS' or 'Find John Smith'. The quality depends on how well the resumes are formatted and on the embeddings/LLM used.")

# ---------------------------
# Small tests
# ---------------------------

def test_extract_candidate_names():
    txt = """John Doe\nExperience: Python\n"""
    names = extract_candidate_names(txt)
    assert "John Doe" in names or names == ["John Doe"], f"Unexpected names: {names}"

    txt2 = "Full Name: Alice Wonder\nSkills: Java"
    names2 = extract_candidate_names(txt2)
    assert "Alice Wonder" in names2, f"Failed to detect 'Alice Wonder' -> {names2}"

    txt3 = "\n\nRandom text with no name" 
    names3 = extract_candidate_names(txt3)
    assert isinstance(names3, list), "Should return a list"
    print("test_extract_candidate_names passed")

# ---------------------------
# Entrypoint
# ---------------------------

def main(argv: List[str] = None):
    parser = argparse.ArgumentParser(description="Resume RAG Bot (Streamlit UI or CLI fallback)")
    parser.add_argument("--pdf", nargs="*", help="PDF resume files to index (CLI mode)")
    parser.add_argument("--run-tests", action="store_true", help="Run internal unit tests and exit")
    parser.add_argument("--no-streamlit", action="store_true", help="Force CLI mode even if streamlit is installed")
    args = parser.parse_args(argv)

    if args.run_tests:
        test_extract_candidate_names()
        print("All tests passed")
        return

    # If streamlit exists and user didn't force CLI, run the Streamlit app
    if STREAMLIT_AVAILABLE and not args.no_streamlit:
        # If the script is invoked by streamlit, Streamlit creates a separate process and will not run main here.
        run_streamlit_app()
        return

    # CLI mode
    print("Streamlit not available or CLI forced. Running in CLI mode.")
    if not args.pdf:
        print("No PDFs provided. You can pass PDF paths with --pdf file1.pdf file2.pdf or run without args to see demo data.")
    run_cli_mode(args.pdf or [])

if __name__ == "__main__":
    main(sys.argv[1:])
