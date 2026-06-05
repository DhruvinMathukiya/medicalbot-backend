import io
import os
import json
import streamlit as st
from pypdf import PdfReader
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import pytesseract
from PIL import Image


# PAGE CONFIG

st.set_page_config(
    page_title="Medical AI Chatbot",
    page_icon="🩺",
    layout="wide"
)

st.title(" Medical AI Chatbot")
st.markdown("Upload a medical document and ask questions about it.")


# CONSTANTS

COLLECTION_NAME    = "medical_docs"
QDRANT_URL         = "http://localhost:6333"
CHAT_HISTORY_FILE  = "chat_history.json"
FILENAME_FILE      = "embedded_filename.txt"


# PERSISTENT STORAGE HELPERS

def load_chat_history():
    """Load chat history from disk. Returns empty list if not found."""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_chat_history(messages):
    """Save chat history to disk after every message."""
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(messages, f, indent=2)


def load_embedded_filename():
    """Load the last embedded filename from disk."""
    if os.path.exists(FILENAME_FILE):
        with open(FILENAME_FILE, "r") as f:
            return f.read().strip()
    return None


def save_embedded_filename(name):
    """Save the embedded filename to disk."""
    with open(FILENAME_FILE, "w") as f:
        f.write(name)


def clear_persistent_storage():
    """Wipe chat history and filename from disk (used on reset)."""
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)
    if os.path.exists(FILENAME_FILE):
        os.remove(FILENAME_FILE)


# LOAD EMBEDDING MODEL (cached — loads once)

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# QDRANT CLIENT (cached — connects once)

@st.cache_resource
def get_qdrant_client():
    try:
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()  # health-check
        return client
    except Exception:
        st.error(
            "❌ Qdrant is not running. "
            "Start it with:  docker-compose up -d"
        )
        st.stop()


# LOAD LLM (cached — loads once)

@st.cache_resource
def load_llm():
    return Ollama(
        model="llama3.2:3b",   
        temperature=0.3
    )


# ---------------------------------------------------------------
# PROMPT TEMPLATE
# ---------------------------------------------------------------
PROMPT_TEMPLATE = """
You are an intelligent and professional AI Medical Assistant with strong medical knowledge.

You are given:

1. Context retrieved from an uploaded medical document using semantic search
2. A user's question

Your job is to provide a clear, accurate, detailed, and helpful medical answer.

----------------------------------------
INSTRUCTIONS
----------------------------------------

1. USE DOCUMENT CONTEXT FIRST
- Carefully review the provided document context before answering.
- If the context is relevant to the question:
  - Answer using the information from the document.
  - Mention clearly that the answer is based on the uploaded document.

2. IF CONTEXT IS PARTIALLY RELEVANT
- If the document contains only part of the answer:
  - Use the document information first
  - Then complete the answer using reliable general medical knowledge
  - Clearly separate both when helpful

3. IF CONTEXT IS NOT RELEVANT
- Ignore the document context
- Answer from your general medical knowledge

4. DOCUMENT QUESTIONS
If the user asks:
- "What does this report say?"
- "Summarize this report"
- "What are the findings?"
- "Explain this document"

Then:
- summarize only what is present in the document
- mention key findings
- mention abnormal observations if available
- mention impression/conclusion if present
- do not add findings that are not written in the report

5. MEDICAL QUESTIONS
If the user asks about:
- symptoms
- causes
- diagnosis
- treatment
- precautions
- medicines
- tests
- recovery

Then provide:
- clear explanation
- possible causes
- common symptoms
- usual treatment options
- precautions or next steps if relevant

6. RESPONSE QUALITY
Your answer should be:
- medically accurate
- detailed but easy to understand
- professional and helpful
- well-structured
- natural sounding

7. IMPORTANT
- Never invent information not present in the document
- Never create fake findings
- Never say "I cannot help with that"
- Never refuse if general medical knowledge can answer it
- If information is missing in the document, say it is not mentioned clearly in the uploaded document
- If answering from general knowledge, state that clearly

----------------------------------------
DOCUMENT CONTEXT
----------------------------------------

{context}

----------------------------------------
USER QUESTION
----------------------------------------

{question}

----------------------------------------
RESPONSE FORMAT
----------------------------------------

Answer using this structure:

Source:
- Uploaded Document
OR
- General Medical Knowledge
OR
- Uploaded Document + General Medical Knowledge

Answer:
[Give a clear, detailed response here]

Notes:
[Optional clarification if needed]

Now provide the best possible answer.
"""


def build_prompt():
    return PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )


# ---------------------------------------------------------------
# STEP 1 — EXTRACT RAW TEXT FROM FILE
# ---------------------------------------------------------------
def extract_text_from_file(uploaded_file):
    """
    Returns raw text string from PDF or TXT.
    Three-layer strategy for PDFs:
      Layer 1 — pypdf     (fast, works on most text PDFs)
      Layer 2 — PyMuPDF   (catches more complex PDFs)
      Layer 3 — Tesseract (last resort for scanned/image PDFs)
    """
    try:
        # ── TXT ──────────────────────────────────────────────
        if uploaded_file.name.lower().endswith(".txt"):
            return uploaded_file.read().decode("utf-8").strip()

        # ── PDF ──────────────────────────────────────────────
        if uploaded_file.name.lower().endswith(".pdf"):

            # Read bytes ONCE so both pypdf and fitz can reuse them
            pdf_bytes = uploaded_file.read()

            # Layer 1: pypdf
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = "".join(
                page.extract_text() or "" for page in reader.pages
            )
            if text.strip():
                return text.strip()

            # Layer 2: PyMuPDF (better at complex layouts)
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            if text.strip():
                return text.strip()

            # Layer 3: OCR for scanned / image-only PDFs
            st.info("📄 No selectable text found — running OCR...")

            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if not os.path.exists(tesseract_path):
                st.error(
                    "❌ Tesseract is not installed. "
                    "Download from: "
                    "https://github.com/UB-Mannheim/tesseract/wiki"
                )
                return ""

            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
            ocr_text = ""
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes(
                    "RGB", [pix.width, pix.height], pix.samples
                )
                ocr_text += pytesseract.image_to_string(img) + "\n"
            doc.close()
            return ocr_text.strip()

    except Exception as e:
        st.error(f"File reading error: {str(e)}")
        return ""

    return ""


# ---------------------------------------------------------------
# STEP 2 — CHUNK THE TEXT
# ---------------------------------------------------------------
def chunk_text(text):
    """
    Splits raw text into overlapping chunks suitable for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------
# STEP 3 — EMBED CHUNKS AND STORE IN QDRANT
# Runs ONCE per uploaded document, not on every question.
# ---------------------------------------------------------------
def embed_and_store(chunks):
    """
    Embeds text chunks with HuggingFace and stores in Qdrant.
    Deletes existing collection first to avoid duplicate chunks.
    """
    embedding_model = load_embedding_model()
    client          = get_qdrant_client()

    # Wipe old collection so re-uploading starts fresh
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    # Embed all chunks and push to Qdrant in one shot
    QdrantVectorStore.from_texts(
        texts=chunks,
        embedding=embedding_model,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME
    )


# RECONNECT VECTORSTORE FROM QDRANT (on app restart)
# Vectors already exist in Qdrant — just reconnect, no re-embedding

def load_existing_vectorstore():
    """
    On app startup, reconnect to an existing Qdrant collection
    instead of re-embedding. Returns None if collection not found.
    """
    try:
        client = get_qdrant_client()
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in existing:
            return QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=load_embedding_model()
            )
    except Exception:
        pass
    return None


# ---------------------------------------------------------------
# STEP 4 — RETRIEVE RELEVANT CHUNKS FROM QDRANT
# ---------------------------------------------------------------
def retrieve_context(vectorstore, question, k=3):
    """
    Searches Qdrant for top-k chunks most similar to the question.
    Returns a single combined context string.
    """
    docs = vectorstore.similarity_search(question, k=k)
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------
# STEP 5 — GENERATE ANSWER FROM LLM
# ---------------------------------------------------------------
def generate_answer(context, question):
    """
    Formats the prompt with retrieved context and question,
    then calls the local Ollama LLM to generate an answer.
    """
    prompt    = build_prompt()
    llm       = load_llm()
    formatted = prompt.format(context=context, question=question)
    return llm.invoke(formatted)


# ---------------------------------------------------------------
# SESSION STATE INIT
# All three values are loaded from disk on startup — not RAM only
# ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_existing_vectorstore()

if "embedded_filename" not in st.session_state:
    st.session_state.embedded_filename = load_embedded_filename()


# ---------------------------------------------------------------
# SIDEBAR — DOCUMENT UPLOAD + EMBEDDING PIPELINE
# ---------------------------------------------------------------
with st.sidebar:

    # ─────────────────────────────────────────
    # App Title
    # ─────────────────────────────────────────
    st.markdown(
        """
        ## 🩺 MediBot
        Your AI Medical Assistant
        """
    )

    st.markdown("---")

    # ─────────────────────────────────────────
    # Upload Section
    # ─────────────────────────────────────────
    st.markdown("""
    <div style="
    padding:16px;
    border:1px solid #2E2E2E;
    border-radius:14px;
    background:#1E1E1E;
    text-align:center;
    margin-bottom:10px;
    ">
    <h4>📄 Upload Medical Report</h4>
    <p>Upload PDF or TXT file</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📄 Upload Medical Document")

    st.caption(
        "Upload a PDF or TXT report. "
        "The document will be embedded once and stored in Qdrant."
    )

    uploaded_file = st.file_uploader(
        "Choose file",
        type=["pdf", "txt"],
        label_visibility="collapsed"
    )

    # ─────────────────────────────────────────
    # File Processing
    # ─────────────────────────────────────────
    if uploaded_file:

        st.success(
            f"📎 **Selected:** {uploaded_file.name}"
        )

        if uploaded_file.name != st.session_state.embedded_filename:

            with st.spinner("🔄 Processing document..."):

                raw_text = extract_text_from_file(uploaded_file)

                if not raw_text:
                    st.error(
                        "❌ Could not read the file."
                    )

                else:
                    st.success(
                        f"✅ Extracted {len(raw_text):,} characters"
                    )

                    chunks = chunk_text(raw_text)

                    st.info(
                        f"📦 Created {len(chunks)} chunks"
                    )

                    embed_and_store(chunks)

                    st.success(
                        "🧠 Embedded into Qdrant successfully"
                    )

                    st.session_state.vectorstore = QdrantVectorStore(
                        client=get_qdrant_client(),
                        collection_name=COLLECTION_NAME,
                        embedding=load_embedding_model()
                    )

                    st.session_state.embedded_filename = uploaded_file.name
                    save_embedded_filename(uploaded_file.name)

                    st.session_state.messages = []
                    save_chat_history([])

        else:
            st.info(
                "✅ Document already available in Qdrant"
            )

    # ─────────────────────────────────────────
    # Active Document
    # ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📌 Active Document")

    if st.session_state.embedded_filename:

        st.success(
            f"📄 {st.session_state.embedded_filename}"
        )

    else:
        st.warning(
            "No document uploaded yet"
        )

    # ─────────────────────────────────────────
    # Quick Actions
    # ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑 Clear Chat"):
            st.session_state.messages = []
            save_chat_history([])
            st.rerun()

    with col2:
        if st.button("➕ New Chat"):
            st.session_state.messages = []
            save_chat_history([])
            st.rerun()

    # ─────────────────────────────────────────
    # Reset Everything
    # ─────────────────────────────────────────
    st.markdown("---")

    if st.button(
        "⚠️ Reset Everything",
        use_container_width=True
    ):

        try:
            client = get_qdrant_client()

            existing = [
                c.name
                for c in client.get_collections().collections
            ]

            if COLLECTION_NAME in existing:
                client.delete_collection(
                    COLLECTION_NAME
                )

        except Exception:
            pass

        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.embedded_filename = None

        clear_persistent_storage()

        st.success(
            "✅ Everything reset successfully"
        )

        st.rerun()

# ---------------------------------------------------------------
# MAIN CHAT AREA
# ---------------------------------------------------------------

# Replay chat history (loaded from disk on startup)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_question = st.chat_input("Ask a question about your document...")

if user_question:

    # Guard: require document to be embedded first
    if st.session_state.vectorstore is None:
        st.warning(
            "⚠️ Please upload a document first using the sidebar."
        )

    else:
        # Show and save user message
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate and show assistant response
        with st.chat_message("assistant"):
            with st.spinner(
                "🔍 Searching document and generating answer..."
            ):
                try:
                    # Step 4 — Retrieve relevant chunks from Qdrant
                    context = retrieve_context(
                        st.session_state.vectorstore,
                        user_question,
                        k=3
                    )

                    # Step 5 — LLM generates answer
                    # Always sent to LLM — uses context if relevant,
                    # falls back to general knowledge if not
                    answer = generate_answer(context, user_question)

                    st.markdown(answer)

                    # Save assistant message to session + disk
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    save_chat_history(st.session_state.messages)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    
                    