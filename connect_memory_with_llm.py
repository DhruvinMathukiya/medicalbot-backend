from pypdf import PdfReader
import os
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# -------------------------
# SETTINGS
# -------------------------

DB_FAISS_PATH = "vectorstore/db_faiss"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LOCAL_MODEL = "llama3.2:3b"


# -------------------------
# LOAD LOCAL LLM
# -------------------------

def load_llm():

    llm = Ollama(
        model=LOCAL_MODEL,
        temperature=0.3
    )

    return llm


# -------------------------
# PROMPT
# -------------------------

CUSTOM_PROMPT_TEMPLATE = """
You are a professional medical assistant.

Use ONLY the context below to answer the question.

Rules:
- Answer only from provided context
- If answer not found say:
  "I don't know based on the provided context."
- Be detailed
- Use bullet points if needed
- Keep response clear and structured

Context:
{context}

Question:
{question}

Answer:
"""


def set_custom_prompt():

    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return prompt


# -------------------------
# EMBEDDINGS
# -------------------------

def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return embeddings


# -------------------------
# LOAD FAISS
# -------------------------

def load_vectorstore():

    embeddings = load_embeddings()

if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"FAISS database not found at {DB_FAISS_PATH}"
    )

if not os.path.exists(DB_FAISS_PATH):
    raise FileNotFoundError(
        f"FAISS database not found at {DB_FAISS_PATH}. Run create_memory.py first."
    )

db = FAISS.load_local(
    DB_FAISS_PATH,

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

# -------------------------
# READ FILE
# -------------------------

def read_uploaded_file(file_path):

    if file_path.endswith(".pdf"):

        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

        return text

    elif file_path.endswith(".txt"):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()

    return ""


# -------------------------
# TEMP VECTORSTORE
# -------------------------

def create_temp_vectorstore(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    embeddings = load_embeddings()

    temp_db = FAISS.from_texts(
        chunks,
        embeddings
    )

    return temp_db


# -------------------------
# COMBINED SEARCH
# -------------------------

def retrieve_context(
    question,
    uploaded_file=None
):

    db = load_vectorstore()

    base_docs = db.similarity_search(
        question,
        k=3
    )

    all_docs = list(base_docs)

    if uploaded_file:

        uploaded_text = read_uploaded_file(
            uploaded_file
        )

        if uploaded_text.strip():

            temp_db = create_temp_vectorstore(
                uploaded_text
            )

            temp_docs = temp_db.similarity_search(
                question,
                k=3
            )

            all_docs.extend(temp_docs)

    context = "\n\n".join(
        doc.page_content
        for doc in all_docs
    )

    return context, all_docs


# -------------------------
# ASK QUESTION
# -------------------------

def ask_question(
    question,
    uploaded_file=None
):

    llm = load_llm()

    prompt = set_custom_prompt()

    context, docs = retrieve_context(
        question,
        uploaded_file
    )

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    result = llm.invoke(
        final_prompt
    )

    return result, docs


# -------------------------
# CLI
# -------------------------


    question = input(
        "Ask medical question: "
    )

    extra_file = input(
        "Upload file path (optional): "
    ).strip()

    if extra_file == "":
        extra_file = None

    answer, source_docs = ask_question(
        question,
        extra_file
    )

    print("\nANSWER:\n")
    print(answer)

    print("\nSOURCES:\n")

    for i, doc in enumerate(source_docs):

        print(
            f"\n[{i+1}] {doc.page_content[:300]}..."
        )