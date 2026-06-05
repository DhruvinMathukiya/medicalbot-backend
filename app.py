from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
import os
import shutil
import json
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama


# -------------------------
# SETTINGS
# -------------------------

APP = FastAPI()

APP.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=False,
	allow_methods=["*"],
	allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_MODEL = "llama3.2:3b"
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
FILENAME_FILE = os.path.join(BASE_DIR, "embedded_filename.txt")


# -------------------------
# HELPERS: persistence
# -------------------------

def load_chat_history():
	if os.path.exists(CHAT_HISTORY_FILE):
		with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
			return json.load(f)
	return []


def save_chat_history(messages):
	with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
		json.dump(messages, f, indent=2)


def load_embedded_filename():
	if os.path.exists(FILENAME_FILE):
		with open(FILENAME_FILE, "r", encoding="utf-8") as f:
			return f.read().strip()
	return None


def save_embedded_filename(name: str):
	with open(FILENAME_FILE, "w", encoding="utf-8") as f:
		f.write(name)


def clear_persistent_storage():
	if os.path.exists(CHAT_HISTORY_FILE):
		os.remove(CHAT_HISTORY_FILE)
	if os.path.exists(FILENAME_FILE):
		os.remove(FILENAME_FILE)
	if os.path.exists(DB_FAISS_PATH):
		shutil.rmtree(DB_FAISS_PATH)


# -------------------------
# MODELS / VECTORSTORE
# -------------------------

def load_embeddings():
	return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_llm():
	return Ollama(model=LOCAL_MODEL, temperature=0.3)


def chunk_text(text: str):
	splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
	return splitter.split_text(text)


def extract_text_from_file(path: str) -> str:
	if path.lower().endswith(".txt"):
		with open(path, "r", encoding="utf-8") as f:
			return f.read()

	if path.lower().endswith(".pdf"):
		reader = PdfReader(path)
		text = ""
		for page in reader.pages:
			page_text = page.extract_text()
			if page_text:
				text += page_text + "\n"
		return text

	return ""


PROMPT_TEMPLATE = """
You are an intelligent and professional AI Medical Assistant with strong medical knowledge.

You are given:

1. Context retrieved from an uploaded medical document using semantic search
2. A user's question

Answer clearly and concisely. When using document content, mention it.

Context:
{context}

Question:
{question}
"""


def generate_answer(context: str, question: str) -> str:
	prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
	llm = load_llm()
	formatted = prompt.format(context=context, question=question)
	return llm.invoke(formatted)


def embed_and_store(text_chunks):
	embeddings = load_embeddings()
	db = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
	os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
	db.save_local(DB_FAISS_PATH)
	return db


def load_existing_vectorstore():
	if os.path.exists(DB_FAISS_PATH):
		embeddings = load_embeddings()
		# FAISS deserialization uses pickle; set allow_dangerous_deserialization=True
		# only if you trust the stored files (they were created by you)
		db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
		return db
	return None


def retrieve_context(db, question: str, k: int = 3) -> str:
	docs = db.similarity_search(question, k=k)
	return "\n\n".join(doc.page_content for doc in docs), docs


# -------------------------
# API endpoints
# -------------------------


@APP.post("/upload")
async def upload_file(file: UploadFile = File(...)):
	filename = file.filename
	dest = os.path.join(UPLOAD_DIR, filename)
	with open(dest, "wb") as f:
		content = await file.read()
		f.write(content)

	text = extract_text_from_file(dest)
	if not text.strip():
		raise HTTPException(status_code=400, detail="No text extracted from file")

	chunks = chunk_text(text)
	embed_and_store(chunks)
	save_embedded_filename(filename)
	# reset chat history when new doc uploaded
	save_chat_history([])

	return {"status": "ok", "filename": filename, "chunks": len(chunks)}


@APP.post("/ask")
def ask_question(payload: dict):
	question = payload.get("question")
	if not question:
		raise HTTPException(status_code=400, detail="Missing question")

	db = load_existing_vectorstore()
	if db is None:
		raise HTTPException(status_code=400, detail="No vectorstore found. Upload a document first.")

	context, docs = retrieve_context(db, question, k=3)
	answer = generate_answer(context, question)

	# persist chat
	messages = load_chat_history()
	messages.append({"role": "user", "content": question})
	messages.append({"role": "assistant", "content": answer})
	save_chat_history(messages)

	sources = [d.page_content for d in docs]

	return {"answer": answer, "sources": sources}


@APP.post("/reset")
def reset_all():
	clear_persistent_storage()
	return {"status": "reset"}


@APP.get("/status")
def status():
	filename = load_embedded_filename()
	db_exists = os.path.exists(DB_FAISS_PATH)
	messages = load_chat_history()
	return {"embedded_filename": filename, "vectorstore_present": db_exists, "chat_count": len(messages)}


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("app:APP", host="0.0.0.0", port=8000, reload=True)
