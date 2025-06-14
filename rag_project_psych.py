# rag_project_psych.py

from dotenv import load_dotenv
load_dotenv()

import os
import glob
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment or .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

# Load and split text files
def load_and_chunk_text(root_folder):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_chunks = []

    for filepath in glob.glob(f"{root_folder}/**/*.txt", recursive=True):
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
            chunks = splitter.split_text(raw_text)
            all_chunks.extend(chunks)

    return all_chunks

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB
chroma_client = chromadb.PersistentClient(path="./research_db", settings=Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="ml_publications", metadata={"hnsw:space": "cosine"})

# Load embed store documents
docs = load_and_chunk_text("data/wikipedia_psych_articles")
print(f"Loaded and chunked {len(docs)} text segments.")

if not docs:
    raise ValueError("No documents were loaded. Please check your folder path and contents.")

embeddings = embedding_model.encode(docs).tolist()
ids = [f"doc_{i}" for i in range(len(docs))]

collection.add(documents=docs, embeddings=embeddings, ids=ids)

# GeePeeTee
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OpenAI.api_key = OPENAI_API_KEY

while True:
    query = input("Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    query_embedding = embedding_model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    if not results["documents"]:
        print("No relevant documents found.")
        continue

    context = "\n---\n".join(results["documents"][0])
    prompt = f"Answer the question using the context below:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

    print("\nAnswer:", response.choices[0].message.content, "\n")
