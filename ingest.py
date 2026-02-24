import os
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb

# Load embedding model once
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings/norms).tolist()

def extract_text(path):
    reader = PdfReader(path)
    text=""
    for page in reader.pages:
        text+=page.extract_text() + "\n"
    return text

def chunk_text(text, size=300):
    words = text.split()
    chunks=[]
    for i in range(0, len(words), size):
        chunks.append(" ".join(words[i:i+size]))
    return chunks

# create chroma client
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="mental_health_documents",
    metadata={"description":"Multi PDF MH documents"}
)

#Folder containing PDFs
PDF_FOLDER = "./documents"

all_chunks =[]
all_embeddings = []
all_ids = []
all_metadatas = []

doc_id_counter = 0

for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        file_path = os.path.join(PDF_FOLDER, filename)
        print(f"Processing: {filename}")

        text = extract_text(file_path)
        chunks = chunk_text(text)

        embeddings = model.encode(chunks)
        embeddings = normalize_embeddings(embeddings)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_embeddings.append(embeddings[i])
            all_ids.append(f"doc_{doc_id_counter}")
            all_metadatas.append({"source": filename})

            doc_id_counter +=1

# add everything at once
collection.add(
    documents=all_chunks,
    embeddings=all_embeddings,
    ids=all_ids,
    metadatas = all_metadatas
)     

print("All PDFs Succesfully Stored!")