import os
import numpy as np
import chromadb
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="mental_health_documents")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).tolist()    

def ask(question:str):
    query_embedding = model.encode([question])
    query_embedding = normalize_embeddings(query_embedding)
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    context = "\n\n".join(docs)
    
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role":"system", "content": "Answer only using provided context"},
            {"role" : "user", "content": f"content: \n{context}\n\nQuestion: {question}"}
        ],
        temperature=0
    )
    
    answer = response.choices[0].message.content
    
    sources = list({meta["source"] for meta in metadatas})
    
    return answer, sources