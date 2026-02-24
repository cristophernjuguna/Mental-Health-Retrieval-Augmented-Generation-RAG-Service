# Mental-Health-Retrieval-Augmented-Generation-RAG-Service
This project implements a Retrieval-Augmented Generation (RAG) service designed to assist users with mental health inquiries. By leveraging a combination of natural language processing and database storage, the service provides accurate answers to user questions based on a curated set of mental health documents derived from PDFs.

Using FastAPI for the web framework and the SentenceTransformer library for generating text embeddings, the application efficiently retrieves relevant context and generates thoughtful responses through an integration with the Groq API.

Features
Dynamic Question Answering: Users can post questions and receive context-aware answers.
Document Ingestion: The service can read and process multiple PDF files, extracting and embedding their contents for efficient retrieval.
Database Storage: Uses ChromaDB for storing and querying document embeddings, allowing for rapid information retrieval.
API Interface: Built on FastAPI, providing endpoints for user interactions and easy integration into other applications.
Summary
The Mental Health RAG Service offers an innovative way to access mental health resources by providing direct answers to users' questions. By combining document processing, embeddings, and a conversational AI, this project enhances the accessibility of information in important areas of health and well-being.

1. requirements.txt
This file lists dependencies needed for the project:

fastapi: Web framework for building APIs.
uvicorn: ASGI server to run FastAPI applications.
sentence-transformers: Library to convert text to embeddings.
chromadb: Database for storing and querying embeddings.
groq: Likely used for generating responses from a context.
python-dotenv: For loading environment variables from a .env file.
pypdf: For extracting text from PDF files.
2. rag_service.py
Imports and Environment Setup
python
import os
import numpy as np
import chromadb
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

Run

os: Provides a way to interact with the operating system (for environment variables).
numpy: Library for numerical operations, specifically here for normalizing embeddings.
chromadb: Interacts with the Chroma database.
groq: Interacts with the Groq API for generating responses.
load_dotenv: Loads environment variables from a .env file.
SentenceTransformer: Loads a pre-trained model for text embedding.
python
load_dotenv()

Run

Loads environment variables from a .env file, making them accessible via os.getenv.
python
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

Run

Initializes a SentenceTransformer model for generating embeddings based on the "BAAI/bge-small-en-v1.5" model.
Function Definitions
python
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).tolist()  

Run

normalize_embeddings: Normalizes the embeddings so that each vector has a length of 1 (unit vector).
norms: Calculates the L2 norms (lengths) of the embedding vectors.
embeddings / norms: Divides each embedding by its norm.
.tolist(): Converts the result back to a list format.
python
def ask(question: str):

Run

ask: Accepts a question as input and generates an answer based on relevant documents.
python
    query_embedding = model.encode([question])
    query_embedding = normalize_embeddings(query_embedding)

Run

Generates an embedding for the input question and normalizes it.
python
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

Run

Queries the ChromaDB collection for the top 3 documents relevant to the question's embedding.
python
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]

Run

Extracts the relevant documents and their metadata from the query results.
python
    context = "\n\n".join(docs)

Run

Joins the relevant documents into a single context string, separating them by new lines.
python
    response = groq_client.chat.completions.create(
        ...
    )

Run

Uses the Groq API to generate a response based on the context and the user's question. The message contains system and user directives.
python
    answer = response.choices[0].message.content
    sources = list({meta["source"] for meta in metadatas})

Run

Extracts the answer from Groq's response and compiles a list of unique source document names.
python
    return answer, sources

Run

Returns the generated answer and the sources it was derived from.
3. main.py
Imports and Model Initialization
python
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb

Run

Same as rag_service.py, but additionally imports PdfReader for reading PDF files.
python
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

Run

Initializes the same model for generating text embeddings.
Function Definitions
python
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).tolist()

Run

Same normalization function as before.
python
def extract_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

Run

extract_text: Reads text from each page of a PDF.
PdfReader(path): Initializes a PDF reader for the given path.
Iterates through each page, extracting and combining text into a single string.
python
def chunk_text(text, size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunks.append(" ".join(words[i:i+size]))
    return chunks

Run

chunk_text: Splits the extracted text into chunks of a specified size (300 words here).
Database Interaction
python
client = chromadb.PersistentClient(path="./chroma_db")

Run

Initializes a persistent client to connect to the ChromaDB database.
python
collection = client.get_or_create_collection(
    name="mental_health_documents",
    metadata={"description": "Multi PDF documents"}
)

Run

Creates or retrieves a collection named "mental_health_documents" with a description.
python
PDF_FOLDER = "./documents"

Run

Specifies the folder containing PDF files to be processed.
Processing PDFs
python
all_chunks = []
all_embeddings = []
all_ids = []
all_metadatas = []

Run

Initializes lists to hold chunks of text, their embeddings, unique IDs, and metadata corresponding to each chunk.
python
doc_id_counter = 0

Run

Initializes a counter for unique document IDs.
python
for filename in os.listdir(PDF_FOLDER):

Run

Iterates through each file in the PDF_FOLDER directory.
python
if filename.endswith(".pdf"):
    ...

Run

Checks if the file is a PDF before processing it.
python
text = extract_text(file_path)
chunks = chunk_text(text)
embeddings = model.encode(chunks)
embeddings = normalize_embeddings(embeddings)

Run

Extracts text, breaks it into chunks, generates embeddings, and normalizes them.
python
for i, chunk in enumerate(chunks):
    all_chunks.append(chunk)
    all_embeddings.append(embeddings[i])
    all_ids.append(f"doc_{doc_id_counter}")
    all_metadatas.append({"source": filename})

Run

Appends each chunk, its embedding, a unique ID, and metadata into the respective lists.
python
doc_id_counter += 1

Run

Increments the document ID counter after processing each file.
python
collection.add(
    documents=all_chunks,
    embeddings=all_embeddings,
    ids=all_ids,
    metadatas=all_metadatas
)

Run

Adds all processed documents, embeddings, IDs, and metadata to the ChromaDB collection at once.
python
print("All PDFs Successfully Stored!")

Run

Confirms that all PDFs have been processed and stored.
4. ingest.py
This file has a similar structure and purpose as main.py, with minor variations in text processing. The explanations are largely identical to the main.py file, highlighting the same core functions and interactions.

5. app.py
Imports and API Setup
python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_service import ask

Run

Imports necessary libraries: FastAPI for the web framework, HTTPException for handling errors, and BaseModel for input/output validation.
Imports the ask function from rag_service.py.
python
app = FastAPI(title="Mental health RAG API")

Run

Initializes the FastAPI application with a title.
Request and Response Models
python
class QuestionRequest(BaseModel):
    question: str

Run

QuestionRequest: Defines the format for incoming requests, expecting a question string.
python
class QuestionResponse(BaseModel):
    answer: str
    sources: list

Run

QuestionResponse: Defines the format for outgoing responses, containing an answer and a list of sources.
Route Definitions
python
@app.get("/")
def root():
    return {"message": "RAG API is running"}

Run

Defines a root endpoint that returns a message confirming the API is running.
python

View all
@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    try:
        answer, sources = ask(request.question)
        
        return QuestionResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

Run

@app.post("/ask"): Defines an endpoint for asking a question.
Accepts a QuestionRequest, calls ask to get the answer and sources.
Returns a QuestionResponse with the result.
Uses a try-except block to handle exceptions, returning a 500 error if anything goes wrong.
6. .env
plaintext
GROQ_API_KEY="xxxxxxxxxxxxxxxxxxx"

Stores environment variables; specifically, the Groq API key for authentication securely.
