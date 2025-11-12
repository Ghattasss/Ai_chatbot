from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import os

# Fixed: Use environment variable correctly
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Load data
data = pd.read_csv("chatbot_dataset_Final.csv")
questions = data["السؤال"].tolist()
answers = data["الإجابة"].tolist()

# Load sentence transformer model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# Create embeddings
print("Creating embeddings...")
question_embeddings = model.encode(questions, convert_to_numpy=True)
question_embeddings = normalize(question_embeddings.astype(np.float32))

# Create FAISS index
index = faiss.IndexFlatIP(question_embeddings.shape[1])
index.add(question_embeddings)
print(f"FAISS index created with {index.ntotal} vectors")

app = FastAPI()

class Question(BaseModel):
    text: str

# Read context from external file
try:
    with open("context.txt", "r", encoding="utf-8") as f:
        context = f.read()
except FileNotFoundError:
    print("Warning: context.txt not found, using empty context")
    context = ""

@app.get("/")
def root():
    return {"status": "healthy", "message": "Arabic Chatbot API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok", "embeddings_loaded": index.ntotal}

@app.post("/ask")
def ask_question(q: Question):
    # Encode user question
    user_embedding = normalize(model.encode([q.text], convert_to_numpy=True).astype(np.float32))
    
    # Search for best match
    similarities, best_index = index.search(user_embedding, 1)
    best_match = best_index[0][0]
    similarity_score = similarities[0][0]
    
    if similarity_score > 0.8:
        return {
            "answer": answers[best_match], 
            "source": "local",
            "similarity": float(similarity_score)
        }
    else:
        prompt = f"{context}\n\nسؤال ولي الأمر: {q.text}\n\nيرجى الرد بشكل واضح وودود مع مراعاة شروط الرد."
        gemini_response = gemini_model.generate_content(prompt)
        return {
            "answer": gemini_response.text, 
            "source": "gemini",
            "similarity": float(similarity_score)
        }