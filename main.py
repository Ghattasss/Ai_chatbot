from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import os

GOOGLE_API_KEY = os.getenv("AIzaSyAeaRCWyUSndTSU3KpC4onaB_uZUMNChpo")
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

data = pd.read_csv("chatbot_dataset_Final.csv")
questions = data["السؤال"].tolist()
answers = data["الإجابة"].tolist()

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

question_embeddings = model.encode(questions, convert_to_numpy=True)
question_embeddings = normalize(question_embeddings.astype(np.float32))

index = faiss.IndexFlatIP(question_embeddings.shape[1])
index.add(question_embeddings)

app = FastAPI()

class Question(BaseModel):
    text: str

# قراءة الكونتكست من ملف خارجي
with open("context.txt", "r", encoding="utf-8") as f:
    context = f.read()

@app.post("/ask")
def ask_question(q: Question):
    user_embedding = normalize(model.encode([q.text], convert_to_numpy=True).astype(np.float32))
    similarities, best_index = index.search(user_embedding, 1)
    best_match = best_index[0][0]
    similarity_score = similarities[0][0]

    if similarity_score > 0.8:
        return {"answer": answers[best_match], "source": "local"}
    else:
        prompt = f"{context}\n\nسؤال ولي الأمر: {q.text}\n\nيرجى الرد بشكل واضح وودود مع مراعاة شروط الرد."
        gemini_response = gemini_model.generate_content(prompt)
        return {"answer": gemini_response.text, "source": "gemini"}
