from flask import Flask, render_template, request
import os
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# -------------------------------------------------------------
# ENV YÜKLEME
# -------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POE_API_KEY = os.getenv("POE_API_KEY")

print("DEBUG → GROQ:", bool(GROQ_API_KEY))
print("DEBUG → GEMINI:", bool(GEMINI_API_KEY))
print("DEBUG → POE:", bool(POE_API_KEY))

# -------------------------------------------------------------
# DATASET YÜKLEME
# -------------------------------------------------------------
CSV_PATH = "questions.csv"
SIMILARITY_THRESHOLD = 0.90  # EŞİK

try:
    df = pd.read_csv(CSV_PATH)
    questions = df["question"].astype(str).tolist()
    answers = df["answer"].astype(str).tolist()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)

    print("Dataset başarıyla yüklendi ve vektörleştirildi.")
except Exception as e:
    print("Dataset yüklenemedi:", e)
    questions, answers = [], []

# -------------------------------------------------------------
# DATASET BENZERLİK FONKSİYONU
# -------------------------------------------------------------
def dataset_lookup(user_question):
    if not questions:
        return {
            "found": False,
            "answer": None,
            "similarity": 0.0,
            "matched": None
        }

    user_vec = vectorizer.transform([user_question])
    sims = cosine_similarity(user_vec, tfidf_matrix)[0]

    best_idx = sims.argmax()
    best_score = float(sims[best_idx])

    if best_score >= SIMILARITY_THRESHOLD:
        return {
            "found": True,
            "answer": answers[best_idx],
            "similarity": best_score,
            "matched": questions[best_idx]
        }

    return {
        "found": False,
        "answer": None,
        "similarity": best_score,
        "matched": questions[best_idx]
    }

# -------------------------------------------------------------
# CEVAP BENZERLİĞİ HESAPLAMA
# -------------------------------------------------------------
def calculate_answer_similarity(answer1: str, answer2: str) -> float:
    """İki cevap arasındaki benzerliği TF-IDF ile hesapla"""
    if not answer1 or not answer2:
        return 0.0
    
    try:
        vec = TfidfVectorizer()
        X = vec.fit_transform([answer1, answer2])
        similarity = float(cosine_similarity(X[0], X[1])[0, 0])
        return round(similarity * 100, 2)  # % cinsinden
    except:
        return 0.0

# -------------------------------------------------------------
# GROQ
# -------------------------------------------------------------
def ask_groq(question):
    if not GROQ_API_KEY:
        return "Groq API anahtarı yok."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "Türkçe yanıt ver. Eğitim amaçlı yanıt üret."},
            {"role": "user", "content": question}
        ],
        "temperature": 0.3,
        "max_tokens": 400
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=25)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("Groq hatası:", e)
        return "Groq yanıtı alınamadı."

# -------------------------------------------------------------
# GEMINI
# -------------------------------------------------------------
def ask_gemini(question):
    if not GEMINI_API_KEY:
        return "Gemini API anahtarı yok."

    model = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"

    payload = {
        "contents": [{"parts": [{"text": question}]}]
    }

    try:
        r = requests.post(url, params={"key": GEMINI_API_KEY}, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()

        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print("Gemini hatası:", e)
        return "Gemini yanıtı alınamadı."

# -------------------------------------------------------------
# POE
# -------------------------------------------------------------
def ask_perplexity(question):
    if not POE_API_KEY:
        return "Poe API anahtarı yok."

    url = "https://api.poe.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {POE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "GPT-3.5-Turbo",
        "messages": [
            {"role": "system", "content": "Türkçe yanıt ver. Eğitim amaçlı yanıt üret."},
            {"role": "user", "content": question}
        ],
        "temperature": 0.3,
        "max_tokens": 400
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=25)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("Poe hatası:", e)
        return "Poe yanıtı alınamadı."

# -------------------------------------------------------------
# FLASK APP
# -------------------------------------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    user_question = None
    similarity_info = None
    model_results = []

    if request.method == "POST":
        user_question = request.form.get("question", "").strip()

        if user_question:
            similarity_info = dataset_lookup(user_question)

            # 🔴 HER ZAMAN AI CEVAPLARINI AL
            groq_answer = ask_groq(user_question)
            gemini_answer = ask_gemini(user_question)
            poe_answer = ask_perplexity(user_question)

            # DOĞRULUK ORANI HESAPLA
            if similarity_info["found"] and similarity_info["answer"]:
                # Veri seti cevabı varsa, AI cevaplarıyla karşılaştır
                dataset_answer = similarity_info["answer"]
                groq_accuracy = calculate_answer_similarity(groq_answer, dataset_answer)
                gemini_accuracy = calculate_answer_similarity(gemini_answer, dataset_answer)
                poe_accuracy = calculate_answer_similarity(poe_answer, dataset_answer)
            else:
                # Veri seti cevabı yoksa, doğruluk = 0
                groq_accuracy = 0.0
                gemini_accuracy = 0.0
                poe_accuracy = 0.0

            model_results = [
                {
                    "name": "Groq – Llama 3.3 70B",
                    "provider": "Groq",
                    "answer": groq_answer,
                    "accuracy": groq_accuracy
                },
                {
                    "name": "Gemini – Gemini 2.5 Flash",
                    "provider": "Google",
                    "answer": gemini_answer,
                    "accuracy": gemini_accuracy
                },
                {
                    "name": "Poe – GPT-3.5-T",
                    "provider": "Poe",
                    "answer": poe_answer,
                    "accuracy": poe_accuracy
                }
            ]

    return render_template(
        "index.html",
        user_question=user_question,
        similarity_info=similarity_info,
        threshold=SIMILARITY_THRESHOLD,
        model_results=model_results
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
