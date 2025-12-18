from flask import Flask, render_template, request
import os
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# -------------------------------------------------------------
# ENV YÜKLEME VE API ANAHTARLARI
# -------------------------------------------------------------
load_dotenv()

# API Anahtarlarını yükle
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POE_API_KEY = os.getenv("POE_API_KEY")
# Eski HF anahtarı bu kodda kullanılmayacak
HF_API_KEY = os.getenv("HF_API_KEY")

print("DEBUG → GROQ:", GROQ_API_KEY is not None and len(GROQ_API_KEY) > 5)
print("DEBUG → GEMINI:", GEMINI_API_KEY is not None and len(GEMINI_API_KEY) > 5)
print("DEBUG → POE:", POE_API_KEY is not None and len(POE_API_KEY) > 5)

# -------------------------------------------------------------
# Dataset Yükleme ve Hazırlama
# -------------------------------------------------------------
CSV_PATH = "questions.csv"
SIMILARITY_THRESHOLD = 0.60

try:
    df = pd.read_csv(CSV_PATH)
    questions = df["question"].astype(str).tolist()
    answers = df["answer"].astype(str).tolist()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
    print("Dataset başarıyla yüklendi ve vektörleştirildi.")
except FileNotFoundError:
    print(f"HATA: {CSV_PATH} dosyası bulunamadı. Dataset fonksiyonları çalışmayacaktır.")
    questions = []
    answers = []
except Exception as e:
    print(f"Dataset yükleme veya vektörleştirme hatası: {e}")
    questions = []
    answers = []


def dataset_lookup(user_question):
    if not questions:
        return {"answer": None, "matched": "N/A", "similarity": 0.0, "found": False}

    user_vec = vectorizer.transform([user_question])
    sims = cosine_similarity(user_vec, tfidf_matrix)[0]

    best_idx = sims.argmax()
    best_score = float(sims[best_idx])

    if best_score >= SIMILARITY_THRESHOLD:
        return {
            "answer": answers[best_idx],
            "matched": questions[best_idx],
            "similarity": best_score,
            "found": True
        }
    else:
        return {
            "answer": None,
            "matched": questions[best_idx],
            "similarity": best_score,
            "found": False
        }


# -------------------------------------------------------------
# GROQ – Llama 3.3 70B (Stabil)
# -------------------------------------------------------------
def ask_groq(user_question):
    if not GROQ_API_KEY:
        return "Groq API Anahtarı eksik. Lütfen .env dosyanızı kontrol edin."

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "Sen eğitim amaçlı bir karaciğer hastalıkları asistanısın. Türkçe yanıt ver."},
            {"role": "user", "content": user_question}
        ],
        "max_tokens": 400,
        "temperature": 0.3
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=25)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("Groq hatası:", e)
        return "Groq yanıtında hata oluştu. Detaylar için konsola bakın."


# -------------------------------------------------------------
# GEMINI – GÜNCEL MODEL (Stabil)
# -------------------------------------------------------------
def ask_gemini(user_question):
    if not GEMINI_API_KEY:
        return "Gemini API Anahtarı eksik. Lütfen .env dosyanızı kontrol edin."

    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent"

    payload = {
        "contents": [
            {"parts": [{"text": user_question}]}
        ]
    }

    try:
        r = requests.post(
            url,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=20
        )
        r.raise_for_status()
        data = r.json()

        if data.get("candidates") and data["candidates"][0]["content"]["parts"][0].get("text"):
            return data["candidates"][0]["content"]["parts"][0]["text"]

        return "Gemini yanıt üretemedi."

    except Exception as e:
        print("Gemini hatası:", e)
        return "Gemini yanıtında hata oluştu. Detaylar için konsola bakın."


# -------------------------------------------------------------
# POE – GPT-3.5-Turbo (HuggingFace yerine kullanılan stabil model)
# -------------------------------------------------------------
def ask_perplexity(user_question):
    if not POE_API_KEY:
        return "Poe API Anahtarı eksik. Lütfen .env dosyanızı kontrol edin."

    # Poe API'nin OpenAI uyumlu uç noktası
    url = "https://api.poe.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {POE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        # DeepSeek yerine Poe'daki en stabil ve yaygın kullanılan model olan GPT-3.5-Turbo kullanıldı.
        "model": "GPT-3.5-Turbo",
        "messages": [
            {"role": "system", "content": "Sen eğitim amaçlı bir karaciğer hastalıkları asistanısın. Türkçe yanıt ver."},
            {"role": "user", "content": user_question}
        ],
        "max_tokens": 400,
        "temperature": 0.3
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=25)
        r.raise_for_status()
        # Yanıt yapısı OpenAI stiline benzer olacaktır
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("Poe hatası:", e)
        return "Poe/GPT-3.5-Turbo yanıtında hata oluştu. Poe'daki puanlarınızın yeterli olduğunu kontrol edin."


# -------------------------------------------------------------
# FLASK UYGULAMASI
# -------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    user_question = None
    ds = None
    model_results = []

    if request.method == "POST":
        user_question = request.form.get("question", "").strip()

        if user_question:
            ds = dataset_lookup(user_question)

            # Model isimlerini güncelledik
            GROQ_MODEL = "Llama 3.3 70B"
            GEMINI_MODEL = "Gemini 2.5 Flash"
            POE_MODEL = "Poe/GPT-3.5-T" # Poe modeli GPT-3.5-T botunu gösteriyor

            if ds["found"]:
                model_results = [
                    {"name": f"Groq – {GROQ_MODEL}", "provider": "Groq", "source": "dataset", "answer": ds["answer"]},
                    {"name": f"Gemini – {GEMINI_MODEL}", "provider": "Google", "source": "dataset", "answer": ds["answer"]},
                    {"name": f"Poe – {POE_MODEL}", "provider": "Poe", "source": "dataset", "answer": ds["answer"]},
                ]

            else:
                # -------------------------------------------------------------
                # Dataset yetersiz → Üç modeli API’den çağır
                # -------------------------------------------------------------
                groq_ans = ask_groq(user_question)
                gemini_ans = ask_gemini(user_question)
                # Poe (GPT-3.5-Turbo) fonksiyonunu çağırıyoruz
                hf_ans = ask_perplexity(user_question)

                model_results = [
                    {"name": f"Groq – {GROQ_MODEL}", "provider": "Groq", "source": "llm", "answer": groq_ans},
                    {"name": f"Gemini – {GEMINI_MODEL}", "provider": "Google", "source": "llm", "answer": gemini_ans},
                    {"name": f"Poe – {POE_MODEL}", "provider": "Poe", "source": "llm", "answer": hf_ans},
                ]

    return render_template(
        "index.html",
        user_question=user_question,
        similarity_info=ds,
        threshold=SIMILARITY_THRESHOLD,
        model_results=model_results
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)