import json
import time
import random
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# app.py içindeki fonksiyonları kullanıyoruz
from app import ask_groq, ask_gemini, ask_perplexity

LABELED_PATH = Path("questions_labeled.csv")  # sizde bu dosya var
SEED = 42
N_PER_LEVEL = 10

# "Doğru" saymak için benzerlik eşiği (istersen sonra kalibre ederiz)
ANSWER_SIM_THRESHOLD = 0.55

def text_similarity(a: str, b: str) -> float:
    """Türkçe serbest metinler için char n-gram TF-IDF benzerliği (daha stabil)."""
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    X = vec.fit_transform([a, b])
    sim = cosine_similarity(X[0], X[1])[0, 0]
    return float(sim)

def pick_questions(df: pd.DataFrame) -> pd.DataFrame:
    rnd = random.Random(SEED)
    picked = []
    for level in ["kolay", "orta", "zor"]:
        subset = df[df["difficulty"] == level].copy()
        if len(subset) < N_PER_LEVEL:
            raise ValueError(f"'{level}' sınıfında yeterli soru yok: {len(subset)}")
        idxs = list(subset.index)
        rnd.shuffle(idxs)
        picked.append(df.loc[idxs[:N_PER_LEVEL]])
    return pd.concat(picked, ignore_index=True)

def safe_call(fn, question: str):
    start = time.time()
    ans = fn(question)
    ms = int((time.time() - start) * 1000)
    return ans, ms

def main():
    df = pd.read_csv(LABELED_PATH)
    required = {"question", "answer", "difficulty"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV kolonları eksik. Gerekli: {required}, mevcut: {set(df.columns)}")

    test_df = pick_questions(df)

    rows = []
    for i, r in test_df.iterrows():
        q = str(r["question"])
        gt = str(r["answer"])
        level = str(r["difficulty"])

        groq_ans, groq_ms = safe_call(ask_groq, q)
        gemini_ans, gemini_ms = safe_call(ask_gemini, q)
        poe_ans, poe_ms = safe_call(ask_perplexity, q)

        groq_sim = text_similarity(groq_ans, gt)
        gemini_sim = text_similarity(gemini_ans, gt)
        poe_sim = text_similarity(poe_ans, gt)

        rows.append({
            "difficulty": level,
            "question": q,
            "ground_truth": gt,

            "groq_answer": groq_ans,
            "groq_sim": groq_sim,
            "groq_correct": int(groq_sim >= ANSWER_SIM_THRESHOLD),
            "groq_latency_ms": groq_ms,

            "gemini_answer": gemini_ans,
            "gemini_sim": gemini_sim,
            "gemini_correct": int(gemini_sim >= ANSWER_SIM_THRESHOLD),
            "gemini_latency_ms": gemini_ms,

            "poe_answer": poe_ans,
            "poe_sim": poe_sim,
            "poe_correct": int(poe_sim >= ANSWER_SIM_THRESHOLD),
            "poe_latency_ms": poe_ms,
        })

        print(f"[{i+1:02d}/30] {level} | sims: groq={groq_sim:.3f} gemini={gemini_sim:.3f} poe={poe_sim:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv("results.csv", index=False, encoding="utf-8-sig")

    def summary_for(prefix: str):
        return {
            "accuracy_overall": float(out[f"{prefix}_correct"].mean()),
            "avg_similarity_overall": float(out[f"{prefix}_sim"].mean()),
            "accuracy_by_difficulty": {
                lvl: float(out[out["difficulty"] == lvl][f"{prefix}_correct"].mean())
                for lvl in ["kolay", "orta", "zor"]
            },
            "avg_similarity_by_difficulty": {
                lvl: float(out[out["difficulty"] == lvl][f"{prefix}_sim"].mean())
                for lvl in ["kolay", "orta", "zor"]
            },
            "avg_latency_ms": float(out[f"{prefix}_latency_ms"].mean()),
        }

    summary = {
        "seed": SEED,
        "n_per_level": N_PER_LEVEL,
        "answer_sim_threshold": ANSWER_SIM_THRESHOLD,
        "models": {
            "groq": summary_for("groq"),
            "gemini": summary_for("gemini"),
            "poe": summary_for("poe"),
        }
    }

    Path("summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n✅ results.csv ve summary.json üretildi.")

if __name__ == "__main__":
    main()
