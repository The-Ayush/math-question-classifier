"""Step 2: Classify maths problems by topic and predict difficulty.

Compares a TF-IDF baseline against sentence-embedding features.
Run from the project root:  python src/02_train.py
Writes results/topic_metrics.csv, results/difficulty_metrics.csv and figures/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             mean_absolute_error)
from sklearn.svm import LinearSVC

SEED = 42
FIG, RES = Path("figures"), Path("results")
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
print(f"Train: {len(train)}  Test: {len(test)}")

# ---------- Features ----------
# Word n-grams capture vocabulary ("probability", "triangle"); character n-grams
# capture LaTeX patterns (\sin, \binom, \pmod) that tokenisers often split badly.
word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                           sublinear_tf=True, max_features=200_000)
X_tr_tfidf = hstack([word_vec.fit_transform(train.problem), char_vec.fit_transform(train.problem)]).tocsr()
X_te_tfidf = hstack([word_vec.transform(test.problem), char_vec.transform(test.problem)]).tocsr()

emb_cache = Path("data/embeddings.npz")
if emb_cache.exists():
    z = np.load(emb_cache)
    E_tr, E_te = z["train"], z["test"]
else:
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    E_tr = encoder.encode(train.problem.tolist(), batch_size=64, show_progress_bar=True,
                          normalize_embeddings=True)
    E_te = encoder.encode(test.problem.tolist(), batch_size=64, show_progress_bar=True,
                          normalize_embeddings=True)
    np.savez(emb_cache, train=E_tr, test=E_te)

# ---------- Task 1: topic classification ----------
topic_models = {
    "TF-IDF + Logistic Regression": (LogisticRegression(max_iter=2000, C=10), X_tr_tfidf, X_te_tfidf),
    "TF-IDF + Linear SVM": (LinearSVC(C=0.5, random_state=SEED), X_tr_tfidf, X_te_tfidf),
    "MiniLM embeddings + Logistic Regression": (LogisticRegression(max_iter=2000, C=10), E_tr, E_te),
}
rows, preds = [], {}
for name, (model, Xtr, Xte) in topic_models.items():
    model.fit(Xtr, train.type)
    p = model.predict(Xte)
    preds[name] = p
    rows.append({"model": name,
                 "accuracy": accuracy_score(test.type, p),
                 "macro_f1": f1_score(test.type, p, average="macro")})
topic = pd.DataFrame(rows).round(3).sort_values("macro_f1", ascending=False)
topic.to_csv(RES / "topic_metrics.csv", index=False)
print("\nTopic classification (held-out test set):\n", topic.to_string(index=False))

best = topic.iloc[0]["model"]
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay.from_predictions(test.type, preds[best], ax=ax, cmap="Blues",
                                        normalize="true", values_format=".2f",
                                        xticks_rotation=45)
ax.set_title(f"Topic confusion matrix - {best}")
plt.tight_layout()
plt.savefig(FIG / "topic_confusion_matrix.png", dpi=150)
plt.close()

per_class = pd.Series(f1_score(test.type, preds[best], average=None,
                               labels=sorted(test.type.unique())),
                      index=sorted(test.type.unique())).round(3)
print("\nPer-topic F1 (best model):\n", per_class.sort_values().to_string())

# ---------- Task 2: difficulty (level 1-5) ----------
# Difficulty is ordinal, so we treat it as regression and round to the nearest level.
diff_rows = []
baseline = np.full(len(test), int(round(train.level.mean())))
for name, Xtr, Xte in [("TF-IDF + Ridge", X_tr_tfidf, X_te_tfidf),
                       ("MiniLM embeddings + Ridge", E_tr, E_te)]:
    reg = Ridge(alpha=1.0).fit(Xtr, train.level)
    p = np.clip(np.rint(reg.predict(Xte)), 1, 5)
    diff_rows.append({"model": name,
                      "MAE": mean_absolute_error(test.level, p),
                      "exact_acc": accuracy_score(test.level, p),
                      "within_1_level": np.mean(np.abs(p - test.level) <= 1)})
diff_rows.append({"model": "Baseline (always predict mean level)",
                  "MAE": mean_absolute_error(test.level, baseline),
                  "exact_acc": accuracy_score(test.level, baseline),
                  "within_1_level": np.mean(np.abs(baseline - test.level) <= 1)})
diff = pd.DataFrame(diff_rows).round(3)
diff.to_csv(RES / "difficulty_metrics.csv", index=False)
print("\nDifficulty prediction (held-out test set):\n", diff.to_string(index=False))
