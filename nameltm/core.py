"""Name That LLM - stylometric authorship attribution (18 classes).

Single self-contained module (pure scikit-learn, works offline on Kaggle):
  * char 1-5 gram TF-IDF  (punctuation / spacing / formatting fingerprints)
  * word 1-3 gram TF-IDF  (vocabulary / phrasing fingerprints, punctuation kept as tokens)
  * hand-made stylometric rates (em-dashes, curly quotes, markdown, sentence stats ...)
  * LinearSVC with balanced class weights (metric = balanced accuracy)
"""
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
LABEL_COLS = ["model", "label", "target", "author", "generator", "source_model"]
TEXT_COLS = ["text", "passage", "response", "completion", "output", "content", "answer"]
TEST_PATH = "/kaggle/input/competitions/name-that-llm/test.csv"


# ----------------------------------------------------------------------------- features
class StyleFeatures(BaseEstimator, TransformerMixin):
    """Length-normalised stylometric rates (no raw length shortcuts)."""

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _row(t):
        t = t if isinstance(t, str) else ""
        n = max(len(t), 1)
        words = re.findall(r"\w+(?:['’]\w+)?", t)
        nw = max(len(words), 1)
        sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", t) if s.strip()]
        sl = np.array([len(re.findall(r"\w+", s)) for s in sents] or [0], dtype=float)
        lines = t.split("\n")
        nl = max(len(lines), 1)
        c = t.count
        per_k = lambda x: 1000.0 * x / n
        return [
            np.mean([len(w) for w in words]) if words else 0.0,
            sl.mean(), sl.std(), sl.max() / nw,
            len(set(w.lower() for w in words)) / nw,
            per_k(c(",")), per_k(c("—")), per_k(c("–")), per_k(c(" - ")), per_k(c("-")),
            per_k(c(";")), per_k(c(":")), per_k(c("(")), per_k(c('"')),
            per_k(c("“") + c("”")), per_k(c("'")), per_k(c("’")), per_k(c("…") + c("...")),
            per_k(c("!")), per_k(c("?")), per_k(c("*")), per_k(c("**")), per_k(c("#")),
            per_k(c("`")), per_k(c("$")), per_k(c("\\")), per_k(c("_")), per_k(c("/")),
            per_k(sum(ch.isdigit() for ch in t)),
            per_k(sum(ch.isupper() for ch in t)),
            per_k(sum(ord(ch) > 127 for ch in t)),
            per_k(c("\n")), per_k(c("\n\n")),
            sum(bool(re.match(r"\s*([-*•]|\d+[.)])\s", ln)) for ln in lines) / nl,
            float(t.rstrip().endswith((".", "!", "?", '"', "”"))),
            float(t[:1].isupper()),
            per_k(len(re.findall(r"\b(I|me|my)\b", t))),
            per_k(len(re.findall(r"\b(you|your)\b", t, flags=re.I))),
            per_k(len(re.findall(r"\b(we|our|us)\b", t, flags=re.I))),
        ]

    def transform(self, X):
        return np.nan_to_num(np.asarray([self._row(t) for t in X], dtype=np.float32))


def build_pipeline(C=0.3, style_weight=0.5):
    feats = FeatureUnion(
        [
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False,
                                     sublinear_tf=True, min_df=2, max_features=600_000,
                                     dtype=np.float32)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 3), lowercase=False,
                                     token_pattern=r"\w+|[^\w\s]", sublinear_tf=True,
                                     min_df=2, max_features=400_000, dtype=np.float32)),
            ("style", Pipeline([("f", StyleFeatures()), ("s", StandardScaler())])),
        ],
        transformer_weights={"char": 1.0, "word": 1.0, "style": style_weight},
    )
    return Pipeline([("feats", feats),
                     ("clf", LinearSVC(C=C, class_weight="balanced"))])


# ----------------------------------------------------------------------------- data
def _read_any(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith((".jsonl", ".json")):
        return pd.read_json(path, lines=path.endswith(".jsonl"))
    return pd.read_csv(path)


def pick_columns(df):
    low = {c.lower(): c for c in df.columns}
    label = next((low[c] for c in LABEL_COLS if c in low), None)
    text = next((low[c] for c in TEXT_COLS if c in low), None)
    if text is None:
        objs = [c for c in df.columns if df[c].dtype == object and c != label and c.lower() != "id"]
        if objs:
            text = max(objs, key=lambda c: df[c].astype(str).str.len().mean())
    return text, label


def find_data(root="/kaggle/input", test_path=None):
    """Return ((train_df, text_col, label_col, path) | None, (test_df, path) | None)."""
    if test_path is None and os.path.exists(TEST_PATH):
        test_path = TEST_PATH
    files = [f for ext in ("csv", "parquet", "jsonl") for f in
             glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)]
    train, test = None, None
    for f in sorted(files):
        if test_path and os.path.abspath(f) == os.path.abspath(test_path):
            continue
        try:
            df = _read_any(f)
        except Exception:
            continue
        text, label = pick_columns(df)
        if text and label and df[label].nunique() > 1 and len(df) > 50:
            if train is None or len(df) > len(train[0]):
                train = (df, text, label, f)
        elif text and os.path.basename(f).startswith("test") and test is None:
            test = (df, f)
    if test_path:
        test = (_read_any(test_path), test_path)
    return train, test


# ----------------------------------------------------------------------------- run
def cross_validate(df, text, label, folds=5, **kw):
    skf = StratifiedKFold(folds, shuffle=True, random_state=0)
    scores = []
    for i, (tr, va) in enumerate(skf.split(df[text], df[label])):
        p = build_pipeline(**kw).fit(df[text].iloc[tr], df[label].iloc[tr])
        s = balanced_accuracy_score(df[label].iloc[va], p.predict(df[text].iloc[va]))
        scores.append(s)
        print(f"fold {i}: balanced acc = {s:.4f}")
    print(f"CV balanced accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return scores


def save_submission(ids, preds, path="submission.parquet"):
    sub = pd.DataFrame({"id": ids, "model": preds})
    assert sub["id"].is_unique and sub["model"].notna().all(), "invalid submission"
    bad = set(sub["model"]) - set(LABELS)
    if bad:
        warnings.warn(f"Labels outside the 18 valid names: {bad}")
    try:
        sub.to_parquet(path, index=False)
    except ImportError:
        path = path.replace(".parquet", ".csv")
        sub.to_csv(path, index=False)
        warnings.warn("pyarrow missing - wrote CSV instead")
    print("wrote", path, sub.shape)
    return sub


def run(root="/kaggle/input", test_path=None, out="submission.parquet",
        model_path="models/model.joblib", cv=0, C=0.3):
    train, test = find_data(root, test_path)
    if train is None:
        raise FileNotFoundError(
            "No labelled training file found under %s. Add the competition's train file "
            "(needs a text column and a model/label column) or attach it as a dataset." % root)
    df, text, label, f = train
    print(f"train file: {f} | rows={len(df)} | text='{text}' label='{label}'")
    df = df.dropna(subset=[text, label])
    if cv:
        cross_validate(df, text, label, folds=cv, C=C)
    pipe = build_pipeline(C=C).fit(df[text].astype(str), df[label])
    try:
        import joblib
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        joblib.dump(pipe, model_path, compress=3)
    except Exception as e:  # saving is optional
        print("model not saved:", e)
    if test is None:
        print("no test file found - model trained only")
        return pipe
    tdf = test[0]
    ttext, _ = pick_columns(tdf)
    preds = pipe.predict(tdf[ttext].astype(str))
    save_submission(tdf["id"], preds, out)
    return pipe


def load_model(path="models/model.joblib"):
    import joblib
    return joblib.load(path)
