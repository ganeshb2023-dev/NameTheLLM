"""Website: paste a passage -> predicted LLM.   python app/app.py  (needs models/model.joblib)"""
import os, sys
import numpy as np
from flask import Flask, jsonify, render_template, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nameltm.core import load_model  # noqa: E402

app = Flask(__name__)
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT, "models", "model.joblib"))
_model = None


def model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


@app.route("/")
def index():
    return render_template("index.html", ready=os.path.exists(MODEL_PATH))


@app.route("/predict", methods=["POST"])
def predict():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if len(text.split()) < 5:
        return jsonify(error="Please paste at least a few sentences."), 400
    if not os.path.exists(MODEL_PATH):
        return jsonify(error="Model not trained yet. Run scripts/train.py first."), 503
    m = model()
    s = m.decision_function([text])[0]
    p = np.exp(s - s.max()); p /= p.sum()
    top = np.argsort(-p)[:5]
    return jsonify(prediction=str(m.classes_[top[0]]),
                   top=[{"model": str(m.classes_[i]), "score": round(float(p[i]), 4)} for i in top])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
