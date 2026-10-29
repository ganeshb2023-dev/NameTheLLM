# Name That LLM

Kaggle code competition: identify which of 18 frontier LLMs wrote a short passage (40–250 words).
Metric: **balanced accuracy**. Submission: `submission.parquet` with columns `id, model`.

## Approach (pure scikit-learn, runs offline)
- Character 1–5-gram TF-IDF (punctuation, spacing, formatting fingerprints)
- Word 1–3-gram TF-IDF with punctuation kept as tokens (vocabulary and phrasing)
- ~40 length-normalised stylometric rates (em-dashes, curly quotes, markdown, sentence stats)
- `LinearSVC(class_weight="balanced")` to match the balanced-accuracy metric

All logic is in `nameltm/core.py`. It auto-detects the train file (a file with a text column and a
`model`/`label` column) and `test.csv`.

## Layout
| Path | Purpose |
|---|---|
| `nameltm/core.py` | features, model, data loading, submission writer |
| `kaggle/name_that_llm.ipynb` | self-contained notebook to submit on Kaggle |
| `scripts/train.py` | local training / CV / submission |
| `app/` | Flask demo website (paste text, get predicted model) |
| `tests/test_smoke.py` | end-to-end test on synthetic data |

## Kaggle
1. New notebook in the competition, internet off.
2. Upload `kaggle/name_that_llm.ipynb` (File > Import) or paste its cells.
3. Run, then Submit. Set `CV = 5` in the notebook to see a local balanced-accuracy estimate first.

## Local
```bash
pip install -r requirements.txt
python scripts/train.py --root data/ --test data/test.csv --cv 5
python app/app.py        # http://localhost:5000
```

## Notes
- If the competition provides no labelled training file, the code stops with a clear message.
  You would then need your own labelled text from the 18 models.
- Score depends on the real data. Tune `C` (0.1–1.0) and `style_weight` using CV.
- Possible upgrades: add a pre-trained transformer embedding (attach as a Kaggle dataset) and blend with the linear model.

## Synthetic sample data (pipeline check only)
`python scripts/make_sample_data.py` creates `sample_data/` (train.csv, test.csv, test_answers.csv) with
18 fake "styles". It is NOT real LLM text and is far easier than the real task, so its ~100% score says
nothing about the real leaderboard. It only proves the pipeline runs end to end.
Replace it with the real Kaggle data: `python scripts/train.py --root <real data folder> --test <real test.csv> --cv 5`.

## Put the website online
1. Train on the real data so `models/model.joblib` exists (it is committed to git; keep it under 100 MB).
2. **Hugging Face Spaces** (free): New Space > SDK "Docker" > upload/push this repo. It builds from `Dockerfile` and serves on port 7860.
3. **Render** (free tier): New Web Service > connect the GitHub repo > Build `pip install -r requirements.txt`, Start `gunicorn app.app:app`.
