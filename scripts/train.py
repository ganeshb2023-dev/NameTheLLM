"""python scripts/train.py --root data/ --test data/test.csv --cv 5"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nameltm.core import run

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/kaggle/input", help="folder containing train file(s)")
ap.add_argument("--test", default=None, help="path to test.csv")
ap.add_argument("--out", default="submission.parquet")
ap.add_argument("--model", default="models/model.joblib")
ap.add_argument("--cv", type=int, default=0, help="k-fold CV (0 = skip)")
ap.add_argument("--C", type=float, default=0.3)
a = ap.parse_args()
run(a.root, a.test, a.out, a.model, a.cv, a.C)
