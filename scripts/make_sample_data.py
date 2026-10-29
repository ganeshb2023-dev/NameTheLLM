"""Create a SYNTHETIC stand-in dataset (18 fake 'LLM styles') to test the pipeline.
Not real LLM text - the real train file from Kaggle replaces it.
python scripts/make_sample_data.py --out sample_data --per-class 200"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nameltm.core import LABELS

TOPICS = ("entropy gradient manifold enzyme theorem protein quantum symmetry algorithm recursion "
          "metaphor consciousness ethics photon genome prime lattice protocol memory narrative "
          "friction orbit cell proof induction ontology syntax cache kernel").split()
FUNC = "the of and a to in that is it for as with on this by from are be can which".split()
OPEN = ["Consider", "At its core,", "In short,", "Let us begin:", "Put simply,", "Notably,", "Here is the idea:",
        "Imagine", "Think of", "It turns out that", "Broadly speaking,", "To start,", "Interestingly,",
        "Fundamentally,", "One way to see this:", "First,", "Remarkably,", "Historically,"]
CONN = ["however", "moreover", "thus", "yet", "so", "indeed", "meanwhile", "still", "therefore", "likewise",
        "nevertheless", "in turn", "that said", "crucially", "essentially", "ultimately", "again", "also"]


def style(k):
    r = np.random.default_rng(1000 + k)
    return dict(open=OPEN[k], conn=[CONN[k], CONN[(k * 5 + 3) % 18]], slen=r.integers(9, 26),
                dash=r.choice(["—", " - ", "–", ""], p=[.4, .2, .1, .3]), curly=bool(k % 2),
                bold=r.random() * .4 * (k % 3 == 0), semi=r.random() * .3, colon=r.random() * .3,
                bullets=bool(k % 4 == 1), fav=list(r.choice(TOPICS + FUNC, 6)), favp=.08 + r.random() * .1)


def passage(rng, st):
    out = []
    for si in range(int(rng.integers(3, 12))):
        n = max(5, int(rng.normal(st["slen"], 4)))
        w = [rng.choice(st["fav"]) if rng.random() < st["favp"] else
             (rng.choice(FUNC) if rng.random() < .45 else rng.choice(TOPICS)) for _ in range(n)]
        if si == 0:
            w.insert(0, st["open"])
        if rng.random() < .35:
            w.insert(len(w) // 2, st["conn"][0 if rng.random() < .5 else 1] + ",")
        s = " ".join(w)
        if st["dash"] and rng.random() < .3:
            s = s.replace(" ", st["dash"] if st["dash"].strip() == "" else st["dash"], 1) if st["dash"] != " - " else s.replace(" ", " - ", 1)
        if rng.random() < st["semi"] * .5:
            s = s.replace(" ", "; ", 1)
        if rng.random() < st["colon"] * .5:
            s = s.replace(" ", ": ", 1)
        if rng.random() < st["bold"]:
            ws = s.split(); ws[-1] = f"**{ws[-1]}**"; s = " ".join(ws)
        if st["curly"] and rng.random() < .3:
            s = f"“{s.split()[0]}” " + " ".join(s.split()[1:])
        out.append((s[0].upper() + s[1:]) + ("." if rng.random() < .93 else "?"))
    t = " ".join(out)
    if st["bullets"] and rng.random() < .5:
        t = t.replace(". ", ".\n- ", 2)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="sample_data")
    ap.add_argument("--per-class", type=int, default=200); ap.add_argument("--test-per-class", type=int, default=40)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0); rows, trows = [], []
    for k, lab in enumerate(LABELS):
        st = style(k)
        rows += [(passage(rng, st), lab) for _ in range(a.per_class)]
        trows += [(passage(rng, st), lab) for _ in range(a.test_per_class)]
    tr = pd.DataFrame(rows, columns=["text", "model"]).sample(frac=1, random_state=0).reset_index(drop=True)
    tr.insert(0, "id", [f"tr{i:05d}" for i in range(len(tr))])
    te = pd.DataFrame(trows, columns=["text", "model"]).sample(frac=1, random_state=1).reset_index(drop=True)
    te.insert(0, "id", [f"te{i:05d}" for i in range(len(te))])
    tr.to_csv(f"{a.out}/train.csv", index=False)
    te[["id", "text"]].to_csv(f"{a.out}/test.csv", index=False)
    te[["id", "model"]].to_csv(f"{a.out}/test_answers.csv", index=False)
    print("train", tr.shape, "test", te.shape)
