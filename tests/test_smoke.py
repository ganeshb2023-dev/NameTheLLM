import os
import tempfile

import numpy as np
import pandas as pd

from nameltm.core import LABELS, run


def test_end_to_end():
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(300)]
    rows = []
    for k, lab in enumerate(LABELS):
        fav = vocab[k * 10:(k + 1) * 10]
        for _ in range(30):
            toks = [rng.choice(fav) if rng.random() < .3 else rng.choice(vocab) for _ in range(80)]
            rows.append((" ".join(toks) + ("—" if k % 2 else "."), lab))
    d = tempfile.mkdtemp()
    df = pd.DataFrame(rows, columns=["text", "model"])
    df.insert(0, "id", range(len(df)))
    df.to_csv(f"{d}/train.csv", index=False)
    df[["id", "text"]].head(20).to_csv(f"{d}/test.csv", index=False)
    run(d, f"{d}/test.csv", f"{d}/sub.parquet", f"{d}/m.joblib", cv=0)
    assert os.path.exists(f"{d}/m.joblib")
