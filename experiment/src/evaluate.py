"""Shared answer log-probability scorer; writes one tidy immutable evaluation table."""

import argparse, csv, json
from common import *


def main(a):
    m, t, _ = loader()
    m.eval()
    rs = read_jsonl(a.rows)
    out = []
    for r in rs:
        s = score(m, t, r["question"], r["answer"])
        out.append(
            {
                "state": a.state,
                "split": a.split,
                "author_id": r.get("author_id", "utility"),
                "row_id": r.get("row_id", ""),
                "metric": "answer_logprob",
                "value": s,
            }
        )
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0])
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--state")
    p.add_argument("--split")
    p.add_argument("--rows")
    p.add_argument("--out")
    main(p.parse_args())
