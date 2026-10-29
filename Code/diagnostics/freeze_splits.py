"""
freeze_splits.py

Builds the train/val/test splits ONCE and saves them to splits_frozen/*.csv,
with a SHA-256 of each file, so every later script (training, attacks, TREC07
evaluation) uses exactly the same rows.

Why: the same code built slightly different splits in different containers
(phishing test rows 745 vs 743, validation 1476 vs 1475), so results from
different runs were not on identical data.

Usage:
    python freeze_splits.py            # build + save (refuses to overwrite)
    python freeze_splits.py --force    # overwrite

Then paste the loader snippet (see freeze_splits_patch.txt) at the top of
build_dataset_combined_corpus() in model_and_data_utils.py.
"""
import hashlib
import os
import sys

OUT = "splits_frozen"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    force = "--force" in sys.argv
    names = ["train", "val", "test"]
    if all(os.path.exists(f"{OUT}/{n}.csv") for n in names) and not force:
        print("Frozen splits already exist (use --force to overwrite):")
        for n in names:
            print(f"  {n}.csv sha256={sha(f'{OUT}/{n}.csv')}")
        return
    # Build with the ORIGINAL logic: make sure the frozen loader is not active yet.
    from model_and_data_utils import build_dataset_combined_corpus
    train, val, test = build_dataset_combined_corpus(verbose=False)
    os.makedirs(OUT, exist_ok=True)
    for n, d in zip(names, (train, val, test)):
        d.to_csv(f"{OUT}/{n}.csv", index=False)
        print(f"{n}: {len(d)} rows (label=1: {int((d['label'] == 1).sum())}) "
              f"sha256={sha(f'{OUT}/{n}.csv')}")
    print(f"\nSaved to {OUT}/ -- copy this folder off the container.")


if __name__ == "__main__":
    main()