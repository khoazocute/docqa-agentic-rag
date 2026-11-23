"""Download the public benchmarks used by the evaluation into data/benchmarks/ (git-ignored).

    python scripts/download_benchmarks.py            # all
    python scripts/download_benchmarks.py zalo       # one of: zalo, viquad, injections

Sources (check each dataset card for terms of use before redistributing anything):
- Zalo AI 2021 Legal Text Retrieval, BEIR format, MIT: GreenNode/zalo-ai-legal-text-retrieval-vn
- UIT-ViQuAD 2.0 (VLSP 2021) community mirror, licence not stated: taidng/UIT-ViQuAD2.0
  (the official release is available from the UIT NLP group on request)
- deepset/prompt-injections, Apache-2.0
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
ROOT = Path(__file__).resolve().parents[1] / "data" / "benchmarks"

DATASETS: dict[str, tuple[str, str, list[str]]] = {
    "zalo": (
        "GreenNode/zalo-ai-legal-text-retrieval-vn",
        "zalo_legal",
        ["corpus.jsonl", "queries.jsonl", "qrels/test.jsonl", "qrels/train.jsonl", "README.md"],
    ),
    "viquad": (
        "taidng/UIT-ViQuAD2.0",
        "viquad2",
        ["data/validation-00000-of-00001.parquet", "README.md"],
    ),
    "injections": (
        "deepset/prompt-injections",
        "deepset_injections",
        [
            "data/test-00000-of-00001-701d16158af87368.parquet",
            "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
            "README.md",
        ],
    ),
}


def download(name: str) -> None:
    repo, folder, files = DATASETS[name]
    for path in files:
        target = ROOT / folder / path
        if target.exists():
            print(f"exists  {target.relative_to(ROOT.parent.parent)}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetch   {repo}/{path}")
        urllib.request.urlretrieve(HF.format(repo=repo, path=path), target)  # noqa: S310


if __name__ == "__main__":
    names = sys.argv[1:] or list(DATASETS)
    unknown = set(names) - set(DATASETS)
    if unknown:
        raise SystemExit(
            f"unknown dataset(s): {', '.join(sorted(unknown))}; choose from {list(DATASETS)}"
        )
    for dataset in names:
        download(dataset)
