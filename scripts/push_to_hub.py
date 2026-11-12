"""Publish the main model (splined logistic regression) to the Hugging Face Hub.

    python scripts/push_to_hub.py --repo <user>/diabetes-readmission-logistic

Uploads the skops-serialized pipeline, the feature module it needs, the model card (hub/README.md),
the metrics and the figures. Requires `hf auth login` beforehand.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        shutil.copy(ROOT / "hub" / "README.md", t / "README.md")
        shutil.copy(ROOT / "hub" / "example.py", t / "example.py")
        shutil.copy(ROOT / "models" / "lr_spline.skops", t / "lr_spline.skops")
        shutil.copy(ROOT / "results" / "results.json", t / "results.json")
        shutil.copytree(ROOT / "src" / "readmission", t / "readmission",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "figures", t / "figures")
        api.upload_folder(folder_path=str(t), repo_id=args.repo, repo_type="model",
                          commit_message="Upload splined logistic regression for 30-day readmission")
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
