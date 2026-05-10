import argparse
import os

from huggingface_hub import snapshot_download


"""
python3 scripts/download_hf_model.py --repo_id deepseek-ai/Janus-1.3B --local_dir Janus-1.3B
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="deepseek-ai/Janus-1.3B")
    parser.add_argument("--local_dir", type=str, default="./Janus-1.3B")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--allow_patterns", nargs="*", default=None)
    parser.add_argument("--ignore_patterns", nargs="*", default=None)
    args = parser.parse_args()

    repo_id = args.repo_id
    local_dir = args.local_dir

    # Favor faster transport backends when available.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1800")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "1800")

    snapshot_download(
        repo_id=repo_id,
        local_dir=os.path.join(local_dir, repo_id.split("/")[1]),
        max_workers=args.max_workers,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
        resume_download=True,
    )
