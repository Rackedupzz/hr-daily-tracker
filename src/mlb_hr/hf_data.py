"""Sync the data/ folder with a private Hugging Face dataset repo.

The season PA file and the fitted models are gitignored (too big, and the
fits change daily), so a fresh clone has no data. This keeps them on the Hub:

    python -m mlb_hr.hf_data push   # upload local data -> Hub
    python -m mlb_hr.hf_data pull   # download files missing locally

Both need HF_TOKEN and HF_DATA_REPO (e.g. "user/hr-daily-tracker-data").
pull runs at container start and is a no-op when either is unset. It only
fetches files that don't exist locally, so it never clobbers fits the running
app has refreshed since the last push.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Top-level files only. The caches (backtest_cache/, slate_cache/) are rebuilt
# on demand, and season_pa.jsonl is the retired v1 file nothing reads.
PATTERNS = ["season_pa_v2.jsonl", "*.json", "*.pkl"]


def _config():
    token = os.environ.get("HF_TOKEN")
    repo = os.environ.get("HF_DATA_REPO")
    return (token, repo) if token and repo else (None, None)


def push() -> None:
    from huggingface_hub import HfApi

    token, repo = _config()
    if not repo:
        sys.exit("[hf_data] set HF_TOKEN and HF_DATA_REPO to push")
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    # Upload a frozen copy: the running app appends to the PA file, and a file
    # that changes between hashing and upload is rejected by the Hub.
    with tempfile.TemporaryDirectory() as staging:
        for pattern in PATTERNS:
            for path in DATA_DIR.glob(pattern):
                shutil.copy2(path, staging)
        api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=staging,
            commit_message="Sync data/",
        )
    print(f"[hf_data] pushed {DATA_DIR} -> datasets/{repo}")


def pull() -> None:
    token, repo = _config()
    if not repo:
        print("[hf_data] HF_TOKEN/HF_DATA_REPO unset, skipping pull")
        return
    try:
        from huggingface_hub import HfApi, hf_hub_download

        remote = HfApi(token=token).list_repo_files(repo, repo_type="dataset")
        missing = [
            f for f in remote
            if "/" not in f and not f.startswith(".") and not (DATA_DIR / f).exists()
        ]
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for name in missing:
            hf_hub_download(repo, name, repo_type="dataset", token=token, local_dir=str(DATA_DIR))
        print(f"[hf_data] pulled {len(missing)} missing file(s) from datasets/{repo}")
    except Exception as exc:  # the app can still start and refit without the Hub
        print(f"[hf_data] pull failed, continuing: {exc}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "push":
        push()
    elif cmd == "pull":
        pull()
    else:
        sys.exit("usage: python -m mlb_hr.hf_data push|pull")
