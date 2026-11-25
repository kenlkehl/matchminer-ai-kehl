#!/usr/bin/env python3
"""Publish the MatchMiner-AI Hugging Face Space: upload ``deploy/hf_space/space/`` with ``MMAI_COMMIT``, the commit of
github.com/kenlkehl/matchminer-ai-kehl whose code the Space's image installs. The upload is a commit to the Space
repository, which rebuilds the Space (with Docker's layer cache, so llama.cpp and the pinned dependencies are not
rebuilt unless their layers change).

    python deploy/hf_space/publish.py                    # this checkout's HEAD, which must be on GitHub
    python deploy/hf_space/publish.py --commit <sha>
    python deploy/hf_space/publish.py --create           # once: create the public Docker Space and request its GPU
    python deploy/hf_space/publish.py --factory-rebuild  # also rebuild without the layer cache

The GitHub workflow ``.github/workflows/hf-space.yml`` runs it on every push to main that changes the app. The Hub
token is ``HF_TOKEN``, else the huggingface_hub login. Needs only ``huggingface_hub``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

SPACE = "ksg-dfci/matchminer-ai-1226"
REPOSITORY = "kenlkehl/matchminer-ai-kehl"
HARDWARE = "l40sx1"
SLEEP_SECONDS = 3600  # the Space sleeps after an hour without visitors
FILES = Path(__file__).resolve().parent / "space"
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def head_commit() -> str:
    """This checkout's HEAD; refused when tracked files have uncommitted changes (the Space builds the commit)."""
    root = FILES.parents[2]
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        sys.exit("Uncommitted changes: commit them (and push) first, or pass --commit.")
    return commit


def on_github(commit: str) -> bool:
    """Whether GitHub has ``commit`` (the image downloads the source from there)."""
    request = urllib.request.Request(f"https://api.github.com/repos/{REPOSITORY}/commits/{commit}",
                                     headers={"Accept": "application/vnd.github+json"})
    if os.environ.get("GITHUB_TOKEN"):
        request.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return bool(json.load(response).get("sha") == commit)
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 422):
            return False
        raise


def stage(commit: str, folder: Path) -> list[str]:
    """The Space's files in ``folder``: ``space/`` and ``MMAI_COMMIT``."""
    names = []
    for path in sorted(FILES.iterdir()):
        if path.is_file():
            shutil.copy2(path, folder / path.name)
            names.append(path.name)
    (folder / "MMAI_COMMIT").write_text(commit + "\n")
    return [*names, "MMAI_COMMIT"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commit", help="the commit to build (default: this checkout's HEAD)")
    parser.add_argument("--space", default=SPACE)
    parser.add_argument("--create", action="store_true", help=f"create the public Docker Space if needed and request "
                        f"{HARDWARE} with a {SLEEP_SECONDS // 60}-minute sleep time (billed while awake)")
    parser.add_argument("--factory-rebuild", action="store_true", help="rebuild without Docker's layer cache")
    parser.add_argument("--no-github-check", action="store_true", help="skip checking that GitHub has the commit")
    parser.add_argument("--dry-run", action="store_true", help="only stage the files and print what would be uploaded")
    args = parser.parse_args(argv)
    commit = args.commit or head_commit()
    if not COMMIT.match(commit):
        sys.exit(f"--commit must be a full 40-character commit hash, not {commit!r}")
    if not args.no_github_check and not on_github(commit):
        sys.exit(f"GitHub {REPOSITORY} has no commit {commit}: push it first (the Space downloads its code there).")
    with tempfile.TemporaryDirectory(prefix="mmai-space-") as tmp:
        names = stage(commit, Path(tmp))
        if args.dry_run:
            print(f"Would upload to {args.space}: {', '.join(names)} (MMAI_COMMIT {commit})")
            return 0
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        try:
            api.whoami()
        except Exception:
            sys.exit("No Hugging Face token: set HF_TOKEN (the GitHub secret of the same name) or log in.")
        if not args.create and not api.repo_exists(args.space, repo_type="space"):
            # Before the Space is created (publish.py --create), a push to main has nothing to rebuild.
            print(f"::warning::{args.space} does not exist yet; create it with deploy/hf_space/publish.py --create")
            return 0
        if args.create:
            # The GPU before the first upload, so the first build runs on it.
            api.create_repo(args.space, repo_type="space", space_sdk="docker", private=False, exist_ok=True,
                            space_hardware=HARDWARE, space_sleep_time=SLEEP_SECONDS)
            runtime = api.request_space_hardware(args.space, HARDWARE, sleep_time=SLEEP_SECONDS)
            print(f"{args.space}: {HARDWARE}, sleeps after {SLEEP_SECONDS // 60} min without visitors "
                  f"({runtime.stage})")
        # Files removed from space/ are removed from the Space too (the Hub keeps .gitattributes).
        info = api.upload_folder(repo_id=args.space, repo_type="space", folder_path=tmp, delete_patterns="*",
                                 commit_message=f"Build {REPOSITORY}@{commit[:12]}",
                                 commit_description=f"https://github.com/{REPOSITORY}/commit/{commit}")
        print(f"Uploaded {', '.join(names)} to {args.space}: {getattr(info, 'commit_url', info)}")
    if args.factory_rebuild:
        api.restart_space(args.space, factory_reboot=True)
        print("Factory rebuild requested")
    print(f"https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
