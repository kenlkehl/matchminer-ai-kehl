#!/usr/bin/env python3
"""Publish a MatchMiner-AI Hugging Face Space: upload ``deploy/hf_space/space/`` with ``MMAI_COMMIT``, the commit of
github.com/kenlkehl/matchminer-ai-kehl whose code the Space's image installs. The upload is a commit to the Space
repository, which rebuilds the Space (with Docker's layer cache, so llama.cpp and the pinned dependencies are not
rebuilt unless their layers change).

Every Space (``SPACES``) gets the same files and commit; a Space may have its own card (``README.md``) and image
variant (``SPACE_VARIANTS``: the Dockerfile's ``MMAI_VARIANT``, full by default; lite has no local model server), and
has its own hardware, by which the image picks its tier (``space/tier.py``).

    python deploy/hf_space/publish.py                    # this checkout's HEAD, which must be on GitHub
    python deploy/hf_space/publish.py --commit <sha>
    python deploy/hf_space/publish.py --space ksg-dfci/matchminer-ai-1226-lite
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
from typing import Any

REPOSITORY = "kenlkehl/matchminer-ai-kehl"
CARDS = Path(__file__).resolve().parent / "cards"
# Each Space's hardware flavor (with --create). The workflow publishes to each.
SPACES = {
    # One RTX PRO 6000 (96 GB, Blackwell): vLLM serves Nemotron 3.5 Super there (site.yaml). Not in huggingface_hub's
    # SpaceHardware enum yet, so passed as a string.
    "ksg-dfci/matchminer-ai-1226": "rtx-pro-6000",
    # One T4 (16 GB): no local LLM; the LLM tasks use OpenRouter with each visitor's own key (tier.py).
    "ksg-dfci/matchminer-ai-1226-lite": "t4-small",
}
# A Space's own card in place of space/README.md.
SPACE_CARDS = {"ksg-dfci/matchminer-ai-1226-lite": CARDS / "matchminer-ai-1226-lite.md"}
# A Space's image variant (the Dockerfile's MMAI_VARIANT), when not the full one: lite builds no llama.cpp, vLLM or
# FlashInfer, for a Space that hosts no LLM.
SPACE_VARIANTS = {"ksg-dfci/matchminer-ai-1226-lite": "lite"}
VARIANT_LINE = "ARG MMAI_VARIANT=full\n"
SPACE = "ksg-dfci/matchminer-ai-1226"
HARDWARE = SPACES[SPACE]
SLEEP_SECONDS = 18000  # a Space sleeps after five hours without visitors (a wake is a long cold start)
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


def stage(commit: str, folder: Path, card: Path | None = None, variant: str = "full") -> list[str]:
    """The Space's files in ``folder``: ``space/``, ``MMAI_COMMIT`` and, when the Space has its own, its card as
    ``README.md``; the Dockerfile builds ``variant``."""
    names = []
    for path in sorted(FILES.iterdir()):
        if path.is_file():
            shutil.copy2(path, folder / path.name)
            names.append(path.name)
    if card is not None:
        shutil.copy2(card, folder / "README.md")
    if variant != "full":
        dockerfile = folder / "Dockerfile"
        text = dockerfile.read_text()
        if text.count(VARIANT_LINE) != 1:
            raise ValueError(f"The Dockerfile has no single {VARIANT_LINE.strip()!r} line to set the variant.")
        dockerfile.write_text(text.replace(VARIANT_LINE, f"ARG MMAI_VARIANT={variant}\n"))
    (folder / "MMAI_COMMIT").write_text(commit + "\n")
    return [*names, "MMAI_COMMIT"]


def refused(exc: Exception) -> bool:
    """Whether a Hub error is the token's lack of access (401 or 403)."""
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) in (401, 403)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commit", help="the commit to build (default: this checkout's HEAD)")
    parser.add_argument("--space", default=SPACE, help=f"the Space (default {SPACE}); known: {', '.join(SPACES)}")
    parser.add_argument("--create", action="store_true", help="create the public Docker Space if needed and request "
                        "--hardware with --sleep-seconds (billed while awake)")
    parser.add_argument("--hardware", help="the Space's hardware flavor with --create (default: the Space's in "
                        "SPACES)")
    parser.add_argument("--sleep-seconds", type=int, default=SLEEP_SECONDS, help="with --create: seconds without "
                        f"visitors before the Space sleeps (default {SLEEP_SECONDS}, {SLEEP_SECONDS // 3600} hours)")
    parser.add_argument("--factory-rebuild", action="store_true", help="rebuild without Docker's layer cache")
    parser.add_argument("--no-github-check", action="store_true", help="skip checking that GitHub has the commit")
    parser.add_argument("--dry-run", action="store_true", help="only stage the files and print what would be uploaded")
    args = parser.parse_args(argv)
    # A flavor name; huggingface_hub's SpaceHardware enum lacks some (rtx-pro-6000), so it stays a string.
    hardware: Any = args.hardware or SPACES.get(args.space) or HARDWARE
    commit = args.commit or head_commit()
    if not COMMIT.match(commit):
        sys.exit(f"--commit must be a full 40-character commit hash, not {commit!r}")
    if not args.no_github_check and not on_github(commit):
        sys.exit(f"GitHub {REPOSITORY} has no commit {commit}: push it first (the Space downloads its code there).")
    with tempfile.TemporaryDirectory(prefix="mmai-space-") as tmp:
        variant = SPACE_VARIANTS.get(args.space, "full")
        names = stage(commit, Path(tmp), SPACE_CARDS.get(args.space), variant)
        if args.dry_run:
            print(f"Would upload to {args.space}: {', '.join(names)} (MMAI_COMMIT {commit}, {variant} image)")
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
        try:
            if args.create:
                # The GPU before the first upload, so the first build runs on it.
                api.create_repo(args.space, repo_type="space", space_sdk="docker", private=False, exist_ok=True,
                                space_hardware=hardware, space_sleep_time=args.sleep_seconds)
                runtime = api.request_space_hardware(args.space, hardware, sleep_time=args.sleep_seconds)
                print(f"{args.space}: {hardware}, sleeps after {args.sleep_seconds // 60} min without visitors "
                      f"({runtime.stage})")
            # Files removed from space/ are removed from the Space too (the Hub keeps .gitattributes).
            info = api.upload_folder(repo_id=args.space, repo_type="space", folder_path=tmp, delete_patterns="*",
                                     commit_message=f"Build {REPOSITORY}@{commit[:12]}",
                                     commit_description=f"https://github.com/{REPOSITORY}/commit/{commit}")
        except Exception as exc:
            if refused(exc):
                sys.exit(f"::error::The Hugging Face token may not write to {args.space}: give the token (the GitHub "
                         "secret HF_TOKEN) write access to this Space.")
            raise
        print(f"Uploaded {', '.join(names)} to {args.space}: {getattr(info, 'commit_url', info)}")
    if args.factory_rebuild:
        api.restart_space(args.space, factory_reboot=True)
        print("Factory rebuild requested")
    print(f"https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
