#!/usr/bin/env python3
"""Restart the MatchMiner-AI Space so it loads the dataset's current files, after a dataset update (for example new
guideline catalogs under ``guideline_catalogs/pdq/``). No rebuild: the image is unchanged, and the Space reads the
dataset's main branch at startup unless the Space variable ``MMAI_DATASET_REVISION`` pins a commit (``site.yaml``).

    python deploy/hf_space/refresh.py                 # show what the Space will load, then restart it
    python deploy/hf_space/refresh.py --wait          # and wait until it runs again
    python deploy/hf_space/refresh.py --pin <commit>  # pin the dataset revision (a Space variable), then restart
    python deploy/hf_space/refresh.py --unpin         # follow the dataset's main again, then restart
    python deploy/hf_space/refresh.py --dry-run       # only show what the Space would load

The Hub token is ``HF_TOKEN``, else the huggingface_hub login (write access to the Space). Needs only
``huggingface_hub``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

SPACE = "ksg-dfci/matchminer-ai-1226"
DATASET = "ksg-dfci/mmai-synthetic-1226"
VARIABLE = "MMAI_DATASET_REVISION"
CATALOGS = "guideline_catalogs/pdq"
COMMIT = re.compile(r"^[0-9a-f]{40}$")
TERMINAL = {"RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "PAUSED", "STOPPED"}


def catalogs_at(api, revision: str) -> int:
    """The guideline catalogs (folders with paradigms.jsonl) the dataset has under CATALOGS at ``revision``."""
    try:
        files = api.list_repo_files(DATASET, repo_type="dataset", revision=revision)
    except Exception as exc:  # noqa: BLE001 -- shown, not fatal
        print(f"Could not list {DATASET}@{revision}: {type(exc).__name__}: {exc}")
        return -1
    return sum(1 for f in files if f.startswith(CATALOGS + "/") and f.endswith("/paradigms.jsonl"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", default=SPACE)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--pin", metavar="COMMIT", help=f"set the Space variable {VARIABLE} to this dataset commit")
    choice.add_argument("--unpin", action="store_true", help=f"delete {VARIABLE}: follow the dataset's main")
    parser.add_argument("--wait", action="store_true", help="wait until the Space runs again (or fails)")
    parser.add_argument("--timeout", type=float, default=3 * 3600, help="seconds --wait waits (default 3 h)")
    parser.add_argument("--dry-run", action="store_true", help="only show what the Space would load")
    args = parser.parse_args(argv)
    if args.pin and not COMMIT.match(args.pin):
        sys.exit("--pin takes a full 40-character dataset commit")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    main_commit = api.dataset_info(DATASET).sha
    try:
        found = api.get_space_variables(args.space) or {}
    except Exception as exc:  # noqa: BLE001 -- e.g. the Space does not exist yet
        sys.exit(f"{args.space}: {type(exc).__name__}: {str(exc)[:200]}")
    variables = {key: getattr(value, "value", value) for key, value in found.items()}
    pinned = None if args.unpin else (args.pin or variables.get(VARIABLE) or None)
    revision = pinned or main_commit
    print(f"{DATASET}: main is {main_commit}")
    print(f"{args.space} will load {'the pinned ' + pinned if pinned else 'main'}: "
          f"{catalogs_at(api, revision)} guideline catalogs under {CATALOGS}/")
    if pinned and pinned != main_commit and not args.pin:
        print(f"Note: {VARIABLE} pins {pinned}, so the restart does not load main (use --unpin or --pin).")
    if args.dry_run:
        return 0
    if args.pin:
        api.add_space_variable(args.space, VARIABLE, args.pin, description="mmai-synthetic-1226 revision the Space "
                               "loads (unset: main)")
    elif args.unpin and VARIABLE in variables:
        api.delete_space_variable(args.space, VARIABLE)
    api.restart_space(args.space)
    print(f"Restarting {args.space}: https://huggingface.co/spaces/{args.space}")
    if not args.wait:
        return 0
    deadline, stage = time.monotonic() + args.timeout, None
    time.sleep(20)  # the restart takes a moment to show
    while time.monotonic() < deadline:
        runtime = api.get_space_runtime(args.space)
        if runtime.stage != stage:
            stage = runtime.stage
            print(time.strftime("%H:%M:%S"), stage, flush=True)
        if stage in TERMINAL:
            return 0 if stage == "RUNNING" else 1
        time.sleep(30)
    print(f"Still {stage} after {args.timeout:.0f} s")
    return 1


if __name__ == "__main__":
    sys.exit(main())
