"""Synthetic mode's run from its published dataset on the Hugging Face Hub (e.g. in a Space: ``deploy/hf_space``).

The site config's ``dashboard.synthetic_run`` may name a published synthetic run instead of a local folder::

    dashboard:
      synthetic_run: hf://datasets/ksg-dfci/mmai-synthetic-1226@<commit>

The dataset is laid out for readers, not as a training run, so ``prepare`` builds the run layout the app reads
(``core.settings.Settings.paths``) in a folder of the data root's ``no_phi/synthetic_runs/`` (``local_run``) from the
few dataset files the app needs, downloaded into the Hugging Face cache (``matchminer_ai.site.hub_path``):

- **Corpus:** ``trialspace/trialspace_corpus/{patients,cohorts,source_trial_pairs}-NNNNN.jsonl.zst``, decompressed into
  the corpus folder's JSON-lines files, with ``corpus.json`` and a corpus stage receipt.
- **Trials:** ``trials/ctgov_sample_trials.parquet`` and ``trials/ctgov_trials_with_cohorts.parquet``, as the trials
  folder's two CSV files (``synthetic_trials``).
- **Patients:** ``summaries/synthetic_patient_metadata.parquet`` and ``patients/all_synthetic_notes.parquet``, linked
  from the Hugging Face cache (not copied).
- **Vectors:** ``trialspace/trialspace_round3/mining/`` with a manifest and stage receipt the round checks accept
  (``core.vectors.validate_round``), encoded by the published TrialSpace (the preset's ``embedding.default_model`` at
  ``default_revision``). The published TrialSpace is round 2's fit, so its vectors are round 3's mining vectors. When
  the dataset publishes them (``trialspace/trialspace_round3/mining/{patients,cohorts}.npy`` and the mining
  ``manifest.json``), they are linked after checking that their manifest names these corpus rows and the same encoder
  settings; otherwise the corpus is embedded here with the settings training's mining uses, which takes a long time
  even on a GPU (``settings.device``).

There is no decision model in the folder, so the app scores with the published MatchEval-1226. Each step is skipped
when its outputs are complete, and ``hub_run.json`` is written last: a restart with the same data root and dataset
revision reuses the folder. Other published files (the extraction store, the GoodOption catalog, guideline catalogs)
are named by their own site config keys as ``hf://`` references and read in place.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
import time

from matchminer_ai.site import HubRef, data_root, hub_path, is_hub_ref, load_site_config, parse_hub_ref

FORMAT = "mmai-hub-run-v1"
RECORD = "hub_run.json"
ROUND = 3  # the round whose mining vectors the published TrialSpace (round 2's fit) makes
CORPUS_FILES = ("patients", "cohorts", "source_trial_pairs")
CORPUS_VERSION = "hub-dataset-corpus-v1"
ROUND_VERSION = "hub-dataset-round-v1"
# Dataset paths read (besides the corpus folder) and where each goes in the run: (dataset path, settings path key,
# file name under it or None for the key's own path, how).
TABLES = (
    ("trials/ctgov_sample_trials.parquet", "trials", "ctgov_sample_trials.csv", "csv"),
    ("trials/ctgov_trials_with_cohorts.parquet", "trials", "ctgov_cohorts/trials_with_cohorts.csv", "csv"),
    ("summaries/synthetic_patient_metadata.parquet", "patient_metadata", None, "link"),
    ("patients/all_synthetic_notes.parquet", "notes", None, "link"),
)
CORPUS_PATH = "trialspace/trialspace_corpus"


def published_run() -> HubRef | None:
    """The site config's ``dashboard.synthetic_run`` when it names a published dataset, else None."""
    value = load_site_config().get("dashboard.synthetic_run", resolve=False)
    if not is_hub_ref(value):
        return None
    ref = parse_hub_ref(value)
    if ref.repo_type != "dataset" or ref.path:
        raise ValueError(f"dashboard.synthetic_run {value!r} must name a whole dataset: "
                         "hf://datasets/<org>/<name>@<revision>")
    return ref


def local_run(ref: HubRef) -> Path:
    """The folder ``prepare`` builds for the published run ``ref``, one per dataset revision."""
    return data_root() / "no_phi" / "synthetic_runs" / f"{ref.repo_id.replace('/', '--')}--{ref.revision}"


def round_encoder_settings(model: str, revision: str | None, device: str):
    """TrialSpace as training's mining encodes the corpus with it (``training.trialspace.mining.encoder_settings``:
    float32 weights, bfloat16 autocast, the trialspace prompts, 8,192 tokens, batches of 32)."""
    from matchminer_ai.embedding.encoder import EncoderSettings
    from matchminer_ai.embedding.trialspace_contract import model_prompts

    return EncoderSettings(model, revision=revision, devices=(device,), batch_size=32, max_seq_length=8192,
                           dtype="float32", autocast="bfloat16", prompts=model_prompts("trialspace"),
                           default_prompt_name="SentenceSimilarity", include_prompt_in_pooling=True, text_only=True)


def published_trialspace() -> tuple[str, str | None]:
    """The preset's published TrialSpace and its pinned revision."""
    from matchminer_ai import load_default_preset

    embedding = load_default_preset().embedding
    return str(embedding["default_model"]), (str(embedding.get("default_revision") or "") or None)


def prepare(settings, *, log=print) -> dict | None:
    """Build ``settings.synthetic_run`` from the published run the site config names (None when it names none, or
    when ``--synthetic-run`` chose another folder). Returns the run's record."""
    ref = published_run()
    if ref is None or Path(settings.synthetic_run) != local_run(ref):
        return None
    run = Path(settings.synthetic_run)
    model, revision = published_trialspace()
    record_path = run / RECORD
    wanted = {"format": FORMAT, "dataset": str(ref), "round": ROUND, "trialspace": f"{model}@{revision}"}
    record = _read(record_path)
    if record and all(record.get(k) == v for k, v in wanted.items()) and _complete(settings, ref, model, revision):
        log(f"Synthetic run from {ref}: ready ({run})")
        return dict(record)
    started = time.monotonic()
    log(f"Preparing the synthetic run from {ref} in {run} ...")
    run.mkdir(parents=True, exist_ok=True)
    corpus = _corpus(settings, ref, log)
    for source, key, name, how in TABLES:
        target = settings.path(key) / name if name else settings.path(key)
        _table(hub_path(HubRef(ref.repo_type, ref.repo_id, ref.revision, source)), target, how)
    vectors = _round(settings, ref, corpus, model, revision, log)
    record = {**wanted, "corpus": corpus["receipt_id"], "round_receipt": vectors["receipt_id"],
              "prepared_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    _write_json(record_path, record)
    log(f"Synthetic run ready in {round(time.monotonic() - started)} s")
    return dict(record)


# ---------------------------------------------------------------------------------------------- steps

def _corpus_identity(ref: HubRef, shards: dict[str, list[Path]]) -> dict:
    return {"version": CORPUS_VERSION, "dataset": str(ref),
            "files": {f"{CORPUS_PATH}/{p.name}": p.stat().st_size for paths in shards.values() for p in paths}}


def _shards(ref: HubRef) -> dict[str, list[Path]]:
    folder = hub_path(HubRef(ref.repo_type, ref.repo_id, ref.revision, CORPUS_PATH))
    shards = {name: sorted(folder.glob(f"{name}-*.jsonl.zst")) for name in CORPUS_FILES}
    missing = [name for name, paths in shards.items() if not paths]
    if missing:
        raise FileNotFoundError(f"{ref}: {CORPUS_PATH} has no {', '.join(missing)} shards")
    return shards


def _corpus(settings, ref: HubRef, log) -> dict:
    """The corpus folder: JSON lines, ``corpus.json`` and a ``prepare`` stage receipt."""
    from matchminer_ai.labeling.corpus import load_corpus
    from matchminer_ai.labeling.files import digest
    from matchminer_ai.training.trialspace import stage

    folder = Path(settings.path("corpus"))
    shards = _shards(ref)
    identity = _corpus_identity(ref, shards)
    if stage.check(folder, "prepare", identity)[0] == stage.COMPLETE:
        return dict(stage.read_receipt(folder))
    folder.mkdir(parents=True, exist_ok=True)
    for name, paths in shards.items():
        _decompress(paths, folder / f"{name}.jsonl")
    patients, cohorts = load_corpus(folder)
    info = {"patients": len(patients), "cohorts": len(cohorts), "patient_sha256": digest(patients),
            "cohort_sha256": digest(cohorts), "source": str(ref)}
    _write_json(folder / "corpus.json", info)
    log(f"  corpus: {len(patients):,} patients, {len(cohorts):,} cohorts")
    return dict(stage.write_receipt(folder, "prepare", identity, [f"{n}.jsonl" for n in CORPUS_FILES] + ["corpus.json"],
                                    content={k: info[k] for k in ("patients", "cohorts", "patient_sha256",
                                                                 "cohort_sha256")}))


VECTOR_FILES = ("manifest.json", "patients.npy", "cohorts.npy")
# Encoder identity fields that are not the model itself: the published vectors must have been made with these.
SETTINGS_FIELDS = ("implementation", "backend", "max_seq_length", "prompts", "default_prompt_name", "text_only",
                   "batch_size", "dtype", "autocast", "include_prompt_in_pooling")


def _published_vectors(ref: HubRef) -> dict[str, Path] | None:
    """The run's own round vectors when the dataset publishes them, else None."""
    files = {}
    for name in VECTOR_FILES:
        try:
            files[name] = hub_path(HubRef(ref.repo_type, ref.repo_id, ref.revision,
                                          f"trialspace/trialspace_round{ROUND}/mining/{name}"))
        except FileNotFoundError:
            return None
    return files


def _round_identity(corpus: dict, model: str, revision: str | None, published: dict | None) -> dict:
    from matchminer_ai.embedding.encoder import encoder_identity

    vectors = ({name: published[name].stat().st_size for name in VECTOR_FILES} if published else "embedded")
    return {"version": ROUND_VERSION, "corpus": corpus["receipt_id"],
            "encoder": encoder_identity(round_encoder_settings(model, revision, "cpu")), "vectors": vectors}


def _check_published(published: dict, corpus: dict, encoder: dict) -> dict:
    """The published mining manifest, when its rows are this corpus's and its encoder settings are ours."""
    from matchminer_ai.apps.dashboard.core.vectors import npy_shape

    manifest: dict = json.loads(published["manifest.json"].read_text())
    content = corpus.get("content") or {}
    if (manifest.get("patients"), manifest.get("cohorts")) != (content.get("patient_sha256"),
                                                                content.get("cohort_sha256")):
        raise ValueError("the published vectors were made from other corpus rows")
    theirs = manifest.get("encoder") or {}
    if any(theirs.get(k) != encoder.get(k) for k in SETTINGS_FIELDS):
        raise ValueError("the published vectors were made with other encoder settings")
    for name, rows in (("patients.npy", content.get("patients")), ("cohorts.npy", content.get("cohorts"))):
        shape, dtype = npy_shape(published[name])
        if len(shape) != 2 or shape[0] != rows or shape[1] != manifest.get("dimension") or dtype not in ("<f4",
                                                                                                     "float32"):
            raise ValueError(f"the published {name} has shape {shape} ({dtype}), not {rows} rows")
    return manifest


def _round(settings, ref: HubRef, corpus: dict, model: str, revision: str | None, log) -> dict:
    """``trialspace_round<ROUND>/mining/``: the corpus's vectors, a manifest and a ``mine`` stage receipt."""
    import numpy as np

    from matchminer_ai.embedding.trialspace_contract import COHORT_PROMPT, PATIENT_PROMPT
    from matchminer_ai.labeling.corpus import load_corpus
    from matchminer_ai.training.trialspace import stage

    folder = Path(settings.path("trialspace")) / f"trialspace_round{ROUND}" / "mining"
    published = _published_vectors(ref)
    identity = _round_identity(corpus, model, revision, published)
    if stage.check(folder, "mine", identity)[0] == stage.COMPLETE:
        return dict(stage.read_receipt(folder))
    folder.mkdir(parents=True, exist_ok=True)
    content = corpus.get("content") or {}
    if published:
        source = _check_published(published, corpus, identity["encoder"])
        for name in ("patients.npy", "cohorts.npy"):
            _table(published[name], folder / name, "link")
        log(f"  vectors: the run's published round {ROUND} vectors")
        manifest = {"version": ROUND_VERSION, "dataset": str(ref), "corpus": identity["corpus"],
                    "encoder": identity["encoder"], "dimension": source["dimension"],
                    "patients": content.get("patient_sha256"), "cohorts": content.get("cohort_sha256"),
                    "scope": f"the run's own round {ROUND} mining vectors, made by the fit published as TrialSpace",
                    "published_encoder": source.get("encoder")}
        _write_json(folder / "manifest.json", manifest)
        return dict(stage.write_receipt(folder, "mine", identity, ["patients.npy", "cohorts.npy", "manifest.json"],
                                        upstream={"corpus": identity["corpus"]}))
    patients, cohorts = load_corpus(settings.path("corpus"))
    device = str(settings.device)
    encoder = round_encoder_settings(model, revision, device)
    if not device.startswith("cuda"):
        log(f"  TrialSpace embeds on {device}: expect hours, not minutes")
    dimension = None
    for name, rows, prompt in (("patients", patients, PATIENT_PROMPT), ("cohorts", cohorts, COHORT_PROMPT)):
        started = time.monotonic()
        log(f"  embedding {len(rows):,} {name} with {model}@{str(revision)[:8]} on {device} ...")
        values = _encode(encoder, [row["text"] for row in rows], prompt, log)
        if values.ndim != 2 or len(values) != len(rows) or not np.isfinite(values).all():
            raise ValueError(f"TrialSpace returned invalid {name} vectors {values.shape}")
        dimension = int(values.shape[1])
        temporary = folder / f"{name}.npy.tmp"
        with temporary.open("wb") as stream:
            np.save(stream, values.astype(np.float32), allow_pickle=False)
        temporary.replace(folder / f"{name}.npy")
        log(f"  {name}: {round(time.monotonic() - started)} s")
    manifest = {"version": ROUND_VERSION, "dataset": str(ref), "corpus": identity["corpus"],
                "encoder": identity["encoder"], "dimension": dimension, "patients": content.get("patient_sha256"),
                "cohorts": content.get("cohort_sha256"),
                "scope": "the published corpus, embedded with the published TrialSpace (no candidate mining)"}
    _write_json(folder / "manifest.json", manifest)
    return dict(stage.write_receipt(folder, "mine", identity, ["patients.npy", "cohorts.npy", "manifest.json"],
                                    upstream={"corpus": identity["corpus"]}))


def _encode(settings, texts: list[str], prompt: str, log, chunk: int = 8192):
    """``texts`` embedded in the model's length order (``batch_order``), in chunks of whole batches so the log shows
    progress; rows come back in ``texts`` order."""
    import numpy as np

    from matchminer_ai.embedding.encoder import batch_order, open_encoder

    with open_encoder(settings) as encoder:
        order = batch_order(encoder.model, texts)
        out = None
        step = chunk - chunk % settings.batch_size  # whole batches, so each chunk keeps one call's batches
        started = time.monotonic()
        for start in range(0, len(order), step):
            part = order[start:start + step]
            values = encoder.encode([texts[i] for i in part], prompt)
            if out is None:
                out = np.empty((len(texts), values.shape[1]), dtype=np.float32)
            out[part] = values
            done = min(start + step, len(order))
            if done < len(order):
                log(f"    {done:,}/{len(order):,} ({round(time.monotonic() - started)} s)")
    _free_cuda()
    return out if out is not None else np.empty((0, 0), dtype=np.float32)


def _free_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # no torch or no CUDA: nothing to free
        pass


def _table(source: Path, target: Path, how: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if how == "link":
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source)
        return
    stamp = target.with_name(target.name + ".source")
    if target.is_file() and _read(stamp) == {"source": str(source), "size": source.stat().st_size}:
        return
    import pyarrow.parquet as pq

    temporary = target.with_name(target.name + ".tmp")
    pq.read_table(source).to_pandas().to_csv(temporary, index=False)
    temporary.replace(target)
    _write_json(stamp, {"source": str(source), "size": source.stat().st_size})


def _decompress(shards: list[Path], target: Path) -> None:
    import pyarrow as pa

    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as out:
        for shard in shards:
            with pa.CompressedInputStream(pa.OSFile(str(shard), "rb"), "zstd") as stream:
                while chunk := stream.read(16 << 20):
                    out.write(chunk)
    temporary.replace(target)


def _complete(settings, ref: HubRef, model: str, revision: str | None) -> bool:
    from matchminer_ai.training.trialspace import stage

    try:
        corpus = stage.read_receipt(settings.path("corpus"))
        if corpus is None or stage.check(settings.path("corpus"), "prepare", _corpus_identity(ref, _shards(ref)))[0]:
            return False
        folder = Path(settings.path("trialspace")) / f"trialspace_round{ROUND}" / "mining"
        if stage.check(folder, "mine", _round_identity(corpus, model, revision, _published_vectors(ref)))[0]:
            return False
        targets = [settings.path(key) / name if name else settings.path(key) for _, key, name, _ in TABLES]
        return all(Path(t).exists() for t in targets)
    except (OSError, ValueError):
        return False


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _write_json(path: Path, value) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1) + "\n")
    temporary.replace(path)
