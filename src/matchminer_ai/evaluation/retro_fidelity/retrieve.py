#!/usr/bin/env python3
"""Stage retrieve: exact cosine top-K of each workflow's queries under both embedders.

Patient-centric: each sampled index event searches every cohort of the corpus. Trial-centric: each sampled cohort
searches every validation index event (all of them, not only the sampled ones). Scores are exact cosine
similarities of the embed stage's normalized vectors (``matchminer_ai.matching.retrieval.score_batches``) and the
top ``--k`` are kept with ``stable_top_k`` (ties by corpus order), as training's mining selects. Both sides of an
embedder must be encoded on the same device kind (CPU float32 and CUDA bf16-autocast vectors are never compared).

``rankings.parquet``: workflow, query_id, embedder, rank (1..K), patient_id, cohort_id, score. Rankers (a) and (b)
are each embedder's top 20 (common.TOP); ranker (c) reranks TrialSpace's top K with the decision model
(lists.py). ``decision_pairs.jsonl``: every pair the decision model scores for the match questions, TrialSpace's
top K and EmbeddingGemma 2's top 20 of every query (so every merged-top-20 pair has decision scores).

A full-denominator corpus (prepare.py corpus --all-queries) has TrialSpace rank every query and EmbeddingGemma 2 only
the comparison queries, the only ones ranker (a) runs for.
"""
import argparse
import json
from pathlib import Path

from matchminer_ai.evaluation.retro_fidelity.common import (EMBEDDERS, RANKERS, TOP, WORKFLOWS, log, pair_rows, private, queries, read_jsonl,  # noqa: E402
                    run_stage, stage, write_json, write_jsonl)

VERSION = 'retro-fidelity-retrieve-v1'


def embedding_dir(root, embedder, side):
    return Path(root) / embedder / side


def search(queries, documents, k):
    """Exact cosine top ``k`` of each query: (document indices, scores), arrays of (queries, min(k, documents)) in
    rank order."""
    import numpy as np
    from matchminer_ai.matching.retrieval import score_batches, stable_top_k
    k = min(k, len(documents))
    index, score = np.zeros((len(queries), k), dtype=np.int64), np.zeros((len(queries), k), dtype=np.float32)
    for start, scores in score_batches(np.asarray(queries), np.asarray(documents), batch_size=128, devices=('cpu',)):
        for offset, values in enumerate(scores):
            top = stable_top_k(values, k)
            index[start + offset], score[start + offset] = top, values[top]
    return index, score


def ranking_frame(workflow, embedder, ids, documents, index, score):
    """The rankings rows of one workflow and embedder (rank order within each query, queries in ``ids`` order)."""
    import numpy as np
    import pandas as pd
    n, k = index.shape
    query = np.repeat(np.asarray(ids, dtype=object), k)
    found = np.asarray([row['id'] for row in documents], dtype=object)[index.ravel()]
    patient, cohort = (query, found) if workflow == 'patient_centric' else (found, query)
    return pd.DataFrame(dict(workflow=[workflow] * (n * k), query_id=list(query), embedder=[embedder] * (n * k),
                             rank=np.tile(np.arange(1, k + 1, dtype=np.int64), n), patient_id=list(patient),
                             cohort_id=list(cohort), score=score.ravel().astype(np.float64)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus', required=True)
    p.add_argument('--embeddings', required=True, help='Root of the embed stages (<embedder>/<side>)')
    p.add_argument('--k', type=int, default=100, help='Candidates kept per query and embedder (ranker (c) reranks TrialSpace\'s)')
    p.add_argument('--output', required=True)
    stage.add_check_arg(p)
    args = p.parse_args(argv)
    private()
    corpus = stage.receipt_id(args.corpus)
    receipts = {f'{e}/{s}': stage.read_receipt(embedding_dir(args.embeddings, e, s))
                for e in EMBEDDERS for s in ('patients', 'cohorts')}
    if any(r is None for r in receipts.values()):
        raise ValueError('Every embedding stage must be complete first')
    identity = dict(version=VERSION, corpus=corpus, k=args.k, top=TOP,
                    embeddings={name: r['receipt_id'] for name, r in receipts.items()})
    output = Path(args.output)

    def work():
        import numpy as np
        import pandas as pd
        patients = read_jsonl(Path(args.corpus) / 'patients.jsonl')
        cohorts = read_jsonl(Path(args.corpus) / 'cohorts.jsonl')
        sample = json.loads((Path(args.corpus) / 'sample.json').read_text())
        pindex = {row['id']: i for i, row in enumerate(patients)}
        cindex = {row['id']: i for i, row in enumerate(cohorts)}
        frames = []
        for embedder in EMBEDDERS:
            kinds = {receipts[f'{embedder}/{s}']['identity']['device_kind'] for s in ('patients', 'cohorts')}
            if len(kinds) != 1:
                raise ValueError(f'{embedder}: patients and cohorts were encoded on different device kinds')
            vectors = {}
            for side, entities in (('patients', patients), ('cohorts', cohorts)):
                directory = embedding_dir(args.embeddings, embedder, side)
                summary = json.loads((directory / 'embedding.json').read_text())
                if summary['ids_sha256'] != stage.digest([row['id'] for row in entities]):
                    raise ValueError(f'{embedder}/{side}: vectors are not in this corpus\'s order')
                vectors[side] = np.load(directory / 'vectors.npy')
            for workflow in WORKFLOWS:
                # (a) runs only for the comparison queries of a full-denominator corpus (every query otherwise)
                ids = queries(sample, workflow, comparison=embedder == RANKERS[0])
                if workflow == 'patient_centric':
                    found, documents = vectors['patients'][[pindex[i] for i in ids]], vectors['cohorts']
                else:
                    found, documents = vectors['cohorts'][[cindex[i] for i in ids]], vectors['patients']
                index, score = search(found, documents, args.k)
                frames.append(ranking_frame(workflow, embedder, ids, cohorts if workflow == 'patient_centric'
                                            else patients, index, score))
                log(f'retrieve {embedder}/{workflow}: {len(ids)} queries')
        frame = pd.concat(frames, ignore_index=True)
        frame.to_parquet(output / 'rankings.parquet', index=False)
        trial_of = {row['id']: row['trial_id'] for row in cohorts}
        chosen = frame[(frame.embedder == 'trialspace_round2') | (frame['rank'] <= TOP)]
        pairs = 0

        def counted(rows):
            nonlocal pairs
            for row in rows:
                pairs += 1
                yield row

        write_jsonl(output / 'decision_pairs.jsonl', counted(pair_rows(chosen, 'embedder', 'source', trial_of)))
        summary = dict(rows=len(frame), decision_pairs=pairs, k=args.k,
                       queries={w: int(frame[frame.workflow == w].query_id.nunique()) for w in WORKFLOWS},
                       shorter_than_k=int((frame.groupby(['workflow', 'embedder', 'query_id']).size() < args.k).sum()))
        write_json(output / 'retrieval.json', summary)
        log(f'retrieve: {summary}')
        return summary

    run_stage(output, 'retro-retrieve', identity, ['rankings.parquet', 'decision_pairs.jsonl', 'retrieval.json'], work,
              check=args.check, upstream=dict(corpus=corpus))


if __name__ == '__main__':
    main()
