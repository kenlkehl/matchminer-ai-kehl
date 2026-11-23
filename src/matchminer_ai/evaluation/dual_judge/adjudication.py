"""Adjudication of a dual-judge run: the units the two judges disagree on (and, with compare --audit-fraction, a sample of agreed ones).

The dashboard's Benchmarking tab serves it (``apps.dashboard.benchmarking``); ``Store`` reads the run directory's
items and ``compare/units.jsonl`` (rerun compare.py to pick up new answers; the store reloads it when it changes) and
appends every decision to ``adjudications.jsonl`` (owner only; the latest decision per unit wins; "skip" withdraws
one). Judges appear as Judge 1 and Judge 2 in an order shuffled per unit (each statement or question, so naming the
judges of one unit tells nothing about the next) and are named only after that unit's decision; for
decision questions the teacher's answer and the decision model's P(yes) are shown only after a decision too, unless
the reviewer asks to see them first. The items are PHI for the summary and decision components.
"""
import os
from pathlib import Path
import random
import threading
import time

from matchminer_ai.evaluation.dual_judge.judging import (COMPONENTS, JUDGE_NAMES, Journal, compared_judges,
                                                          component_module, latest_decisions, read_jsonl)


class Store:
    def __init__(self, run_dir, user=None):
        self.run_dir = Path(run_dir)
        self.user = user if user is not None else os.environ.get('USER', '')
        self.lock = threading.Lock()
        self.modules = {c: component_module(c) for c in COMPONENTS}
        self.items = {c: {i['item_id']: i for i in read_jsonl(self.run_dir / 'items' / f'{c}.jsonl')} for c in COMPONENTS}
        self.journal = Journal(self.run_dir / 'adjudications.jsonl')
        self.units_path = self.run_dir / 'compare' / 'units.jsonl'
        self.units_mtime, self.units, self.by_item = None, {}, {}
        self.judges = ()
        self.reload()

    def reload(self):
        mtime = self.units_path.stat().st_mtime if self.units_path.exists() else None
        if mtime == self.units_mtime:
            return
        units = read_jsonl(self.units_path)
        self.judges = compared_judges(self.run_dir)
        self.units = {u['unit_id']: u for u in units}
        self.by_item = {}
        for u in units:
            self.by_item.setdefault((u['component'], u['item_id']), []).append(u['unit_id'])
        self.units_mtime = mtime

    def decisions(self):
        return latest_decisions(read_jsonl(self.run_dir / 'adjudications.jsonl'))

    def order(self, unit_id):
        """The unit's judges as Judge 1 and Judge 2: a shuffle seeded by the unit, the same on every reload."""
        judges = list(self.judges)
        random.Random(f'{unit_id}:judge-order').shuffle(judges)
        return judges

    def state(self):
        with self.lock:
            self.reload()
            decisions = self.decisions()
            out = {}
            for component in COMPONENTS:
                module, rows = self.modules[component], []
                for item_id, item in self.items[component].items():
                    ids = self.by_item.get((component, item_id), [])
                    queued = [i for i in ids if self.units[i]['queued']]
                    view = module.view(item)
                    rows.append(dict(item_id=item_id, title=view['title'], stratum=item.get('stratum'),
                                     units=len(ids), queued=len(queued),
                                     decided=sum(i in decisions for i in queued)))
                out[component] = dict(items=rows, queued=sum(r['queued'] for r in rows),
                                      decided=sum(r['decided'] for r in rows))
            return dict(components=out, compared_at=self.units_mtime)

    def unit_view(self, unit, item, decisions, show_reference=False):
        order = self.order(unit['unit_id'])
        module = self.modules[unit['component']]
        kind = module.KINDS[unit['kind']]
        collapse = kind.get('collapse') or {}
        choices = list(dict.fromkeys(collapse.get(v, v) for v in kind['values']))
        judges = [dict(slot=n, value=unit['values'][j], compared=unit['compared'][j], detail=unit['details'][j])
                  for n, j in enumerate(order, 1)]
        row = dict(unit_id=unit['unit_id'], key=unit['key'], kind=unit['kind'], label=kind['label'], choices=choices,
                   target=module.unit_target(item, unit['key']), status=unit['status'], queued=unit['queued'],
                   queue_reason=unit['queue_reason'], judges=judges, decision=None,
                   question=module.question(item, unit['key']) if hasattr(module, 'question') else None)
        decision = decisions.get(unit['unit_id'])
        if hasattr(module, 'reveal') and (decision is not None or show_reference):
            row['reference'] = module.reveal(item, unit['key'])
        if decision is not None:
            row['decision'] = dict(value=decision['value'], note=decision.get('note', ''), at=decision['at'],
                                   stale=decision.get('seen') != unit['compared'])
            row['reveal'] = dict(judges={n: JUDGE_NAMES.get(j, j) for n, j in enumerate(order, 1)})
        return row

    def item(self, component, item_id, show_reference=False):
        with self.lock:
            self.reload()
            item = self.items[component][item_id]
            decisions = self.decisions()
            units = [self.unit_view(self.units[i], item, decisions, show_reference)
                     for i in self.by_item.get((component, item_id), [])]
            return dict(component=component, item_id=item_id, view=self.modules[component].view(item), units=units)

    def decide(self, body):
        with self.lock:
            self.reload()
            unit = self.units.get(body.get('unit_id'))
            if unit is None:
                raise KeyError('unknown unit')
            choice, value = body.get('choice'), body.get('value')
            if choice != 'skip':
                module = self.modules[unit['component']]
                kind = module.KINDS[unit['kind']]
                allowed = {(kind.get('collapse') or {}).get(v, v) for v in kind['values']}
                if value not in allowed:
                    raise ValueError('value is not one of the unit\'s choices')
            order = self.order(unit['unit_id'])
            if choice in ('judge1', 'judge2'):
                picked = order[int(choice[-1]) - 1]
            else:  # a value chosen from the list: the judge who gave it, when exactly one did
                givers = [j for j in order if unit['compared'].get(j) == value]
                picked = givers[0] if choice != 'skip' and len(givers) == 1 else None
            self.journal.append(dict(unit_id=unit['unit_id'], choice=choice, value=value, picked_judge=picked,
                                     note=str(body.get('note', ''))[:4000], seen=unit['compared'],
                                     at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), user=self.user))
        return self.item(unit['component'], unit['item_id'], bool(body.get('show_reference')))
