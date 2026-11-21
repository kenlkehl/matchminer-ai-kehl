"""Token use, throughput and rate limits of a gateway request log (``$MMAI_GATEWAY_REQUEST_LOG``).

The log holds one JSON line per Claude request (status, seconds, token counts; never content) and one per change of
the adaptive concurrency limit (``gateway_limits``). This reports requests by status, input tokens split into
uncached, cache reads and cache writes, output tokens, requests and tokens per minute, latency, the concurrency
limit over time and when rate limits began, and, with ``--rates``, the cost::

    python -m matchminer_ai.llm.gateway_usage requests.jsonl --rates 0.10,0.50 --long-rates 0.50,2.50 --discount 0.75

Rates are USD per million input and output tokens; cache reads cost 0.1x and 5-minute cache writes 1.25x the input
rate unless given as the third and fourth values. ``--long-rates`` price a whole request whose prompt (uncached,
cache-read and cache-written input) is over ``--long-threshold`` tokens (100,000 by default), as Claude Haiku 5.5's
long-prompt rate card does. ``--discount`` is a fraction taken off every rate (0.75 for 75% off).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

TOKENS = ("uncached_input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")
LONG_THRESHOLD = 100_000


def _card(rates):
    """(input, output, cache read, cache write) per million tokens from 2-4 values."""
    if not rates:
        return None
    if len(rates) not in (2, 3, 4):
        raise ValueError("Give 2 to 4 rates: input,output[,cache_read,cache_write].")
    input_rate, output_rate = rates[0], rates[1]
    return (input_rate, output_rate, rates[2] if len(rates) > 2 else input_rate * 0.1,
            rates[3] if len(rates) > 3 else input_rate * 1.25)


def _cost(tokens: dict, card) -> float:
    return (tokens["uncached_input_tokens"] * card[0] + tokens["output_tokens"] * card[1]
            + tokens["cache_read_input_tokens"] * card[2] + tokens["cache_creation_input_tokens"] * card[3]) / 1e6


def _time(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q * len(values)))], 2)


def summarize(path: str | Path, rates: tuple[float, ...] | None = None, *,
              long_rates: tuple[float, ...] | None = None, long_threshold: int = LONG_THRESHOLD,
              discount: float = 0.0) -> dict:
    """The report of one log (see the module docstring), as a dict."""
    if not 0 <= discount < 1:
        raise ValueError("discount is a fraction from 0 to below 1 (0.75 for 75% off).")
    statuses: Counter = Counter()
    tokens = dict.fromkeys(TOKENS, 0)
    # Tokens by price band: prompts up to long_threshold, and longer ones.
    bands = {"standard": dict.fromkeys(TOKENS, 0), "long": dict.fromkeys(TOKENS, 0)}
    band_requests = Counter()
    seconds: list[float] = []
    minutes: dict[str, Counter] = defaultdict(Counter)
    limits: list[dict] = []
    first: datetime | None = None
    last: datetime | None = None
    first_limited = None
    over_band = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("event") == "concurrency":
            limits.append({k: entry.get(k) for k in ("at", "limit", "reason", "active", "waiting")})
            continue
        started = _time(entry.get("started_at"))
        if started is None:
            continue
        first = started if first is None or started < first else first
        last = started if last is None or started > last else last
        status = entry.get("status")
        statuses[str(status)] += 1
        minute = started.strftime("%Y-%m-%dT%H:%M")
        minutes[minute]["requests"] += 1
        if status in (429, 503, 529):
            minutes[minute]["rate_limited"] += 1
            if first_limited is None:
                current = limits[-1]["limit"] if limits else None
                first_limited = {"at": entry["started_at"], "status": status, "concurrency_limit": current}
        if status != 200:
            continue
        seconds.append(float(entry.get("seconds") or 0))
        uncached = entry.get("uncached_input_tokens")
        if uncached is None:  # logs written before the split: input_tokens includes cache reads
            uncached = int(entry.get("input_tokens") or 0) - int(entry.get("cache_read_input_tokens") or 0)
        values = {"uncached_input_tokens": int(uncached),
                  "cache_read_input_tokens": int(entry.get("cache_read_input_tokens") or 0),
                  "cache_creation_input_tokens": int(entry.get("cache_creation_input_tokens") or 0),
                  "output_tokens": int(entry.get("output_tokens") or 0)}
        for key, value in values.items():
            tokens[key] += value
        prompt = values["uncached_input_tokens"] + values["cache_read_input_tokens"] \
            + values["cache_creation_input_tokens"]
        band = "long" if prompt > long_threshold else "standard"
        band_requests[band] += 1
        for key, value in values.items():
            bands[band][key] += value
        over_band += prompt > long_threshold
        minutes[minute]["input_tokens"] += prompt
        minutes[minute]["output_tokens"] += values["output_tokens"]
    succeeded = statuses.get("200", 0)
    input_total = tokens["uncached_input_tokens"] + tokens["cache_read_input_tokens"] \
        + tokens["cache_creation_input_tokens"]
    report = {
        "log": Path(path).name,
        "first_request": first.isoformat() if first else None,
        "last_request": last.isoformat() if last else None,
        "requests": dict(statuses),
        "succeeded": succeeded,
        "tokens": {**tokens, "input_total": input_total},
        "cache_read_share_of_input": round(tokens["cache_read_input_tokens"] / input_total, 4) if input_total else None,
        "mean_per_request": {k: round(v / succeeded) for k, v in tokens.items()} if succeeded else None,
        "requests_over_long_threshold": over_band,
        "long_threshold": long_threshold,
        "latency_seconds": {"p50": _percentile(seconds, 0.5), "p95": _percentile(seconds, 0.95),
                            "max": max(seconds) if seconds else None},
        "per_minute_peak": {key: max((m[key] for m in minutes.values()), default=0)
                            for key in ("requests", "input_tokens", "output_tokens", "rate_limited")},
        "per_minute_median": {key: _percentile([float(m[key]) for m in minutes.values()], 0.5)
                              for key in ("requests", "input_tokens", "output_tokens")},
        "first_rate_limited": first_limited,
        "concurrency": {"changes": len(limits), "max": max((c["limit"] or 0 for c in limits), default=None),
                        "last": limits[-1] if limits else None, "trajectory": limits[-40:]},
    }
    standard = _card(rates)
    if standard:
        long_card = _card(long_rates) or standard
        by_band = {"standard": _cost(bands["standard"], standard), "long": _cost(bands["long"], long_card)}
        listed = sum(by_band.values())
        report["cost"] = {
            "list_usd": round(listed, 2),
            "discount": discount,
            "usd": round(listed * (1 - discount), 2),
            "by_band_list_usd": {k: round(v, 2) for k, v in by_band.items()},
            "requests_by_band": dict(band_requests),
            "rates_per_mtok": {"standard": dict(zip(("input", "output", "cache_read", "cache_write"), standard)),
                               "long": dict(zip(("input", "output", "cache_read", "cache_write"), long_card))},
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="a gateway request log (JSON lines)")
    parser.add_argument("--rates", help="USD per million tokens: input,output[,cache_read,cache_write]")
    parser.add_argument("--long-rates", help="the same, for requests whose prompt is over --long-threshold")
    parser.add_argument("--long-threshold", type=int, default=LONG_THRESHOLD,
                        help="prompt tokens above which --long-rates apply (default: %(default)s)")
    parser.add_argument("--discount", type=float, default=0.0,
                        help="fraction taken off every rate, e.g. 0.75 for 75%% off (default: 0)")
    parser.add_argument("--full", action="store_true", help="include the concurrency trajectory")
    args = parser.parse_args(argv)
    def values(text):
        return tuple(float(v) for v in text.split(",")) if text else None

    report = summarize(args.log, values(args.rates), long_rates=values(args.long_rates),
                       long_threshold=args.long_threshold, discount=args.discount)
    if not args.full:
        report["concurrency"].pop("trajectory")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
