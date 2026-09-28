"""Learning from customers' own runs: the episodes xybernetex-openclaw 0.4+
reports to POST /outcome, which the policy Worker stores in the
xybernetex_outcomes Analytics Engine dataset (cfworker src/outcome.js). Each
episode becomes a learn.bandit Record, so the same estimators and learner
that run on the graded hard2 arms run on the fleet.

What differs from the graded arms, and why:

  propensity  Each episode carries the probability of what was actually
              applied, logged at decision time (observe mode and failed starts
              are "none" with probability 1). It's used directly; there's no
              arm mixture to reconstruct.
  reward      A customer's run has no grader, so success is a proxy: the
              run's work stands (it finished, or a retry finished it) and the
              user's next message wasn't a correction or the same request
              again. It's noisier than graded success and is reported on its
              own, never pooled with the graded arms.
  weight      No customer may be more than --cap of the records. Larger ones
              are subsampled (seeded), so one customer - including one sending
              made-up episodes - can't steer the shared policy.
  folds       Cross-fitting is by customer once there are enough of them, so
              an estimate for one customer's runs comes from a model fit on
              the others'. With few customers it falls back to buckets within
              each customer, and exporting a shared table is refused.

    python -m learn.fleet --days 30                       # estimates only
    python -m learn.fleet --days 30 --export out/fleet_policy.json
    python -m learn.fleet --from-json rows.json           # an exported query result

Reading the dataset needs CF_ACCOUNT_ID and CF_ANALYTICS_TOKEN (Account
Analytics: Read) in the environment. Nothing here uploads anything.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from learn import bandit

DATASET = "xybernetex_outcomes"
BAD_REPLIES = {"correction", "repeat"}
MIN_CUSTOMERS = 3  # below this a shared table would really be one customer's

QUERY = """SELECT index1 AS customer, blob2 AS model, blob3 AS decided, blob4 AS applied, blob5 AS rule,
  blob8 AS user, blob9 AS mode, double1 AS probability, double2 AS run_success, double3 AS retriable,
  double4 AS tool_calls, double5 AS tokens, double6 AS followup_ran, double7 AS followup_success,
  double11 AS followup_tokens, _sample_interval AS sample_interval
FROM {dataset} WHERE timestamp > NOW() - INTERVAL '{days}' DAY AND blob2 != 'smoke-test'
ORDER BY timestamp LIMIT {limit} FORMAT JSON"""


def fetch_rows(days: int, limit: int = 100_000, dataset: str = DATASET, opener=urllib.request.urlopen) -> list[dict]:
    account, token = os.environ.get("CF_ACCOUNT_ID"), os.environ.get("CF_ANALYTICS_TOKEN")
    if not account or not token:
        raise SystemExit("set CF_ACCOUNT_ID and CF_ANALYTICS_TOKEN (Account Analytics: Read) to read the fleet's episodes")
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{account}/analytics_engine/sql",
        data=QUERY.format(dataset=dataset, days=max(1, min(90, int(days))), limit=int(limit)).encode(),
        headers={"authorization": f"Bearer {token}"}, method="POST")
    with opener(req) as res:
        return json.loads(res.read().decode()).get("data", [])


def _num(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def proxy_success(row: dict) -> bool:
    """The run's work stands, and the user didn't correct it or ask again."""
    finished = _num(row.get("run_success")) >= 1
    rescued = row.get("applied") == "retry" and _num(row.get("followup_success")) >= 1
    return (finished or rescued) and row.get("user") not in BAD_REPLIES


def to_records(rows: list[dict]) -> list[bandit.Record]:
    """One Record per episode (repeated for sampled rows); rows whose logged
    action or probability doesn't fit the context are dropped."""
    records = []
    for row in rows:
        finished = _num(row.get("run_success")) >= 1
        applied = row.get("applied") or "none"
        p = _num(row.get("probability"), -1)
        r = bandit.Record(arm="fleet", scenario=str(row.get("customer") or "?"), model=row.get("model") or "unknown",
                          finished=finished, retriable=_num(row.get("retriable")) >= 1,
                          tool_calls=int(_num(row.get("tool_calls"))), action=applied, success=proxy_success(row),
                          tokens=int(max(0, _num(row.get("tokens"))) + max(0, _num(row.get("followup_tokens")))),
                          seconds=0.0)
        if applied not in r.actions or not 0 < p <= 1:
            continue  # e.g. a verify logged on a dead run, or a malformed probability
        other = next(a for a in r.actions if a != applied)
        r.behavior = {applied: p, other: 1 - p}
        records.extend([r] + [bandit.Record(**{**r.__dict__}) for _ in range(max(0, int(_num(row.get("sample_interval"), 1)) - 1))])
    return records


def cap_customers(records: list[bandit.Record], cap: float, seed: int = 0) -> tuple[list[bandit.Record], dict]:
    """Subsample any customer above `cap` of the records. Returns the kept
    records and how many were dropped per customer."""
    by = defaultdict(list)
    for r in records:
        by[r.scenario].append(r)
    if len(by) < 2:
        return records, {}
    rng = random.Random(seed)
    sizes = {c: len(rs) for c, rs in by.items()}
    for _ in range(len(sizes)):
        total = sum(sizes.values())
        over = [c for c, n in sizes.items() if n > cap * total]
        if not over:
            break
        c = max(over, key=lambda k: sizes[k])
        others = total - sizes[c]
        sizes[c] = max(1, int(cap * others / (1 - cap)))
    kept, dropped = [], {}
    for c, rs in by.items():
        if sizes[c] < len(rs):
            dropped[c] = len(rs) - sizes[c]
            rs = rng.sample(rs, sizes[c])
        kept.extend(rs)
    return kept, dropped


def assign_folds(records: list[bandit.Record], min_customers: int = 5, buckets: int = 5) -> None:
    """learn.bandit cross-fits by Record.scenario. By customer when there are
    enough; otherwise split each customer into buckets (in order)."""
    customers = sorted({r.scenario for r in records})
    if len(customers) >= min_customers:
        return
    seen = Counter()
    for r in records:
        seen[r.scenario] += 1
        r.scenario = f"{r.scenario}#{seen[r.scenario] % buckets}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--from-json", type=Path, help="rows from an earlier query (a JSON list, or {data: [...]})")
    parser.add_argument("--cap", type=float, default=0.25, help="largest share of records one customer may have")
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--export", type=Path, help="write a learned /intervene table (never uploaded here)")
    parser.add_argument("--allow-few-customers", action="store_true",
                        help=f"export even with fewer than {MIN_CUSTOMERS} customers (the table then reflects them only)")
    args = parser.parse_args()

    if args.from_json:
        raw = json.loads(args.from_json.read_text(encoding="utf-8"))
        rows = raw.get("data", []) if isinstance(raw, dict) else raw
    else:
        rows = fetch_rows(args.days)
    records = to_records(rows)
    customers = Counter(r.scenario for r in records)
    records, dropped = cap_customers(records, args.cap)
    print(f"{len(rows)} episodes from {len(customers)} customer(s) -> {len(records)} usable records"
          + (f"; capped at {args.cap:.0%}: dropped {dropped}" if dropped else ""))
    sup = bandit.support(records)
    print("logged examples: " + ", ".join(f"{k} {v}" for k, v in sup.items()))
    if len(records) < 20 or sup["finished/verify"] < 5:
        raise SystemExit("too few episodes with a check-your-work turn to evaluate yet")
    assign_folds(records)
    text, _ = bandit.report(records, args.lam, 0.0, args.rounds)
    print("Success here is a proxy (work stands and no correction), not a graded result.\n")
    print("\n".join(text.splitlines()[2:]))
    if args.export:
        if len(customers) < MIN_CUSTOMERS and not args.allow_few_customers:
            raise SystemExit(f"not exporting: {len(customers)} customer(s) is too few for a shared table "
                             f"(needs {MIN_CUSTOMERS}; --allow-few-customers overrides)")
        table = bandit.export_policy(records, args.lam, args.epsilon)
        table["source"] = {"kind": "fleet-proxy", "customers": len(customers), "cap": args.cap, "days": args.days}
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(table, indent=1), encoding="utf-8")
        print(f"\nwrote {args.export} (not uploaded)")


if __name__ == "__main__":
    main()
