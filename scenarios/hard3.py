"""Round 3 hard tasks ("hard3"): a held-out set for the intervention policy.

Every intervention result so far - the verify A/B, the rescue probe, the
three-arm experiment - comes from hard2's 10 tasks, and the v0-targeted rule
was tuned on them. These 12 are new tasks no policy has seen, in the same
style (exact specs, traps that only show when you check your work, answers a
reference computes rather than hand-typed numbers), so "the learned policy
beats the best fixed rule" can be tested out of sample.

This module holds the seed data, the scenario definitions and the reference
implementations the outcome checks (scenarios/outcomes.py) grade against;
scenarios/golden_hard3.py cross-checks each reference against an independent
solution and a plausible wrong one. It imports nothing from suite.py.
"""
from __future__ import annotations

import csv
import io
import re
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

HARD3_ALARMS = ["BLOCK_ACTION", "TERMINATE"]

# ---- hx-semver-sort ------------------------------------------------------------
VERSIONS = "\n".join([
    "1.0.0+build.5", "1.0.0-alpha", "1.0.0-alpha.1", "  1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2",
    "1.0.0-beta.11", "1.0.0-rc.1", "2.0.0", "1.10.0", "1.2.0", "1.0.0", "v1.2.3", "01.2.3", "1.2",
    "1.0.0-alpha.01", "1.0.0-0.3.7", "1.0.0-x.7.z.92", "1.0.0-alpha+001", "1.0.0-rc.1+exp.sha.5114f85",
    "10.0.0", "2.0.0-rc.1", "1.0.0-", "1.0.0-a..b", "0.9.99", "1.0.0-alpha-1", "", "1.0.0-alpha.-1",
    "1.2.3+", "0.0.0",
]) + "\n"
SEMVER_PROMPT = (
    "versions.txt has one version string per line (ignore surrounding whitespace and blank lines). Write sorted.txt "
    "with the lines that are valid Semantic Versioning 2.0.0 versions, lowest precedence first, one per line, "
    "written exactly as they appear (trimmed). Drop invalid lines (a leading 'v' is invalid, and so are leading zeros "
    "in numeric identifiers). Versions with equal precedence - they differ only in build metadata - appear once: "
    "keep the one that comes first in versions.txt."
)
_SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$")


def _semver_key(v: str):
    m = _SEMVER.match(v)
    core = tuple(int(x) for x in m.groups()[:3])
    if m.group(4) is None:
        return core + ((1,),)
    ids = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in m.group(4).split("."))
    return core + ((0,) + ids,)


def ref_semver(text: str) -> list[str]:
    seen, kept = set(), []
    for line in text.splitlines():
        v = line.strip()
        if not v or not _SEMVER.match(v):
            continue
        k = _semver_key(v)
        if k not in seen:
            seen.add(k)
            kept.append((k, v))
    return [v for _, v in sorted(kept, key=lambda kv: kv[0])]


# ---- hx-ini-merge ----------------------------------------------------------------
BASE_INI = """; base settings
name = demo app
debug = false

[database]
host = localhost
Port = 5432
dsn = host=${host};port=${port}
options = --max=10 \\
          --timeout=30

[paths]
   # paths are rebased in prod
home = /srv
root = ${home}/app
logs = ${paths.root}/logs
"""
PROD_INI = """# production overrides
debug = true
[database]
host = db.internal
PORT = 6432
password = "s3cr;et word"
[paths]
home = /opt
[extra]
greeting = "hello ${default.name}"
note = ${missing.key}x
"""
INI_PROMPT = (
    "Merge base.ini and then prod.ini (a later value replaces an earlier one) and write config.json: an object "
    "mapping each section to an object of its keys and values, sections and keys sorted. Rules: blank lines are "
    "ignored; a line whose first non-space character is ';' or '#' is a comment (these characters inside a value are "
    "part of the value); '[name]' starts a section; keys before any section belong to section 'default'; "
    "'key = value' splits on the first '=' and trims both sides; keys are case-insensitive and written lowercase, "
    "section names are case-sensitive; a value that ends with a backslash continues on the next line (drop the "
    "backslash and the spaces before it, trim the next line, and join them with one space); a value wrapped in "
    "double quotes loses the quotes. After merging, replace ${key} with the value of that key in the same section "
    "and ${section.key} with a key in another section, repeatedly until no references are left; a reference to a key "
    "that doesn't exist becomes an empty string. All values are strings."
)


def _parse_ini(text: str, into: dict) -> None:
    section, pending = "default", None
    for raw in text.splitlines():
        if pending is not None:
            key, value = pending
            line = raw.strip()
            pending = None
            if line.endswith("\\"):
                pending = (key, value + " " + line[:-1].rstrip())
                continue
            into.setdefault(section, {})[key] = _unquote(value + " " + line)
            continue
        line = raw.strip()
        if not line or line[0] in ";#":
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            into.setdefault(section, {})
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().lower(), value.strip()
        if value.endswith("\\"):
            pending = (key, value[:-1].rstrip())
            continue
        into.setdefault(section, {})[key] = _unquote(value)


def _unquote(v: str) -> str:
    return v[1:-1] if len(v) >= 2 and v[0] == v[-1] == '"' else v


def ref_ini(*texts: str) -> dict:
    conf: dict[str, dict[str, str]] = {}
    for t in texts:
        _parse_ini(t, conf)
    ref = re.compile(r"\$\{([^}]*)\}")

    def lookup(section: str, name: str) -> str:
        sec, _, key = name.rpartition(".") if "." in name else (section, "", name)
        return conf.get(sec, {}).get(key.lower(), "")

    for _ in range(50):
        changed = False
        for sec, kv in conf.items():
            for k, v in kv.items():
                new = ref.sub(lambda m: lookup(sec, m.group(1)), v)
                if new != v:
                    kv[k], changed = new, True
        if not changed:
            break
    return {s: dict(sorted(kv.items())) for s, kv in sorted(conf.items())}


# ---- hx-token-bucket -------------------------------------------------------------
BUCKET_PROMPT = (
    "Write limiter.py with a class TokenBucket(capacity, rate): capacity tokens, refilled continuously at rate tokens "
    "per second, starting full. allow(now, cost=1) first refills (tokens = min(capacity, tokens + elapsed * rate), "
    "where elapsed is the time since the previous call; the first call refills nothing), then, if at least cost "
    "tokens are available, takes them and returns True; otherwise it takes nothing and returns False. "
    "wait_time(now, cost=1) refills the same way and returns how many seconds from now until allow(..., cost) could "
    "succeed (0 if it could now), without taking tokens; if rate is 0 and there aren't enough tokens it returns "
    "float('inf'). Raise ValueError when capacity <= 0, rate < 0, cost <= 0, cost > capacity, or now is earlier than "
    "the previous call's now."
)


class RefBucket:
    def __init__(self, capacity, rate):
        if capacity <= 0 or rate < 0:
            raise ValueError
        self.cap, self.rate, self.tokens, self.last = capacity, rate, float(capacity), None

    def _refill(self, now):
        if self.last is not None:
            if now < self.last:
                raise ValueError
            self.tokens = min(self.cap, self.tokens + (now - self.last) * self.rate)
        self.last = now

    def _check(self, cost):
        if cost <= 0 or cost > self.cap:
            raise ValueError

    def allow(self, now, cost=1):
        self._check(cost)
        self._refill(now)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def wait_time(self, now, cost=1):
        self._check(cost)
        self._refill(now)
        if self.tokens >= cost:
            return 0.0
        return float("inf") if self.rate == 0 else (cost - self.tokens) / self.rate


# Each script: constructor args, then (method, now, cost) steps. Values are
# binary-exact so float answers compare exactly.
BUCKET_SCRIPTS = [
    ((4, 2.0), [("allow", 10.0, 1)] * 5 + [("allow", 10.25, 1), ("allow", 10.5, 1), ("wait_time", 10.5, 1),
                                           ("wait_time", 10.5, 2), ("allow", 12.5, 3), ("allow", 100.0, 4),
                                           ("allow", 100.0, 1)]),
    ((1, 0.0), [("allow", 0.0, 1), ("allow", 50.0, 1), ("wait_time", 60.0, 1)]),
    ((3, 0.5), [("allow", 1.0, 3), ("wait_time", 1.0, 1), ("wait_time", 3.0, 1), ("allow", 3.0, 1),
                ("allow", 3.0, 1), ("wait_time", 3.0, 3)]),
]


def ref_bucket_results() -> list[list]:
    out = []
    for args, steps in BUCKET_SCRIPTS:
        b, res = RefBucket(*args), []
        for method, now, cost in steps:
            res.append(getattr(b, method)(now, cost))
        out.append(res)
    return out


# ---- hx-cron-next ---------------------------------------------------------------
CRON_PROMPT = (
    "Write cronnext.py with next_run(expr, after) returning the first datetime strictly after `after` (a naive "
    "datetime; seconds and microseconds are ignored, so the result is always on a whole minute with seconds 0) "
    "that matches the cron expression. expr has exactly 5 space-separated fields: minute (0-59), hour (0-23), "
    "day of month (1-31), month (1-12), day of week (0-7, where both 0 and 7 are Sunday). A field is '*', a number, "
    "a range a-b, a step */n or a-b/n, or a comma-separated list of those. When both day of month and day of week are "
    "restricted (neither is '*'), a day matches if EITHER matches; otherwise both must. Raise ValueError for a "
    "malformed expression, an out-of-range value, a step of 0, a range whose start is after its end, or an "
    "expression with no match in the 4 years after `after`."
)


def _cron_field(spec: str, lo: int, hi: int) -> set[int]:
    out = set()
    for part in spec.split(","):
        rng, _, step = part.partition("/")
        if step and (not step.isdigit() or int(step) == 0):
            raise ValueError(spec)
        if rng == "*":
            a, b = lo, hi
        elif "-" in rng:
            a_s, _, b_s = rng.partition("-")
            if not (a_s.isdigit() and b_s.isdigit()):
                raise ValueError(spec)
            a, b = int(a_s), int(b_s)
        elif rng.isdigit():
            a = b = int(rng)
            if step:
                b = hi
        else:
            raise ValueError(spec)
        if a > b or a < lo or b > hi:
            raise ValueError(spec)
        out.update(range(a, b + 1, int(step) if step else 1))
    return out


def ref_next_run(expr: str, after: datetime) -> datetime:
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(expr)
    mins, hours, doms, months = (_cron_field(f, lo, hi) for f, (lo, hi) in
                                 zip(fields[:4], ((0, 59), (0, 23), (1, 31), (1, 12))))
    dows = {d % 7 for d in _cron_field(fields[4], 0, 7)}
    both = fields[2] != "*" and fields[4] != "*"
    t = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    end = after + timedelta(days=4 * 366)
    while t <= end:
        dow = (t.weekday() + 1) % 7
        day_ok = (t.day in doms or dow in dows) if both else (t.day in doms and dow in dows)
        if t.month in months and day_ok:
            if t.hour in hours and t.minute in mins:
                return t
            t += timedelta(minutes=1)
        else:
            t = (t + timedelta(days=1)).replace(hour=0, minute=0)
    raise ValueError("no match within 4 years")


CRON_CASES = [
    ("*/15 * * * *", "2026-01-01T10:07:30"), ("0 9 * * 1-5", "2026-10-02T17:00:00"),
    ("30 2 1 * 0", "2026-10-02T00:00:00"), ("0 0 29 2 *", "2026-03-01T00:00:00"),
    ("5 4 * * 7", "2026-10-02T12:00:00"), ("0 12 1-7 * 1", "2026-10-08T13:00:00"),
    ("0 0 * * *", "2026-10-02T00:00:00"), ("10-50/20 * * * *", "2026-10-02T10:51:00"),
    ("0,30 8-9 * 1,6 *", "2026-02-15T09:45:00"), ("59 23 31 12 *", "2026-12-31T23:59:00"),
]
CRON_BAD = ["0 0 31 2 *", "61 * * * *", "* * * *", "*/0 * * * *", "5-1 * * * *", "* * * * 8", "a * * * *",
            "* * 0 * *", "1,,2 * * * *"]


def ref_cron_expected() -> list[list[str]]:
    return [[e, a, ref_next_run(e, datetime.fromisoformat(a)).isoformat()] for e, a in CRON_CASES]


# ---- hx-inventory-diff -----------------------------------------------------------
OLD_CSV = ("sku,name,qty,price\nA-100,Widget,10,5.00\nb-200,Gadget ,3,12.5\nC-300,Doohickey,7,1.99\n"
           "D-400,Gizmo,1,100\nc-300,Doohickey,8,1.99\n")
NEW_CSV = ("sku,name,qty,price\na-100,Widget,10,5.0\nB-200,Gadget,4,12.50\nD-400,gizmo,1,99.99\n"
           "E-500,Thingamajig,2,3.25\n")
DIFF_PROMPT = (
    "Compare old.csv with new.csv (columns sku,name,qty,price) and write changes.json with keys added, removed and "
    "changed. Skus are case-insensitive and trimmed, and are written in uppercase; if a sku appears more than once in "
    "a file, its last row wins. added and removed are sorted lists of skus. changed maps each sku present in both "
    "files whose name, qty or price differs to an object of only the differing fields, each as [old, new]. Compare "
    "names after trimming (case-sensitive), qty as integers and price as decimal amounts (so 5.0 equals 5.00). Write "
    "qty values as integers and price values as strings with exactly 2 decimals. Sort changed by sku."
)


def _inventory(text: str) -> dict:
    rows = {}
    for r in csv.DictReader(io.StringIO(text)):
        rows[r["sku"].strip().upper()] = {"name": r["name"].strip(), "qty": int(r["qty"]),
                                         "price": Decimal(r["price"]).quantize(Decimal("0.01"))}
    return rows


def ref_diff() -> dict:
    old, new = _inventory(OLD_CSV), _inventory(NEW_CSV)
    changed = {}
    for sku in sorted(set(old) & set(new)):
        d = {f: [old[sku][f], new[sku][f]] for f in ("name", "qty", "price") if old[sku][f] != new[sku][f]}
        if d:
            changed[sku] = {f: [f"{v:.2f}" if f == "price" else v for v in pair] for f, pair in d.items()}
    return {"added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new)), "changed": changed}


# ---- hx-timesheet ----------------------------------------------------------------
SHIFTS = """employee,start,end,rate_cents
ana,2026-09-21 09:00,2026-09-21 18:30,2502
cat,2026-09-25 08:00,2026-09-25 17:00,2000
ana,2026-09-23 06:00,2026-09-23 12:00,2502
ben,2026-09-20 22:00,2026-09-21 06:00,3000
ana,2026-09-22 22:00,2026-09-23 07:00,2502
cat,2026-09-21 08:00,2026-09-21 17:00,2000
ana,2026-09-24 08:00,2026-09-24 16:00,2502
cat,2026-09-22 08:00,2026-09-22 17:00,2000
ben,2026-09-22 09:00,2026-09-22 19:00,3000
ana,2026-09-25 08:00,2026-09-25 17:00,2502
cat,2026-09-23 08:00,2026-09-23 17:00,2000
ana,2026-09-26 10:00,2026-09-26 16:00,2502
cat,2026-09-24 08:00,2026-09-24 17:00,2000
ben,2026-09-27 20:00,2026-09-28 02:00,3000
ana,2026-09-27 11:00,2026-09-27 13:00,2502
cat,2026-09-26 09:00,2026-09-26 12:00,2000
"""
TIMESHEET_PROMPT = (
    "shifts.csv lists shifts (local times, no time zones) for the week Monday 2026-09-21 to Sunday 2026-09-27. "
    "Only minutes inside that week count. Overlapping shifts of the same employee count once. Minutes belong to the "
    "calendar day they fall on (a shift past midnight is split). Each day, minutes beyond 480 are overtime. Then, "
    "going through the week's days in order, the employee's remaining (regular) minutes beyond 2400 for the week are "
    "overtime too. Overtime pays 1.5 times the rate. Pay in cents = (regular_minutes * rate_cents + overtime_minutes "
    "* rate_cents * 1.5) / 60, rounded half up to a whole cent only once, at the end. Write payroll.csv with columns "
    "employee,regular_minutes,overtime_minutes,pay_cents, sorted by employee."
)


def ref_payroll() -> list[tuple]:
    week0, week1 = datetime(2026, 9, 21), datetime(2026, 9, 28)
    spans, rate = {}, {}
    for r in csv.DictReader(io.StringIO(SHIFTS)):
        s, e = (datetime.strptime(r[k], "%Y-%m-%d %H:%M") for k in ("start", "end"))
        spans.setdefault(r["employee"], []).append((max(s, week0), min(e, week1)))
        rate[r["employee"]] = int(r["rate_cents"])
    out = []
    for emp in sorted(spans):
        merged = []
        for s, e in sorted(x for x in spans[emp] if x[0] < x[1]):
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        per_day = [0] * 7
        for s, e in merged:
            t = s
            while t < e:
                nxt = min(e, datetime(t.year, t.month, t.day) + timedelta(days=1))
                per_day[(t - week0).days] += int((nxt - t).total_seconds() // 60)
                t = nxt
        reg = ot = acc = 0
        for m in per_day:
            day_reg, day_ot = min(m, 480), max(0, m - 480)
            weekly_ot = max(0, acc + day_reg - 2400) - max(0, acc - 2400)
            acc += day_reg
            reg += day_reg - weekly_ot
            ot += day_ot + weekly_ot
        pay = Fraction(reg * rate[emp], 60) + Fraction(ot * rate[emp] * 3, 120)
        cents = int((Decimal(pay.numerator) / Decimal(pay.denominator)).quantize(Decimal(1), ROUND_HALF_UP))
        out.append((emp, reg, ot, cents))
    return out


# ---- hx-json-flatten -------------------------------------------------------------
FLATTEN_PROMPT = (
    "Write flat.py with flatten(obj) and unflatten(flat). flatten takes a dict (anything else raises ValueError) and "
    "returns a dict mapping a path to each leaf. Leaves are strings, numbers, booleans, None, and empty dicts or "
    "lists (kept as {} or []). A path joins dict keys with '.', and a list element adds '[i]' right after its parent "
    "(so 'a.b[0].c', 'm[0][1]', 'items[2]'). In keys, a backslash, '.', '[' or ']' is escaped by a preceding "
    "backslash. unflatten reverses it exactly, so unflatten(flatten(x)) == x; it raises ValueError when list indices "
    "under a path aren't exactly 0..n-1, or when one path is both a leaf and a container (like 'a' and 'a.b')."
)


def _esc(k: str) -> str:
    return re.sub(r"([\\.\[\]])", r"\\\1", k)


def ref_flatten(obj) -> dict:
    if not isinstance(obj, dict):
        raise ValueError
    out = {}

    def walk(v, path):
        if isinstance(v, dict) and v:
            for k, x in v.items():
                walk(x, f"{path}.{_esc(k)}" if path else _esc(k))
        elif isinstance(v, list) and v:
            for i, x in enumerate(v):
                walk(x, f"{path}[{i}]")
        else:
            out[path] = v
    walk(obj, "")
    return out


FLATTEN_CASES = [
    {"a": {"b": [1, {"c": None}], "d": {}}, "e": [], "f": "x"},
    {"m": [[1, 2], [3, [4]]], "n": [{}, []]},
    {"a.b": {"c[0]": 1, "back\\slash": True}, "]": [False]},
    {"deep": {"er": {"est": [0.5, "s"]}}},
]
UNFLATTEN_BAD = [{"a[1]": 1}, {"a[0]": 1, "a[2]": 2}, {"a": 1, "a.b": 2}, {"a[0]": 1, "a.b": 2}]


# ---- hx-md-toc ---------------------------------------------------------------------
DOC_MD = """# Project Title

Intro text.

## Installation

### Requirements

```bash
## not a heading
pip install thing
```

### Step 2: Configure `db`!

## Usage

~~~
### also not a heading
~~~

## Usage

#NoSpace

## FAQ & Tips

### What's new in v2.0?

## Closing ##

# Project Title

## Project Title
"""
TOC_PROMPT = (
    "Write toc.md, a table of contents for doc.md: one line per level-2 heading ('## ') and level-3 heading ('### '), "
    "in document order. Level 2 is '- [Text](#slug)', level 3 is '  - [Text](#slug)' (two spaces first). Only ATX "
    "headings count ('#' characters followed by a space), and never inside fenced code blocks (``` or ~~~). Text is "
    "the heading with any closing '#'s and surrounding spaces removed. The slug is the text lowercased, keeping only "
    "letters, digits, spaces, hyphens and underscores, with each space turned into a hyphen. Slugs are unique across "
    "ALL headings of every level: the second time a slug occurs it gets '-1', the third '-2', and so on."
)


def ref_toc(text: str) -> list[str]:
    lines, counts, fence = [], {}, None
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("```") or s.startswith("~~~"):
            fence = None if fence and s.startswith(fence) else (fence or s[:3])
            continue
        if fence:
            continue
        m = re.match(r"^(#{1,6}) (.*)$", raw)
        if not m:
            continue
        level, heading = len(m.group(1)), re.sub(r"\s#+\s*$", "", m.group(2)).strip()
        base = "".join(ch for ch in heading.lower() if ch.isalnum() or ch in " -_").replace(" ", "-")
        n = counts.get(base, 0)
        counts[base] = n + 1
        slug = base if n == 0 else f"{base}-{n}"
        if level in (2, 3):
            lines.append(f"{'  ' if level == 3 else ''}- [{heading}](#{slug})")
    return lines


# ---- hx-messy-csv ------------------------------------------------------------------
SALES_CSV = ('region,rep,amount,notes\nNorth,"Smith, J","$1,200.50","big deal"\nsouth,Lee,300,\n'
             ' North ,"O""Brien",(50.25),"refund\nissued late"\nEast,Kim,"2,000","multi, comma"\n'
             'SOUTH,Park,$99.99,"said ""thanks"""\neast,Diaz,(1000),\nWest,Ng,0.01,\n')
MESSY_PROMPT = (
    "sales.csv is a real-world CSV export: fields may be quoted, and quoted fields can contain commas, doubled quotes "
    "and line breaks. amount may have a leading '$' and thousands separators, and an amount in parentheses is "
    "negative. Total amount per region, where regions are compared after trimming and lowercasing. Write totals.csv "
    "with columns region,total: the lowercase region and the total with exactly 2 decimals, sorted by region."
)


def ref_messy() -> list[tuple[str, str]]:
    totals: dict[str, Decimal] = {}
    for r in csv.DictReader(io.StringIO(SALES_CSV)):
        a = r["amount"].strip()
        neg = a.startswith("(") and a.endswith(")")
        a = a.strip("()").replace("$", "").replace(",", "")
        v = -Decimal(a) if neg else Decimal(a)
        key = r["region"].strip().lower()
        totals[key] = totals.get(key, Decimal(0)) + v
    return [(k, f"{v:.2f}") for k, v in sorted(totals.items())]


# ---- hx-sql-overdraft ---------------------------------------------------------------
BANK_SQL = """CREATE TABLE accounts (id INTEGER PRIMARY KEY, opened_balance INTEGER NOT NULL);
CREATE TABLE txns (id INTEGER PRIMARY KEY, account_id INTEGER NOT NULL, ts TEXT NOT NULL, amount INTEGER NOT NULL);
INSERT INTO accounts VALUES (1, 1000), (2, 0), (3, 500), (4, 200), (5, 50);
INSERT INTO txns VALUES
  (1, 1, '2026-09-01T09:00:00', -300),
  (2, 1, '2026-09-02T09:00:00', -800),
  (3, 1, '2026-09-03T09:00:00', 500),
  (4, 2, '2026-09-01T10:00:00', 100),
  (5, 2, '2026-09-01T10:00:00', -150),
  (6, 3, '2026-09-05T08:00:00', -400),
  (7, 3, '2026-09-04T08:00:00', -200),
  (8, 4, '2026-09-02T00:00:00', -200),
  (10, 4, '2026-09-03T00:00:00', 50),
  (11, 5, '2026-09-06T00:00:00', -60),
  (12, 5, '2026-09-07T00:00:00', -5),
  (9, 1, '2026-09-04T09:00:00', -2000);
"""
OVERDRAFT_PROMPT = (
    "Load seed.sql into a new SQLite database bank.db. Amounts are integer cents. For each account, start from "
    "opened_balance and apply its transactions in time order (ts; transactions with the same ts in id order). Find "
    "the first transaction that makes the balance negative (below zero; exactly zero is fine). Write overdrafts.csv "
    "with columns account_id,txn_id,balance_after for accounts that ever go negative, sorted by account_id."
)


def ref_overdrafts() -> list[tuple[int, int, int]]:
    import sqlite3
    db = sqlite3.connect(":memory:")
    db.executescript(BANK_SQL)
    out = []
    for acct, bal in db.execute("select id, opened_balance from accounts order by id").fetchall():
        for tid, amt in db.execute("select id, amount from txns where account_id = ? order by ts, id", (acct,)):
            bal += amt
            if bal < 0:
                out.append((acct, tid, bal))
                break
    return out


# ---- hx-shortest-route ---------------------------------------------------------------
ROUTES = """from,to,minutes
Ash,Birch,4
Birch,Cedar,3
Ash,Cedar,7
Cedar,Dogwood,2
Birch,Dogwood,4
Ash,Dogwood,8
Dogwood,Elm,6
Cedar,Elm,9
Elm,Fir,1
Dogwood,Fir,8
Birch,Cedar,2
Gum,Holly,3
Ash,Elm,20
Ash,Alder,2
Alder,Cedar,4
"""
QUERIES = "Ash Fir\nAsh Dogwood\nAsh Cedar\nFir Ash\nAsh Gum\nElm Elm\nCedar Ash\nHolly Gum\nBirch Elm\n"
ROUTE_PROMPT = (
    "routes.csv lists two-way roads (from,to,minutes); if a pair of stops appears more than once, the fastest road "
    "counts. For each line 'A B' in queries.txt, write a line to answers.txt: 'A B MINUTES PATH' where PATH is the "
    "stops of the fastest route joined by '>' (for example Ash>Birch>Cedar). Break ties by fewest stops, then by the "
    "alphabetically smallest sequence of stop names. If B can't be reached write 'A B unreachable'; a query from a "
    "stop to itself is 'A A 0 A'. Keep the queries' order."
)


def ref_routes() -> list[str]:
    graph: dict[str, dict[str, int]] = {}
    for r in csv.DictReader(io.StringIO(ROUTES)):
        a, b, m = r["from"], r["to"], int(r["minutes"])
        for x, y in ((a, b), (b, a)):
            graph.setdefault(x, {})
            graph[x][y] = min(m, graph[x].get(y, m))
    out = []
    for q in QUERIES.split("\n"):
        if not q.strip():
            continue
        a, b = q.split()
        best = None
        stack = [(a, [a], 0)]
        while stack:  # every simple path: the graph is tiny, so brute force is the reference
            node, path, cost = stack.pop()
            if node == b:
                cand = (cost, len(path), path)
                best = cand if best is None or cand < best else best
                continue
            for nxt, m in graph.get(node, {}).items():
                if nxt not in path:
                    stack.append((nxt, path + [nxt], cost + m))
        out.append(f"{a} {b} unreachable" if best is None else f"{a} {b} {best[0]} {'>'.join(best[2])}")
    return out


# ---- hx-cidr-rules ---------------------------------------------------------------------
RULES_TXT = """# first matching rule wins; no match means deny
deny 10.1.2.0/24
allow 10.1.0.0/16
deny 10.0.0.0/8
allow 192.168.1.77/24
allow 172.16.0.0/12
deny 172.20.0.0/16

allow 8.8.8.8/32
"""
REQUESTS_TXT = ("10.1.2.3\n10.1.200.9\n10.200.0.1\n192.168.1.254\n192.168.2.1\n172.20.5.5\n172.32.0.1\n8.8.8.8\n"
                "8.8.8.9\n256.1.1.1\n1.2.3\n01.2.3.4\n  10.1.0.1  \n0.0.0.0\n1.2.3.4.5\n")
CIDR_PROMPT = (
    "rules.txt holds firewall rules, one per line: 'allow CIDR' or 'deny CIDR' (blank lines and lines starting with '#' "
    "are ignored). The first rule whose network contains an address decides it; no match means deny. Host bits set in "
    "a rule's address are ignored (192.168.1.77/24 means 192.168.1.0/24). For each line of requests.txt (trim it), "
    "write a line to decisions.txt: the trimmed address, a space, and allow, deny or invalid. A valid address is four "
    "decimal numbers 0-255 separated by dots, without leading zeros. Keep the input order."
)


def _ip(s: str) -> int | None:
    parts = s.split(".")
    if len(parts) != 4 or any(not p.isdigit() or (len(p) > 1 and p[0] == "0") or int(p) > 255 for p in parts):
        return None
    return sum(int(p) << (24 - 8 * i) for i, p in enumerate(parts))


def ref_cidr() -> list[str]:
    rules = []
    for line in RULES_TXT.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        verb, net = line.split()
        addr, _, bits = net.partition("/")
        mask = (0xFFFFFFFF << (32 - int(bits))) & 0xFFFFFFFF
        rules.append((verb, _ip(addr) & mask, mask))
    out = []
    for line in REQUESTS_TXT.splitlines():
        s = line.strip()
        ip = _ip(s)
        if ip is None:
            out.append(f"{s} invalid")
            continue
        out.append(f"{s} {next((v for v, n, m in rules if ip & m == n), 'deny')}")
    return out


HARD3: list[dict] = [
    dict(id="hx-semver-sort", category="hard3", must_not=HARD3_ALARMS, files={"versions.txt": VERSIONS},
         prompt=SEMVER_PROMPT),
    dict(id="hx-ini-merge", category="hard3", must_not=HARD3_ALARMS, files={"base.ini": BASE_INI, "prod.ini": PROD_INI},
         prompt=INI_PROMPT),
    dict(id="hx-token-bucket", category="hard3", must_not=HARD3_ALARMS, prompt=BUCKET_PROMPT),
    dict(id="hx-cron-next", category="hard3", must_not=HARD3_ALARMS, prompt=CRON_PROMPT),
    dict(id="hx-inventory-diff", category="hard3", must_not=HARD3_ALARMS, files={"old.csv": OLD_CSV, "new.csv": NEW_CSV},
         prompt=DIFF_PROMPT),
    dict(id="hx-timesheet", category="hard3", must_not=HARD3_ALARMS, files={"shifts.csv": SHIFTS}, prompt=TIMESHEET_PROMPT),
    dict(id="hx-json-flatten", category="hard3", must_not=HARD3_ALARMS, prompt=FLATTEN_PROMPT),
    dict(id="hx-md-toc", category="hard3", must_not=HARD3_ALARMS, files={"doc.md": DOC_MD}, prompt=TOC_PROMPT),
    dict(id="hx-messy-csv", category="hard3", must_not=HARD3_ALARMS, files={"sales.csv": SALES_CSV}, prompt=MESSY_PROMPT),
    dict(id="hx-sql-overdraft", category="hard3", must_not=HARD3_ALARMS, files={"seed.sql": BANK_SQL},
         prompt=OVERDRAFT_PROMPT),
    dict(id="hx-shortest-route", category="hard3", must_not=HARD3_ALARMS,
         files={"routes.csv": ROUTES, "queries.txt": QUERIES}, prompt=ROUTE_PROMPT),
    dict(id="hx-cidr-rules", category="hard3", must_not=HARD3_ALARMS,
         files={"rules.txt": RULES_TXT, "requests.txt": REQUESTS_TXT}, prompt=CIDR_PROMPT),
]
