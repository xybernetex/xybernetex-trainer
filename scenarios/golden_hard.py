"""Golden solutions for the hard scenarios: every outcome check must pass a
correct solution and reject a plausible wrong one. Runs the checks for real,
sandbox included (needs Docker). Rerun after changing a hard task or check:

    python -m scenarios.golden_hard
"""
import csv, hashlib, io, json, shutil, sys, tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

from scenarios.outcomes import evaluate, _deep_merge, _canonical_email
from scenarios.suite import by_id

S = by_id()
seed = lambda sid, name: S[sid]["files"][name]

ROMAN = '''import re
_V = [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),(50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]
def to_roman(n):
    if not isinstance(n, int) or not 1 <= n <= 3999: raise ValueError(n)
    out = ""
    for v, s in _V:
        while n >= v: out += s; n -= v
    return out
_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
def from_roman(s):
    if not s or not _RE.match(s): raise ValueError(s)
    n = i = 0
    for v, sym in _V:
        while s.startswith(sym, i): n += v; i += len(sym)
    return n
'''
DURATIONS = '''import re
_U = {"d": 86400, "h": 3600, "m": 60, "s": 1}
def parse_duration(s):
    t = s.strip().lower().replace(" ", "")
    if not t or not re.fullmatch(r"(\\d+(\\.\\d+)?[dhms])+", t): raise ValueError(s)
    seen, total = set(), 0.0
    for num, unit in re.findall(r"(\\d+(?:\\.\\d+)?)([dhms])", t):
        if unit in seen: raise ValueError(s)
        seen.add(unit); total += float(num) * _U[unit]
    return int(round(total))
'''
BRACKETS = '''def is_balanced(s):
    stack, quote, i = [], None, 0
    pairs = {")": "(", "]": "[", "}": "{"}
    while i < len(s):
        c = s[i]
        if quote:
            if c == "\\\\": i += 2; continue
            if c == quote: quote = None
        elif c in "\\"'": quote = c
        elif c in "([{": stack.append(c)
        elif c in ")]}":
            if not stack or stack.pop() != pairs[c]: return False
        i += 1
    return not stack and quote is None
'''
REPORT = '''def _ranked(scores):
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
def summary(scores):
    return ", ".join(f"{k}={v}" for k, v in _ranked(scores))
def top_n(scores, n):
    return [k for k, _ in _ranked(scores)][:n]
'''
INTEREST = '''from decimal import Decimal, ROUND_HALF_UP
b, r = Decimal("1000.00"), Decimal("0.05") / 12
for _ in range(12):
    b = (b * (1 + r)).quantize(Decimal("0.01"), ROUND_HALF_UP) + Decimal("100.00")
print(f"{b:.2f}")
'''


def business_days():
    hol = {date.fromisoformat(d) for d in seed("h-business-days", "holidays.txt").split()}
    out = "start,end,business_days\n"
    for row in csv.DictReader(io.StringIO(seed("h-business-days", "requests.csv"))):
        d, e, n = date.fromisoformat(row["start"]), date.fromisoformat(row["end"]), 0
        while d <= e:
            n += d.weekday() < 5 and d not in hol; d += timedelta(days=1)
        out += f"{row['start']},{row['end']},{n}\n"
    return out


def sessions():
    ev = sorted((datetime.fromisoformat(t), u, a) for t, u, a in (l.split() for l in seed("h-sessions", "events.log").splitlines()))
    tot, op = {}, {}
    for t, u, a in ev:
        if a == "login": op[u] = t
        else: tot[u] = tot.get(u, 0) + int((t - op.pop(u)).total_seconds() // 60)
    return json.dumps(tot)


def manifest_files():
    files = {f"data/part-{n:02d}.txt": "".join(f"{n}-{i}\n" for i in range(1, n + 1)) for n in range(1, 21)}
    files["manifest.txt"] = "".join(f"{p} {p and c.count(chr(10))} {hashlib.sha256(c.encode()).hexdigest()}\n" for p, c in files.items())
    return files


GOOD = {
    "h-median-edges": {"stats2.py": "def median(xs):\n    if not xs: raise ValueError('empty')\n    s = sorted(xs); m = len(s) // 2\n    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2\n"},
    "h-csv-revenue": {"revenue.txt": "79.98\n"},
    "h-business-days": {"results.csv": business_days()},
    "h-deep-merge": {"merged.json": json.dumps(_deep_merge(json.loads(seed("h-deep-merge", "base.json")), json.loads(seed("h-deep-merge", "override.json"))))},
    "h-sessions": {"durations.json": sessions()},
    "h-silent-output": {"report.txt": "rows: 7\n"},
    "h-roman": {"roman.py": ROMAN},
    "h-emails": {"unique.txt": "\n".join(sorted({e for e in map(_canonical_email, seed("h-emails", "signups.txt").splitlines()) if e})) + "\n"},
    "h-sql-top": {"top.txt": "Ana\n"},
    "h-duration-parse": {"durations.py": DURATIONS},
    "h-fix-tests": {"report.py": REPORT},
    "h-wordfreq": None,  # filled below from the checker's own definition, then checked by hand
    "h-brackets": {"brackets.py": BRACKETS},
    "h-reconcile": {"mismatches.csv": "id,issue\n2,amount_mismatch\n3,missing_in_bank\n4,missing_in_ledger\n11,missing_in_ledger\n"},
    "h-interest": {"interest.py": INTEREST},
    "h-manifest": manifest_files(),
}
BAD = {  # plausible wrong answers each check must reject
    "h-median-edges": {},  # untouched buggy median
    "h-csv-revenue": {"revenue.txt": "84.98\n"},  # forgot to skip refunded
    "h-business-days": {"results.csv": "start,end,business_days\n2026-09-01,2026-09-30,22\n2026-12-24,2027-01-04,8\n2026-10-10,2026-10-11,0\n2026-11-20,2026-11-30,7\n"},
    "h-deep-merge": {"merged.json": json.dumps({**json.loads(seed("h-deep-merge", "base.json")), **json.loads(seed("h-deep-merge", "override.json"))})},
    "h-sessions": {"durations.json": json.dumps({"ada": 160, "linus": 45, "grace": 20})},
    "h-silent-output": {},
    "h-roman": {"roman.py": ROMAN.replace("if not s or not _RE.match(s): raise ValueError(s)", "pass")},
    "h-emails": {"unique.txt": "adalovelace@gmail.com\nlinus@kernel.org\n"},
    "h-sql-top": {"top.txt": "Cy\n"},
    "h-duration-parse": {"durations.py": DURATIONS.replace("if unit in seen: raise ValueError(s)", "pass")},
    "h-fix-tests": {"report.py": REPORT, "test_report.py": "import unittest\n"},
    "h-wordfreq": {"top5.txt": "the 5\n"},
    "h-brackets": {"brackets.py": "def is_balanced(s):\n    return s.count('(') == s.count(')')\n"},
    "h-reconcile": {"mismatches.csv": "id,issue\n2,amount_mismatch\n3,missing_in_bank\n4,missing_in_ledger\n"},
    "h-interest": {"interest.py": "print(round(1000*(1+0.05/12)**12 + 100*12, 2))\n"},
    "h-manifest": {k: v for k, v in manifest_files().items() if k != "data/part-20.txt"},
}


def run(sid, files):
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        for name, text in {**S[sid].get("files", {}), **files}.items():
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text(text, encoding="utf-8", newline="\n")
        return evaluate(S[sid], folder, None)


import re
words = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)*", seed("h-wordfreq", "text.txt").casefold())
counts = {}
for w in words: counts[w] = counts.get(w, 0) + 1
ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
print("wordfreq ranking (check by hand):", ranked[:8])
GOOD["h-wordfreq"] = {"top5.txt": "".join(f"{w} {n}\n" for w, n in ranked[:5])}

fails = 0
for sid in [s for s in S if s.startswith("h-")]:
    good, bad = run(sid, GOOD[sid]), run(sid, BAD[sid])
    ok = good.task is True and bad.task is False
    fails += not ok
    print(f"{'OK ' if ok else 'BAD'} {sid:18s} good={good.task} ({good.detail[:70]}) | bad={bad.task} ({bad.detail[:60]})")
print("ALL CHECKS VALID" if not fails else f"{fails} check(s) need fixing")
