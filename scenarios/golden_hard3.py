"""Golden solutions for the hard3 scenarios. Each good solution is written
independently of the reference in scenarios/hard3.py (a different algorithm
where possible: day jumping instead of minute stepping, a minute bitmap
instead of interval merging, Dijkstra instead of brute force, the ipaddress
module instead of bit masks), so agreement cross-checks the reference
against the spec text, not just the checker against itself. Each bad
solution is a plausible mistake the check must reject. Needs Docker. Run:

    python -m scenarios.golden_hard3
"""
import csv, heapq, io, ipaddress, json, re, sqlite3, tempfile
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from functools import cmp_to_key
from pathlib import Path

from scenarios import hard3 as H3
from scenarios.outcomes import evaluate
from scenarios.suite import by_id

S = by_id()


# ---- semver: hand parser + comparator ----------------------------------------------
def _valid_ident(p, numeric_ok_leading_zero=False):
    return bool(p) and all(c.isalnum() or c == "-" for c in p) and p.isascii() and \
        (numeric_ok_leading_zero or not (p.isdigit() and len(p) > 1 and p[0] == "0"))


def _parse_semver(v):
    core, plus, build = v.partition("+")
    if plus and not all(_valid_ident(b, True) for b in build.split(".")):
        return None
    core, dash, pre = core.partition("-")
    nums = core.split(".")
    if len(nums) != 3 or not all(n.isdigit() and n.isascii() and (n == "0" or n[0] != "0") for n in nums):
        return None
    if dash and not all(_valid_ident(p) for p in pre.split(".")):
        return None
    return [int(n) for n in nums], pre.split(".") if dash else None


def _cmp_semver(a, b):
    (na, pa), (nb, pb) = a, b
    if na != nb:
        return -1 if na < nb else 1
    if pa is None or pb is None:
        return (pa is None) - (pb is None) if (pa is None) != (pb is None) else 0
    for x, y in zip(pa, pb):
        if x == y:
            continue
        if x.isdigit() and y.isdigit():
            return -1 if int(x) < int(y) else 1
        if x.isdigit() != y.isdigit():
            return -1 if x.isdigit() else 1
        return -1 if x < y else 1
    return (len(pa) > len(pb)) - (len(pa) < len(pb))


def semver_good():
    kept = []
    for line in H3.VERSIONS.splitlines():
        v = line.strip()
        p = v and _parse_semver(v)
        if p and not any(_cmp_semver(p, q) == 0 for q, _ in kept):
            kept.append((p, v))
    kept.sort(key=cmp_to_key(lambda a, b: _cmp_semver(a[0], b[0])))
    return "\n".join(v for _, v in kept) + "\n"


def semver_bad():  # prerelease compared as plain strings: beta.11 < beta.2
    good = semver_good().split()
    return "\n".join(sorted(good, key=lambda v: (_parse_semver(v)[0], _parse_semver(v)[1] is None, v))) + "\n"


# ---- ini: regex line classifier, then a fixed-point substitution ------------------
def ini_good(strip_inline_comments=False):
    conf = {}
    for text in (H3.BASE_INI, H3.PROD_INI):
        _ini_file(text, conf, strip_inline_comments)
    return _ini_resolve(conf)


def _ini_file(text, conf, strip_inline_comments):
    sec, buf = "default", None  # each file starts in 'default'
    for raw in text.splitlines():
        if buf is not None:
            buf[1].append(raw.strip())
            if raw.rstrip().endswith("\\"):
                continue
            key, parts = buf
            buf = None
            joined = " ".join(re.sub(r"\s*\\$", "", p) for p in parts)
            conf.setdefault(sec, {})[key] = joined[1:-1] if len(joined) > 1 and joined[0] == joined[-1] == '"' else joined
            continue
        if re.match(r"^\s*([;#]|$)", raw):
            continue
        if m := re.match(r"^\s*\[(.+)\]\s*$", raw):
            sec = m[1].strip()
            conf.setdefault(sec, {})
            continue
        k, v = raw.split("=", 1)
        if strip_inline_comments:
            v = v.split(";")[0]
        k, v = k.strip().lower(), v.strip()
        if v.endswith("\\"):
            buf = (k, [v])
            continue
        conf.setdefault(sec, {})[k] = v[1:-1] if len(v) > 1 and v[0] == v[-1] == '"' else v


def _ini_resolve(conf):
    while True:
        done = True
        for s, kv in conf.items():
            for k, v in kv.items():
                m = re.search(r"\$\{([^}]*)\}", v)
                if m:
                    target = m[1].split(".", 1) if "." in m[1] else [s, m[1]]
                    kv[k] = v[:m.start()] + conf.get(target[0], {}).get(target[1].lower(), "") + v[m.end():]
                    done = False
        if done:
            break
    return json.dumps({s: dict(sorted(kv.items())) for s, kv in sorted(conf.items())}, indent=2)


# ---- token bucket -------------------------------------------------------------------
BUCKET_GOOD = '''class TokenBucket:
    def __init__(self, capacity, rate):
        if capacity <= 0 or rate < 0: raise ValueError("bad bucket")
        self.capacity, self.rate, self.level, self.t = capacity, rate, capacity, None
    def _tick(self, now, cost):
        if cost <= 0 or cost > self.capacity: raise ValueError("bad cost")
        if self.t is not None:
            if now < self.t: raise ValueError("time went backwards")
            self.level = min(self.capacity, self.level + (now - self.t) * self.rate)
        self.t = now
    def allow(self, now, cost=1):
        self._tick(now, cost)
        if self.level < cost: return False
        self.level -= cost
        return True
    def wait_time(self, now, cost=1):
        self._tick(now, cost)
        missing = cost - self.level
        if missing <= 0: return 0.0
        return float("inf") if self.rate == 0 else missing / self.rate
'''
BUCKET_BAD = BUCKET_GOOD.replace("min(self.capacity, self.level + (now - self.t) * self.rate)",
                                 "self.level + (now - self.t) * self.rate")

# ---- cron: parse to sets, then jump by day / hour instead of stepping minutes -------
CRON_GOOD = '''from datetime import datetime, timedelta
def _field(s, lo, hi):
    vals = set()
    for item in s.split(","):
        base, sep, step = item.partition("/")
        if sep and (not step.isdigit() or int(step) < 1): raise ValueError(s)
        if base == "*": a, b = lo, hi
        elif base.count("-") == 1:
            x, y = base.split("-")
            if not (x.isdigit() and y.isdigit()): raise ValueError(s)
            a, b = int(x), int(y)
        elif base.isdigit(): a, b = int(base), (hi if sep else int(base))
        else: raise ValueError(s)
        if not (lo <= a <= b <= hi): raise ValueError(s)
        vals |= set(range(a, b + 1, int(step) if sep else 1))
    return vals
def next_run(expr, after):
    f = expr.split(" ")
    if len(f) != 5 or any(not x for x in f): raise ValueError(expr)
    mi, ho, dm, mo = _field(f[0], 0, 59), _field(f[1], 0, 23), _field(f[2], 1, 31), _field(f[3], 1, 12)
    dw = {7 if d == 0 else d for d in _field(f[4], 0, 7)}
    either = f[2] != "*" and f[4] != "*"
    start = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    day = start.replace(hour=0, minute=0)
    for i in range(4 * 366 + 2):
        d = day + timedelta(days=i)
        wd = d.isoweekday()
        ok = (d.day in dm or wd in dw) if either else (d.day in dm and wd in dw)
        if d.month not in mo or not ok: continue
        for h in sorted(ho):
            for m in sorted(mi):
                t = d.replace(hour=h, minute=m)
                if t >= start:
                    if t > after + timedelta(days=4 * 366): raise ValueError("none within 4 years")
                    return t
    raise ValueError("none within 4 years")
'''
CRON_BAD = CRON_GOOD.replace("(d.day in dm or wd in dw) if either else ", "")

# ---- inventory diff -----------------------------------------------------------------
def diff_good(price_as_text=False):
    def load(text):
        out = {}
        for r in csv.reader(io.StringIO(text.strip())):
            if r[0] == "sku":
                continue
            out[r[0].strip().upper()] = (r[1].strip(), int(r[2]), r[3] if price_as_text else Decimal(r[3]))
        return out
    o, n = load(H3.OLD_CSV), load(H3.NEW_CSV)
    fmt = lambda f, v: f"{Decimal(v):.2f}" if f == "price" else v  # noqa: E731
    changed = {}
    for sku in sorted(o.keys() & n.keys()):
        diffs = {f: [fmt(f, o[sku][i]), fmt(f, n[sku][i])] for i, f in enumerate(("name", "qty", "price"))
                 if o[sku][i] != n[sku][i]}
        if diffs:
            changed[sku] = diffs
    return json.dumps({"added": sorted(n.keys() - o.keys()), "removed": sorted(o.keys() - n.keys()), "changed": changed})


# ---- timesheet: a set of worked minute slots per employee ---------------------------
def payroll_good(merge_overlaps=True):
    week0 = datetime(2026, 9, 21)
    worked, rates, raw = {}, {}, {}
    for r in csv.DictReader(io.StringIO(H3.SHIFTS)):
        s = int((datetime.fromisoformat(r["start"]) - week0).total_seconds() // 60)
        e = int((datetime.fromisoformat(r["end"]) - week0).total_seconds() // 60)
        slots = range(max(s, 0), min(e, 7 * 1440))
        worked.setdefault(r["employee"], set()).update(slots)
        raw.setdefault(r["employee"], []).extend(slots)
        rates[r["employee"]] = int(r["rate_cents"])
    lines = ["employee,regular_minutes,overtime_minutes,pay_cents"]
    for emp in sorted(worked):
        mins = worked[emp] if merge_overlaps else raw[emp]
        days = [sum(1 for m in mins if m // 1440 == d) for d in range(7)]
        reg = ot = 0
        for d in days:
            r_today = min(d, 480)
            spill = max(0, reg + r_today - 2400)
            reg += r_today - spill
            ot += d - r_today + spill
        pay = (Decimal(reg * rates[emp]) + Decimal(ot * rates[emp]) * Decimal("1.5")) / 60
        lines.append(f"{emp},{reg},{ot},{pay.quantize(Decimal(1), ROUND_HALF_UP)}")
    return "\n".join(lines) + "\n"


# ---- json flatten -------------------------------------------------------------------
FLAT_GOOD = r'''import re
def _k(k):
    return "".join("\\" + c if c in "\\.[]" else c for c in k)
def flatten(obj):
    if not isinstance(obj, dict): raise ValueError("need a dict")
    out = {}
    stack = [("", obj)]
    while stack:
        path, v = stack.pop()
        if isinstance(v, dict) and v:
            for k in reversed(list(v)): stack.append(((path + "." if path else "") + _k(k), v[k]))
        elif isinstance(v, list) and v:
            for i in reversed(range(len(v))): stack.append((f"{path}[{i}]", v[i]))
        else:
            out[path] = v
    return out
def _steps(path):
    steps, i, cur = [], 0, None
    while i < len(path):
        c = path[i]
        if c == "\\":
            cur = (cur or "") + path[i + 1]; i += 2; continue
        if c == ".":
            steps.append(cur); cur = None; i += 1; continue
        if c == "[":
            if cur is not None: steps.append(cur); cur = None
            j = path.index("]", i); steps.append(int(path[i + 1:j])); i = j + 1
            if i < len(path) and path[i] == ".": i += 1
            continue
        cur = (cur or "") + c; i += 1
    if cur is not None: steps.append(cur)
    return steps
LEAF = object()
def unflatten(flat):
    tree = {}
    for path, value in flat.items():
        node, steps = tree, _steps(path)
        for s, nxt in zip(steps, steps[1:] + [LEAF]):
            if nxt is LEAF:
                if s in node: raise ValueError(path)
                node[s] = ("leaf", value)
            else:
                child = node.setdefault(s, {})
                if isinstance(child, tuple): raise ValueError(path)
                node = child
    def build(node):
        if isinstance(node, tuple): return node[1]
        keys = list(node)
        if keys and all(isinstance(k, int) for k in keys):
            if sorted(keys) != list(range(len(keys))): raise ValueError("gap in list")
            return [build(node[k]) for k in range(len(keys))]
        if any(isinstance(k, int) for k in keys): raise ValueError("mixed")
        return {k: build(node[k]) for k in keys}
    return build(tree)
'''
FLAT_BAD = FLAT_GOOD.replace('return "".join("\\\\" + c if c in "\\\\.[]" else c for c in k)', "return k")

# ---- md toc -----------------------------------------------------------------------
def toc_good(skip_fences=True):
    out, seen, in_code = [], {}, False
    for line in H3.DOC_MD.split("\n"):
        if skip_fences and re.match(r"^\s*(```|~~~)", line):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = re.match(r"^(#+) +(.+?)(?: +#+)? *$", line)
        if not m or len(m[1]) > 6:
            continue
        text = m[2]
        slug = re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        if len(m[1]) in (2, 3):
            out.append(("  " if len(m[1]) == 3 else "") + f"- [{text}](#{slug}{'-' + str(n) if n else ''})")
    return "\n".join(out) + "\n"


# ---- messy csv ------------------------------------------------------------------------
def messy_good(parens_negative=True):
    tot = {}
    for row in list(csv.reader(io.StringIO(H3.SALES_CSV)))[1:]:
        region, amt = row[0].strip().lower(), row[2].strip()
        sign = -1 if parens_negative and amt.startswith("(") else 1
        tot[region] = tot.get(region, Decimal(0)) + sign * Decimal(re.sub(r"[^\d.]", "", amt))
    return "region,total\n" + "".join(f"{k},{v:.2f}\n" for k, v in sorted(tot.items()))


# ---- overdraft: a window query -------------------------------------------------------
def overdraft_good(order="t.ts, t.id"):
    db = sqlite3.connect(":memory:")
    db.executescript(H3.BANK_SQL)
    rows = db.execute(f"""
        WITH run AS (SELECT t.account_id, t.id, a.opened_balance + SUM(t.amount) OVER
                       (PARTITION BY t.account_id ORDER BY {order} ROWS UNBOUNDED PRECEDING) AS bal,
                       ROW_NUMBER() OVER (PARTITION BY t.account_id ORDER BY {order}) AS n
                     FROM txns t JOIN accounts a ON a.id = t.account_id),
             firstneg AS (SELECT account_id, MIN(n) AS n FROM run WHERE bal < 0 GROUP BY account_id)
        SELECT r.account_id, r.id, r.bal FROM run r JOIN firstneg f USING (account_id, n) ORDER BY r.account_id""")
    return "account_id,txn_id,balance_after\n" + "".join(f"{a},{t},{b}\n" for a, t, b in rows)


# ---- routes: Dijkstra over (minutes, stops, path) -------------------------------------
def routes_good(tie_on_stops=True):
    g = {}
    for r in csv.DictReader(io.StringIO(H3.ROUTES)):
        for a, b in ((r["from"], r["to"]), (r["to"], r["from"])):
            g.setdefault(a, {})
            g[a][b] = min(int(r["minutes"]), g[a].get(b, 10 ** 9))
    lines = []
    for q in H3.QUERIES.split("\n"):
        if not q:
            continue
        src, dst = q.split()
        heap, done = [(0, 1 if tie_on_stops else 0, [src])], set()
        found = None
        while heap:
            cost, stops, path = heapq.heappop(heap)
            node = path[-1]
            if node in done:
                continue
            done.add(node)
            if node == dst:
                found = (cost, path)
                break
            for nxt, m in g.get(node, {}).items():
                if nxt not in done:
                    heapq.heappush(heap, (cost + m, stops + 1 if tie_on_stops else 0, path + [nxt]))
        lines.append(f"{src} {dst} " + (f"{found[0]} {'>'.join(found[1])}" if found else "unreachable"))
    return "\n".join(lines) + "\n"


# ---- cidr: the ipaddress module ---------------------------------------------------------
def cidr_good(ignore_host_bits=True):
    rules = []
    for line in H3.RULES_TXT.splitlines():
        if line.strip() and not line.strip().startswith("#"):
            verb, net = line.split()
            if ignore_host_bits:
                rules.append((verb, ipaddress.ip_network(net, strict=False)))
            else:
                addr, bits = net.split("/")
                rules.append((verb, (ipaddress.ip_address(addr), int(bits))))
    out = []
    for line in H3.REQUESTS_TXT.splitlines():
        s = line.strip()
        try:
            ip = ipaddress.IPv4Address(s)
        except ValueError:
            out.append(f"{s} invalid")
            continue
        verdict = "deny"
        for verb, net in rules:
            hit = ip in net if ignore_host_bits else (int(ip) >> (32 - net[1])) == (int(net[0]) >> (32 - net[1])) and \
                int(net[0]) & ((1 << (32 - net[1])) - 1) == 0
            if hit:
                verdict = verb
                break
        out.append(f"{s} {verdict}")
    return "\n".join(out) + "\n"


GOOD = {
    "hx-semver-sort": {"sorted.txt": semver_good()},
    "hx-ini-merge": {"config.json": ini_good()},
    "hx-token-bucket": {"limiter.py": BUCKET_GOOD},
    "hx-cron-next": {"cronnext.py": CRON_GOOD},
    "hx-inventory-diff": {"changes.json": diff_good()},
    "hx-timesheet": {"payroll.csv": payroll_good()},
    "hx-json-flatten": {"flat.py": FLAT_GOOD},
    "hx-md-toc": {"toc.md": toc_good()},
    "hx-messy-csv": {"totals.csv": messy_good()},
    "hx-sql-overdraft": {"overdrafts.csv": overdraft_good()},
    "hx-shortest-route": {"answers.txt": routes_good()},
    "hx-cidr-rules": {"decisions.txt": cidr_good()},
}
BAD = {
    "hx-semver-sort": {"sorted.txt": semver_bad()},
    "hx-ini-merge": {"config.json": ini_good(strip_inline_comments=True)},
    "hx-token-bucket": {"limiter.py": BUCKET_BAD},
    "hx-cron-next": {"cronnext.py": CRON_BAD},
    "hx-inventory-diff": {"changes.json": diff_good(price_as_text=True)},
    "hx-timesheet": {"payroll.csv": payroll_good(merge_overlaps=False)},
    "hx-json-flatten": {"flat.py": FLAT_BAD},
    "hx-md-toc": {"toc.md": toc_good(skip_fences=False)},
    "hx-messy-csv": {"totals.csv": messy_good(parens_negative=False)},
    "hx-sql-overdraft": {"overdrafts.csv": overdraft_good(order="t.id")},
    "hx-shortest-route": {"answers.txt": routes_good(tie_on_stops=False)},
    "hx-cidr-rules": {"decisions.txt": cidr_good(ignore_host_bits=False)},
}


def run(sid, files):
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        for name, text in S[sid].get("files", {}).items():
            (folder / name).write_text(text, encoding="utf-8", newline="\n")
        for name, text in files.items():
            (folder / name).write_text(text, encoding="utf-8", newline="\n")
        return evaluate(S[sid], folder, None)


if __name__ == "__main__":
    fails = 0
    for sid in [s for s in S if s.startswith("hx-")]:
        good, bad = run(sid, GOOD[sid]), run(sid, BAD[sid])
        ok = good.task is True and bad.task is False
        fails += not ok
        print(f"{'OK ' if ok else 'BAD'} {sid:18s} good={good.task} ({good.detail[:110]}) | bad={bad.task}")
    print("ALL CHECKS VALID" if not fails else f"{fails} check(s) need fixing")
