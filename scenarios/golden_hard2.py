"""Golden solutions for the hard2 scenarios. Each good solution is written
independently of the reference in scenarios/hard2.py (different algorithm
where possible), so agreement cross-checks the reference against the spec
text, not just the checker against itself. Each bad solution is a plausible
mistake the check must reject. Needs Docker. Run:

    python -m scenarios.golden_hard2
"""
import calendar, csv, heapq, io, json, os, re, tempfile
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from scenarios import hard2 as H2
from scenarios.outcomes import evaluate
from scenarios.suite import by_id

S = by_id()

KV_GOOD = r'''import json, os, re, sys
STORE = "store.json"
def load():
    return json.load(open(STORE)) if os.path.exists(STORE) else {}
def save(d):
    json.dump(d, open(STORE, "w"))
def die(msg, code):
    print(msg, file=sys.stderr); sys.exit(code)
def key_ok(k):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", k): die(f"error: invalid key: {k}", 2)
a = sys.argv[1:]
USAGE = "usage: kv.py set KEY VALUE | get KEY | delete KEY | list [--json]"
if a[:1] == ["set"] and len(a) == 3:
    key_ok(a[1]); d = load(); d[a[1]] = a[2]; save(d)
elif a[:1] in (["get"], ["delete"]) and len(a) == 2:
    key_ok(a[1]); d = load()
    if a[1] not in d: die(f"error: no such key: {a[1]}", 1)
    if a[0] == "get": print(d[a[1]])
    else: del d[a[1]]; save(d)
elif a == ["list"]:
    for k, v in sorted(load().items()): print(f"{k}={v}")
elif a == ["list", "--json"]:
    print(json.dumps(load(), sort_keys=True))
else:
    die(USAGE, 2)
'''
CART_GOOD = '''from decimal import Decimal, ROUND_HALF_UP
def add_item(cart, sku, qty, price_cents):
    if not isinstance(qty, int) or qty <= 0: raise ValueError(qty)
    if sku in cart: cart[sku]["qty"] += qty
    else: cart[sku] = {"qty": qty, "price_cents": price_cents}
def remove_item(cart, sku, qty=None):
    if sku not in cart: raise ValueError(sku)
    if qty is None: del cart[sku]; return
    if qty > cart[sku]["qty"]: raise ValueError(qty)
    cart[sku]["qty"] -= qty
    if cart[sku]["qty"] == 0: del cart[sku]
def total(cart, code=None):
    sub = sum(i["qty"] * i["price_cents"] for i in cart.values())
    n = sum(i["qty"] for i in cart.values())
    if code is None: return sub
    if code == "SAVE10": pct = Decimal("0.90")
    elif code == "BULK": pct = Decimal("0.95") if n >= 10 else Decimal(1)
    else: raise ValueError(code)
    return int((sub * pct).quantize(Decimal(1), ROUND_HALF_UP))
'''


def log_good(text):
    out = defaultdict(int)
    a = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ (\w+) ")
    b = re.compile(r"^\[(\d\d/\w{3}/\d{4}:\d\d:\d\d:\d\d) ([+-]\d{4})\] level=(\w+) ")
    for line in text.splitlines():
        if m := a.match(line):
            t, lvl = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S"), m[2]
        elif m := b.match(line):
            local = datetime.strptime(m[1], "%d/%b/%Y:%H:%M:%S")
            sign = 1 if m[2][0] == "+" else -1
            t = local - sign * timedelta(hours=int(m[2][1:3]), minutes=int(m[2][3:]))
            lvl = m[3]
        else:
            continue
        if lvl.upper() == "ERROR":
            out[t.strftime("%Y-%m-%dT%H")] += 1
    return json.dumps(dict(sorted(out.items())))


def topo_good(text):
    deps, users = defaultdict(set), defaultdict(set)
    for line in filter(str.strip, text.splitlines()):
        t, ds = line.split(":")
        deps[t.strip()] |= set(ds.split())
        for d in ds.split():
            deps[d]; users[d].add(t.strip())
    indeg = {t: len(ds) for t, ds in deps.items()}
    heap = [t for t, n in indeg.items() if n == 0]; heapq.heapify(heap); order = []
    while heap:
        t = heapq.heappop(heap); order.append(t)
        for u in users[t]:
            indeg[u] -= 1
            if indeg[u] == 0: heapq.heappush(heap, u)
    return "cycle detected\n" if len(order) < len(deps) else "\n".join(order) + "\n"


def bills_good(text):
    bal = defaultdict(int)
    for row in csv.DictReader(io.StringIO(text)):
        cents = int(Decimal(row["amount"]) * 100)
        people = sorted(row["split_among"].split(";"))
        base = cents // len(people); extra = cents - base * len(people)
        bal[row["payer"]] += cents
        for p in people:
            bal[p] -= base + (1 if extra > 0 else 0); extra -= 1
    lines, work = [], dict(bal)
    while any(work.values()):
        d = min((v, k) for k, v in work.items() if v < 0)[1]
        c = max((v, [-ord(ch) for ch in k], k) for k, v in work.items() if v > 0)[2]
        amt = min(-work[d], work[c]); work[d] += amt; work[c] -= amt
        lines.append(f"{d} pays {c} {amt}")
    return ("name,net_cents\n" + "".join(f"{k},{v}\n" for k, v in sorted(bal.items())), "\n".join(lines) + "\n")


def fetch_good(folder):
    import importlib.util, sys
    cwd = os.getcwd(); os.chdir(folder)
    try:
        spec = importlib.util.spec_from_file_location("fakeapi_golden", folder / "fakeapi.py")
        api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
        items, cursor = {}, None
        while True:
            try: page = api.fetch_page(cursor)
            except api.RateLimited: continue
            for it in page["items"]: items[it["id"]] = it
            cursor = page["next"]
            if cursor is None: break
        return json.dumps([items[k] for k in sorted(items)])
    finally:
        os.chdir(cwd)


MIGRATE_GOOD = """
CREATE TABLE people_new (id INTEGER PRIMARY KEY, first_name TEXT NOT NULL, last_name TEXT NOT NULL, phone TEXT);
WITH n AS (SELECT id, phone, trim(replace(replace(replace(replace(full_name, '  ', ' '), '  ', ' '), '  ', ' '), '  ', ' ')) AS full FROM people),
     d AS (SELECT id, full, replace(replace(replace(replace(replace(replace(phone, '(', ''), ')', ''), '-', ''), '.', ''), ' ', ''), '+', '') AS digits FROM n)
INSERT INTO people_new
SELECT id,
       CASE WHEN instr(full, ' ') > 0 THEN substr(full, 1, instr(full, ' ') - 1) ELSE full END,
       CASE WHEN instr(full, ' ') > 0 THEN substr(full, instr(full, ' ') + 1) ELSE '' END,
       CASE WHEN digits IS NULL THEN NULL WHEN length(digits) = 11 AND substr(digits, 1, 1) = '1' THEN substr(digits, 2) ELSE digits END
FROM d;
DROP TABLE people;
ALTER TABLE people_new RENAME TO people;
"""


def revenue_good():
    q = lambda t: list(csv.DictReader(io.StringIO(t)))
    reg = {c["id"]: c["region"] for c in q(H2.CUSTOMERS)}
    rate = {(x["month"], x["currency"]): Decimal(x["usd_rate"]) for x in q(H2.RATES)}
    orders = {o["id"]: o for o in q(H2.ORDERS)}
    net = defaultdict(Decimal)
    for o in orders.values():
        net[(o["date"][:7], reg[o["customer_id"]])] += Decimal(o["amount"]) * rate[(o["date"][:7], o["currency"])]
    for f in q(H2.REFUNDS):
        o = orders[f["order_id"]]
        net[(f["date"][:7], reg[o["customer_id"]])] -= Decimal(f["amount"]) * rate[(f["date"][:7], o["currency"])]
    rows = [(m, r, v.quantize(Decimal("0.01"), ROUND_HALF_UP)) for (m, r), v in sorted(net.items())]
    return "month,region,net_usd\n" + "".join(f"{m},{r},{v}\n" for m, r, v in rows if v != 0)


def recur_good():
    days = set()
    for m in (10, 11, 12):
        cal = calendar.monthcalendar(2026, m)
        tues = [w[calendar.TUESDAY] for w in cal if w[calendar.TUESDAY]]
        fris = [w[calendar.FRIDAY] for w in cal if w[calendar.FRIDAY]]
        days |= {date(2026, m, tues[1]), date(2026, m, fris[-1])}
    d = date(2026, 10, 3)
    while d <= date(2026, 12, 31):
        x = d
        while x.weekday() >= 5: x += timedelta(days=1)
        if x <= date(2026, 12, 31): days.add(x)
        d += timedelta(days=10)
    return "\n".join(sorted(x.isoformat() for x in days)) + "\n"


WRAP_GOOD = '''def wrap(text, width):
    if width < 1: raise ValueError(width)
    blocks, cur = [], []
    for raw in text.splitlines():
        if raw.strip(): cur.extend(raw.split())
        elif cur: blocks.append(cur); cur = []
    if cur: blocks.append(cur)
    lines = []
    for b, words in enumerate(blocks):
        if b: lines.append("")
        cur = None
        for w in words:
            pieces = [w[i:i + width] for i in range(0, len(w), width)] if len(w) > width else [w]
            for i, p in enumerate(pieces):
                if cur is None: cur = p
                elif i == 0 and len(cur) + 1 + len(p) <= width: cur += " " + p
                else: lines.append(cur); cur = p
        if cur is not None: lines.append(cur)
    return lines
'''

GOOD = {
    "hh-kv-cli": {"kv.py": KV_GOOD},
    "hh-cart-spec": {"cart.py": CART_GOOD},
    "hh-log-hourly": {"errors_by_hour.json": log_good(H2.LOG_HOURLY)},
    "hh-toposort": {"order_a.txt": topo_good(H2.TASKS_A), "order_b.txt": topo_good(H2.TASKS_B)},
    "hh-bill-split": dict(zip(("balances.csv", "settlements.txt"), bills_good(H2.EXPENSES))),
    "hh-paged-fetch": "FETCH",
    "hh-migration": {"migrate.sql": MIGRATE_GOOD},
    "hh-net-revenue": {"report.csv": revenue_good()},
    "hh-recurrence": {"dates.txt": recur_good()},
    "hh-wrap": {"wrap.py": WRAP_GOOD},
}
BAD = {
    "hh-kv-cli": {"kv.py": KV_GOOD.replace('die(f"error: no such key: {a[1]}", 1)', 'die(f"error: no such key: {a[1]}", 2)')},
    "hh-cart-spec": {},  # untouched buggy cart.py
    "hh-log-hourly": {"errors_by_hour.json": json.dumps({"2026-09-27T00": sum("ERROR" in l.upper() for l in H2.LOG_HOURLY.splitlines())})},
    "hh-toposort": {"order_a.txt": "compile\nfonts\ndocs\nimages\nassets\nbuild\nlint\ntest\ndeploy\n", "order_b.txt": "cycle detected\n"},
    "hh-bill-split": {"balances.csv": "name,net_cents\nana,1\n", "settlements.txt": ""},
    "hh-paged-fetch": {"items.json": json.dumps([i for page in H2.PAGES for i in page])},
    "hh-migration": {"migrate.sql": MIGRATE_GOOD.replace("trim(replace(replace(replace(replace(full_name, '  ', ' '), '  ', ' '), '  ', ' '), '  ', ' '))", "full_name")},
    "hh-net-revenue": {"report.csv": revenue_good().replace("2026-09,US,-99.99\n", "")},
    "hh-recurrence": {"dates.txt": recur_good().replace("2026-10-05\n", "2026-10-03\n")},
    "hh-wrap": {"wrap.py": "import textwrap\ndef wrap(text, width):\n    return textwrap.wrap(text, width)\n"},
}


def run(sid, files):
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        for name, text in S[sid].get("files", {}).items():
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text(text, encoding="utf-8", newline="\n")
        if files == "FETCH":
            files = {"items.json": fetch_good(folder)}
        for name, text in files.items():
            (folder / name).write_text(text, encoding="utf-8", newline="\n")
        return evaluate(S[sid], folder, None)


fails = 0
for sid in [s for s in S if s.startswith("hh-")]:
    good, bad = run(sid, GOOD[sid]), run(sid, BAD[sid])
    ok = good.task is True and bad.task is False
    fails += not ok
    print(f"{'OK ' if ok else 'BAD'} {sid:15s} good={good.task} ({good.detail[:110]}) | bad={bad.task}")
print("ALL CHECKS VALID" if not fails else f"{fails} check(s) need fixing")
