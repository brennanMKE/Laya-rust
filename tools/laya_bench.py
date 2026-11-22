#!/usr/bin/env python3
"""Benchmark a laya /predict server with real System 1 Arcade game requests.

Stdlib only. Usage:
  laya_bench.py --url http://yoyo:8001 --label go --golden <laya-go>/testdata/golden --out go.json
"""
import argparse, json, os, statistics, threading, time, urllib.request


def post(url, body, timeout=300):
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return time.perf_counter() - t, data


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summ(lat):
    ms = [x * 1000 for x in lat]
    return {"n": len(ms), "mean": statistics.mean(ms), "p50": pct(ms, 50), "p90": pct(ms, 90), "p95": pct(ms, 95),
            "p99": pct(ms, 99), "min": min(ms), "max": max(ms), "raw": ms}


def load_golden(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "games.jsonl"))]
    edge = [json.loads(l) for l in open(os.path.join(d, "edge.jsonl"))]
    return recs, edge


def parity(answers, expected):
    """Max abs probability difference vs. the PyTorch reference, and choice agreement."""
    maxd, agree, total = 0.0, 0, 0
    for key, qs in expected.items():
        for qn, exp in qs.items():
            got = answers.get(f"{key}.{qn}") if key else answers.get(qn)
            if got is None:
                continue
            total += 1
            if exp.get("type") == "choice":
                agree += got.get("choice") == exp.get("choice")
                for o, p in exp["probabilities"].items():
                    maxd = max(maxd, abs(got["probabilities"].get(o, 0) - p))
            else:
                for f in ("score", "noul", "value", "probability"):
                    if f in exp and isinstance(exp[f], (int, float)):
                        maxd = max(maxd, abs(got.get(f, 0) - exp[f]))
                agree += 1
    return maxd, agree, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-game", type=int, default=20, help="sequential requests per game")
    ap.add_argument("--single", type=int, default=30, help="sequential single-state requests")
    ap.add_argument("--conc", default="1,2,4")
    ap.add_argument("--parity-n", type=int, default=60, help="golden requests to check (0 = all)")
    ap.add_argument("--conc-secs", type=float, default=30)
    a = ap.parse_args()
    base = a.url.rstrip("/")
    pred = base + "/predict"
    recs, edge = load_golden(a.golden)
    by = {}
    for r in recs:
        by.setdefault(r["source"], []).append(r)
    res = {"label": a.label, "url": base, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    res["health"] = get(base + "/health")
    res["stats_before"] = get(base + "/stats")

    # Cold-ish first request, then warm up.
    first = json.dumps(by["frogger"][0]["request"]).encode()
    res["first_request_ms"] = post(pred, first)[0] * 1000
    for r in by["tetris"][:3] + by["frogger"][:3]:
        post(pred, json.dumps(r["request"]).encode())

    # Parity: every golden game request once.
    maxd, agree, total, plat = 0.0, 0, 0, []
    import random
    prs = random.Random(1).sample(recs, a.parity_n) if a.parity_n else recs
    for r in prs:
        dt, body = post(pred, json.dumps(r["request"]).encode())
        plat.append(dt)
        ans = json.loads(body)["answers"]
        exp = r.get("predict_many") or {"": r.get("predict", {})}
        d, g, t = parity(ans, exp)
        maxd, agree, total = max(maxd, d), agree + g, total + t
    res["parity"] = {"requests": len(prs), "answers": total, "choice_agree": agree,
                     "max_abs_prob_diff": maxd}

    # Sequential latency per workload.
    seq = {}
    singles = [e["request"] for e in edge if "batch" not in e["request"]]
    lat = []
    for i in range(a.single):
        lat.append(post(pred, json.dumps(singles[i % len(singles)]).encode())[0])
    seq["single"] = summ(lat) | {"states": 1}
    for g in ("frogger", "invaders", "tetris"):
        rs = by[g]
        lat, states = [], []
        for i in range(a.per_game):
            r = rs[i % len(rs)]["request"]
            states.append(len(r.get("batch", {})) or 1)
            lat.append(post(pred, json.dumps(r).encode())[0])
        seq[g] = summ(lat) | {"states": statistics.mean(states)}
        print(f"[{a.label}] {g}: p50 {seq[g]['p50']:.1f} ms", flush=True)
    res["sequential"] = seq

    # Throughput under concurrency: mixed game requests for conc-secs each.
    mix = [json.dumps(r["request"]).encode() for r in recs]
    conc = {}
    for c in [int(x) for x in a.conc.split(",")]:
        lat, errs, states = [], [0], [0]
        lock = threading.Lock()
        stop = time.perf_counter() + a.conc_secs

        def worker(wid):
            i = wid * 37
            while time.perf_counter() < stop:
                body = mix[i % len(mix)]
                i += 1
                try:
                    dt, _ = post(pred, body)
                    with lock:
                        lat.append(dt)
                        states[0] += len(recs[(i - 1) % len(recs)]["request"].get("batch", {})) or 1
                except Exception:
                    with lock:
                        errs[0] += 1

        t0 = time.perf_counter()
        ths = [threading.Thread(target=worker, args=(w,)) for w in range(c)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        el = time.perf_counter() - t0
        conc[str(c)] = summ(lat) | {"rps": len(lat) / el, "states_per_s": states[0] / el,
                                    "errors": errs[0], "elapsed": el}
        print(f"[{a.label}] c={c}: {conc[str(c)]['rps']:.2f} req/s p50 {conc[str(c)]['p50']:.0f} ms", flush=True)
    res["concurrency"] = conc
    res["stats_after"] = get(base + "/stats")
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res["parity"]))


if __name__ == "__main__":
    main()
