"""Paired statistics for cell comparisons on shared (query, seed) tasks."""
import math
import random
from collections import defaultdict


def parse_results(log_text):
    import json
    out = []
    for line in log_text.splitlines():
        if line.startswith("RESULT "):
            try:
                out.append(json.loads(line[len("RESULT "):]))
            except json.JSONDecodeError:
                pass
    return out


def by_task(results, metric):
    """dict (qid, seed) -> metric value."""
    return {(r["qid"], r["seed"]): float(r[metric]) for r in results}


def paired_diff(cell_a, cell_b, metric, n_boot=20000, seed=0):
    """Mean(b) - mean(a) on shared tasks, bootstrap CI resampling queries."""
    a, b = by_task(cell_a, metric), by_task(cell_b, metric)
    keys = sorted(set(a) & set(b))
    diffs_by_q = defaultdict(list)
    for k in keys:
        diffs_by_q[k[0]].append(b[k] - a[k])
    qids = sorted(diffs_by_q)
    per_q = [sum(v) / len(v) for v in (diffs_by_q[q] for q in qids)]
    mean_diff = sum(per_q) / len(per_q)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [per_q[rng.randrange(len(per_q))] for _ in per_q]
        boots.append(sum(s) / len(s))
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    p_boot = 2 * min(sum(x <= 0 for x in boots), sum(x >= 0 for x in boots)) / n_boot
    return {"n_tasks": len(keys), "n_queries": len(qids),
            "mean_a": sum(a[k] for k in keys) / len(keys),
            "mean_b": sum(b[k] for k in keys) / len(keys),
            "diff": mean_diff, "ci95": (lo, hi), "p_boot": min(p_boot, 1.0)}


def mcnemar(cell_a, cell_b, metric="correct"):
    """Exact binomial McNemar on discordant (query,seed) pairs."""
    a, b = by_task(cell_a, metric), by_task(cell_b, metric)
    keys = set(a) & set(b)
    n01 = sum(1 for k in keys if a[k] < 0.5 <= b[k])  # b right, a wrong
    n10 = sum(1 for k in keys if b[k] < 0.5 <= a[k])
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return {"n01": n01, "n10": n10, "p": min(p, 1.0)}
