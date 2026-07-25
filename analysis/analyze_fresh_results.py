#!/usr/bin/env python3
"""Analyze fresh ORX terminal logs without redistributing benchmark plaintext.

Usage:
  python analysis/analyze_fresh_results.py \
    --logs-dir /tmp/arex-fresh-logs \
    --output-json results/clean_rerun_summary.json \
    --images-dir reports/arex-inference-reproduction/images
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CELLS = ["base", "acu", "outer", "full"]
LABELS = {
    "base": "Fixed context",
    "acu": "+ Context update",
    "outer": "+ Audit only",
    "full": "Update + audit",
}
COLORS = {
    "base": "#9CA3AF",
    "acu": "#2563EB",
    "outer": "#F59E0B",
    "full": "#0F766E",
}
PAPER_ACCURACY = {"base": 0.596, "acu": 0.714, "outer": 0.698, "full": 0.825}


def read_log(path: Path):
    records, aggregate = [], None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("RESULT "):
            records.append(json.loads(line[len("RESULT ") :]))
        elif line.startswith("AGGREGATE "):
            aggregate = json.loads(line[len("AGGREGATE ") :])
    if not records or aggregate is None:
        raise RuntimeError(f"{path} lacks complete RESULT/AGGREGATE evidence")
    return records, aggregate


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def cluster_bootstrap(a, b, field: str, n_boot=20000, seed=260721461):
    """Paired effect b-a, resampling query ids so both seeds stay together."""
    qids = sorted({int(r["qid"]) for r in a})
    ma = {(int(r["qid"]), int(r["seed"])): float(r[field]) for r in a}
    mb = {(int(r["qid"]), int(r["seed"])): float(r[field]) for r in b}
    per_q = np.array(
        [
            np.mean(
                [
                    mb[(q, s)] - ma[(q, s)]
                    for s in sorted({int(r["seed"]) for r in a if int(r["qid"]) == q})
                ]
            )
            for q in qids
        ]
    )
    rng = np.random.default_rng(seed)
    chunks = []
    for _ in range(math.ceil(n_boot / 2000)):
        idx = rng.integers(0, len(per_q), size=(min(2000, n_boot - len(chunks) * 2000), len(per_q)))
        chunks.append(per_q[idx].mean(axis=1))
    boot = np.concatenate(chunks)
    return float(per_q.mean()), [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def cluster_rate_ci(records, field: str, n_boot=20000, seed=21461):
    qids = sorted({int(r["qid"]) for r in records})
    per_q = np.array(
        [
            np.mean([float(r[field]) for r in records if int(r["qid"]) == q])
            for q in qids
        ]
    )
    rng = np.random.default_rng(seed)
    boot = []
    remaining = n_boot
    while remaining:
        n = min(2000, remaining)
        idx = rng.integers(0, len(per_q), size=(n, len(per_q)))
        boot.append(per_q[idx].mean(axis=1))
        remaining -= n
    vals = np.concatenate(boot)
    return [float(x) for x in np.quantile(vals, [0.025, 0.975])]


def paired_test(a, b, field):
    ka = {(int(r["qid"]), int(r["seed"])): r for r in a}
    kb = {(int(r["qid"]), int(r["seed"])): r for r in b}
    if set(ka) != set(kb):
        raise RuntimeError("Cells are not exactly paired by (query, seed)")
    keys = sorted(ka)
    out = {
        "n_pairs": len(keys),
        "effect": float(np.mean([float(kb[k][field]) - float(ka[k][field]) for k in keys])),
    }
    effect, ci = cluster_bootstrap(a, b, field)
    out["effect"] = effect
    out["cluster_bootstrap_95ci"] = ci
    if field == "correct":
        wins = sum(not bool(ka[k][field]) and bool(kb[k][field]) for k in keys)
        losses = sum(bool(ka[k][field]) and not bool(kb[k][field]) for k in keys)
        out.update({"discordant_wins": wins, "discordant_losses": losses, "mcnemar_exact_p": exact_mcnemar(wins, losses)})
    return out


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.dpi": 160,
        }
    )


def save_accuracy(summary, images_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    vals = [summary["cells"][c]["aggregate"]["acc_em"] * 100 for c in CELLS]
    cis = [summary["cells"][c]["exact_match_cluster_bootstrap_95ci"] for c in CELLS]
    err = np.array([[v - ci[0] * 100 for v, ci in zip(vals, cis)], [ci[1] * 100 - v for v, ci in zip(vals, cis)]])
    bars = ax.bar(range(4), vals, color=[COLORS[c] for c in CELLS], yerr=err, capsize=4)
    ax.set_xticks(range(4), [LABELS[c] for c in CELLS])
    ax.set_ylabel("Strict normalized exact match (%)")
    ax.set_title("Fresh BrowseComp-Plus results")
    ax.set_ylim(0, max(vals) + 10)
    ax.grid(axis="y", alpha=0.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.3, f"{val:.1f}%", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(images_dir / "headline_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def save_effects(summary, images_dir):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    claims = [summary["claims"]["context_update"], summary["claims"]["outer_audit"]]
    labels = ["Context update\nvs fixed", "Full audit\nvs update only"]
    obs = [c["exact_match"]["effect"] * 100 for c in claims]
    cis = [c["exact_match"]["cluster_bootstrap_95ci"] for c in claims]
    paper = [11.8, 11.1]
    y = np.arange(2)
    xerr = np.array([[v - ci[0] * 100 for v, ci in zip(obs, cis)], [ci[1] * 100 - v for v, ci in zip(obs, cis)]])
    ax.errorbar(obs, y, xerr=xerr, fmt="o", color="#111827", capsize=5, label="Fresh reproduction")
    ax.scatter(paper, y, marker="x", s=80, color="#DC2626", linewidth=2, label="Paper")
    ax.axvline(0, color="#6B7280", lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired strict exact-match change (percentage points)")
    ax.set_title("Mechanism gains: observed versus reported")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(images_dir / "paired_effects.png", bbox_inches="tight")
    plt.close(fig)


def save_evidence(summary, images_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    vals = [summary["cells"][c]["aggregate"]["ev_recall_gold"] * 100 for c in CELLS]
    bars = ax.bar(range(4), vals, color=[COLORS[c] for c in CELLS])
    ax.set_xticks(range(4), [LABELS[c] for c in CELLS])
    ax.set_ylabel("Gold-document recall in cited evidence (%)")
    ax.set_title("Did the final answer retain the benchmark evidence?")
    ax.set_ylim(0, max(vals) + 8)
    ax.grid(axis="y", alpha=0.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(images_dir / "evidence_retention.png", bbox_inches="tight")
    plt.close(fig)


def save_seed_robustness(summary, images_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(4)
    for seed, marker in [("0", "o"), ("1", "s")]:
        vals = [summary["cells"][c]["aggregate"]["by_seed"][seed]["acc_em"] * 100 for c in CELLS]
        ax.plot(x, vals, marker=marker, lw=2, label=f"Seed {seed}")
    ax.set_xticks(x, [LABELS[c] for c in CELLS])
    ax.set_ylabel("Strict normalized exact match (%)")
    ax.set_title("The scaffold ordering across independent sampling seeds")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(images_dir / "seed_robustness.png", bbox_inches="tight")
    plt.close(fig)


def save_mechanism(summary, images_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(4)
    width = 0.24
    fields = [
        ("mean_updates", "Context updates", "#2563EB"),
        ("mean_truncations", "Dropped-history chunks", "#DC2626"),
        ("mean_rounds", "Research rounds", "#0F766E"),
    ]
    for i, (field, label, color) in enumerate(fields):
        vals = [summary["cells"][c]["aggregate"][field] for c in CELLS]
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
    ax.set_xticks(x, [LABELS[c] for c in CELLS])
    ax.set_ylabel("Mean events per task")
    ax.set_title("The intended mechanisms changed agent behavior")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(images_dir / "mechanism_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    args = ap.parse_args()

    records, aggregates = {}, {}
    for cell in CELLS:
        records[cell], aggregates[cell] = read_log(args.logs_dir / f"{cell}.log")

    keys = [{(int(r["qid"]), int(r["seed"])) for r in records[c]} for c in CELLS]
    if any(k != keys[0] for k in keys[1:]):
        raise RuntimeError("The four cells did not evaluate identical query-seed pairs")

    summary = {
        "paper": {
            "paper_id": "2607.21461",
            "browsecomp_accuracy": PAPER_ACCURACY,
            "reported_context_update_gain_pp": 11.8,
            "reported_outer_audit_gain_over_acu_pp": 11.1,
        },
        "design": {
            "benchmark": "BrowseComp-Plus",
            "query_subset": "530-query held-out complement of a fixed 300-query setup sample",
            "n_queries": len({int(r["qid"]) for r in records["base"]}),
            "seeds": sorted({int(r["seed"]) for r in records["base"]}),
            "n_paired_tasks": len(records["base"]),
            "model": "Qwen/Qwen3.5-4B",
            "corpus_documents": 100195,
            "backend": "Kubernetes",
            "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
            "peak_concurrent_gpu_count": 16,
        },
        "cells": {},
        "claims": {},
    }
    for cell in CELLS:
        summary["cells"][cell] = {
            "aggregate": aggregates[cell],
            "guarded_answer_cluster_bootstrap_95ci": cluster_rate_ci(records[cell], "correct"),
            "exact_match_cluster_bootstrap_95ci": cluster_rate_ci(records[cell], "em"),
            "record_count": len(records[cell]),
        }
    summary["claims"]["context_update"] = {
        "comparison": "acu - base",
        "exact_match": paired_test(records["base"], records["acu"], "em"),
        "guarded_answer": paired_test(records["base"], records["acu"], "correct"),
        "evidence_retention": paired_test(records["base"], records["acu"], "ev_recall_gold"),
    }
    summary["claims"]["outer_audit"] = {
        "comparison": "full - acu",
        "exact_match": paired_test(records["acu"], records["full"], "em"),
        "guarded_answer": paired_test(records["acu"], records["full"], "correct"),
        "evidence_retention": paired_test(records["acu"], records["full"], "ev_recall_gold"),
    }
    summary["factorial_controls"] = {
        "audit_only_vs_base_exact_match": paired_test(records["base"], records["outer"], "em"),
        "full_vs_audit_only_exact_match": paired_test(records["outer"], records["full"], "em"),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.images_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    style()
    save_accuracy(summary, args.images_dir)
    save_effects(summary, args.images_dir)
    save_evidence(summary, args.images_dir)
    save_seed_robustness(summary, args.images_dir)
    save_mechanism(summary, args.images_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
