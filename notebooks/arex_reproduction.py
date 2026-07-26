# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.23.0",
#   "matplotlib>=3.9.0",
#   "numpy>=2.0.0",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # AREX mechanism reproduction

        Long research traces can bury earlier evidence, and a single pass can stop
        before every clue is checked. AREX proposes letting an agent rewrite its
        working context and adding an outer, constraint-by-constraint audit. This
        notebook walks through a fresh public test of those two ideas.

        **Verdict: partially reproduced.** Context updating produced a clear
        **+5.00 percentage-point** strict exact-match gain; adding the audit
        produced a smaller **+2.55-point** gain whose exact binary test was
        borderline. The notebook embeds terminal aggregates, so viewing it never
        reruns the expensive Kubernetes experiments.
        """
    )
    return (mo,)


@app.cell
def _():
    cells = ["Fixed context", "Context update", "Audit only", "Update + audit"]
    colors = ["#9CA3AF", "#2563EB", "#F59E0B", "#0F766E"]
    primary = {
        "strict exact match": [25.94, 30.94, 29.34, 33.49],
        "gold-evidence recall": [20.37, 26.38, 24.36, 29.94],
        "guarded answer": [27.26, 32.55, 30.85, 35.28],
    }
    exact_ci = [(22.7, 29.2), (27.6, 34.3), (25.8, 32.9), (30.0, 37.1)]
    effects = {
        "Context update − fixed": {
            "estimate": 5.00,
            "ci": (2.36, 7.74),
            "wins": 129,
            "losses": 76,
            "p": 0.00026197,
        },
        "Full − context update": {
            "estimate": 2.55,
            "ci": (0.09, 5.00),
            "wins": 107,
            "losses": 80,
            "p": 0.05697,
        },
    }
    return cells, colors, effects, exact_ci, primary


@app.cell
def _(cells, colors, exact_ci, mo, primary):
    import matplotlib.pyplot as _plt
    import numpy as np

    values = primary["strict exact match"]
    low = [v - ci[0] for v, ci in zip(values, exact_ci)]
    high = [ci[1] - v for v, ci in zip(values, exact_ci)]
    _fig, _ax = _plt.subplots(figsize=(8, 4.5))
    _bars = _ax.bar(
        range(4),
        values,
        color=colors,
        yerr=np.array([low, high]),
        capsize=4,
    )
    _ax.set_xticks(range(4), cells)
    _ax.set_ylabel("Strict normalized exact match (%)")
    _ax.set_title("Primary result: matched BrowseComp-Plus evaluation")
    _ax.set_ylim(0, 43)
    _ax.grid(axis="y", alpha=0.2)
    for _bar, _value in zip(_bars, values):
        _ax.text(
            _bar.get_x() + _bar.get_width() / 2,
            _value + 1.2,
            f"{_value:.1f}%",
            ha="center",
            fontweight="bold",
        )
    _fig.tight_layout()
    mo.vstack(
        [
            _fig,
            mo.md(
                "Whiskers resample question IDs, keeping both primary seeds "
                "together. The same 1,060 query-seed pairs were evaluated in "
                "every bar."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Matched setup

    The experiment used 530 held-out BrowseComp-Plus questions, two primary
    sampling seeds, Qwen/Qwen3.5-4B, and a frozen 100,195-document BM25
    corpus. Every cell had the same 32 search/open-call cap and 60,000 shared
    completion-token cap. A deterministic checker scored normalized exact
    match and recall of benchmark gold documents in the final citations.

    - **Fixed context:** search, open document, structured finish.
    - **Context update:** adds an agent-invoked rewrite of retained working
      evidence.
    - **Audit only:** adds outer accept/refine/restart rounds.
    - **Update + audit:** combines both interfaces.

    Earlier setup runs were excluded because their shared completion counter
    did not charge every model call. All numbers here come from corrected,
    fresh Kubernetes runs.
    """)
    return


@app.cell
def _(effects, mo):
    rows = []
    for comparison, _value in effects.items():
        rows.append(
            {
                "paired comparison": comparison,
                "strict gain (pp)": _value["estimate"],
                "query-bootstrap 95% CI": (
                    f"{_value['ci'][0]:+.2f} to {_value['ci'][1]:+.2f}"
                ),
                "discordant wins / losses": f"{_value['wins']} / {_value['losses']}",
                "exact McNemar p": _value["p"],
            }
        )
    mo.vstack(
        [
            mo.md("## Claim-by-claim evidence"),
            mo.ui.table(rows, selection=None),
            mo.md(
                """
                Context updating also improved gold-evidence recall by **+6.01
                points** (95% CI +3.78 to +8.26). The full controller improved
                evidence recall by another **+3.55 points** (95% CI +1.64 to
                +5.50), even though its strict-answer discordance test was
                borderline.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo, primary):
    metric = mo.ui.dropdown(
        options=list(primary),
        value="gold-evidence recall",
        label="Diagnostic metric",
    )
    mo.vstack([mo.md("## Evidence and answer diagnostics"), metric])
    return (metric,)


@app.cell
def _(cells, colors, metric, primary):
    import matplotlib.pyplot as _plt

    _selected = metric.value
    _fig, _ax = _plt.subplots(figsize=(8, 3.8))
    _vals = primary[_selected]
    _bars = _ax.bar(range(4), _vals, color=colors)
    _ax.set_xticks(range(4), cells)
    _ax.set_ylabel(f"{_selected} (%)")
    _ax.set_ylim(0, max(_vals) + 8)
    _ax.grid(axis="y", alpha=0.2)
    for _bar, _value in zip(_bars, _vals):
        _ax.text(
            _bar.get_x() + _bar.get_width() / 2,
            _value + 0.8,
            f"{_value:.1f}%",
            ha="center",
        )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Robustness

    A complete independent seed-2 factorial repeated the pattern: context
    updating improved strict exact match by **+6.04 points** (95% CI +2.45 to
    +9.62; p=0.0014), while full versus update-only improved by **+1.89
    points** with an interval crossing zero. Partial seed 3 again favored
    context updating by **+3.40 points** and evidence retention by **+4.71
    points**. Its full cell is excluded because two Kubernetes attempts were
    externally cancelled during active partial execution and emitted no terminal
    aggregate.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation and limits

    The intended interfaces changed behavior: context updating averaged 1.02
    rewrites per task and reduced dropped-history chunks from 14.68 to 4.83;
    audited cells used about two research rounds instead of one. Audit-only
    also beat fixed context by +3.40 strict points, while the full system beat
    audit-only by +4.15, a coherent factorial pattern.

    The paper reported much larger gains (+11.8 and +11.1 points). Absolute
    rates are not directly comparable because this public reproduction uses a
    base 4B checkpoint, reconstructed prompts, 24K context, offline lexical
    retrieval, and BrowseComp-Plus rather than AREX's proprietary training,
    checkpoint, and live-search environment.

    Every evidence-bearing run used OpenResearch **Kubernetes** on **NVIDIA
    RTX PRO 6000 Blackwell** GPUs, with **16 GPUs** allocated at peak. Actual
    fresh-campaign elapsed wall time: **5.29 hours**.
    """)
    return


if __name__ == "__main__":
    app.run()
