"""Deterministic exact-answer checker + evidence-retention metrics."""
import re

ARTICLES = {"a", "an", "the"}


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    toks = [t for t in s.split() if t not in ARTICLES]
    return " ".join(toks)


def exact_match(pred: str, gold: str) -> bool:
    return norm(pred) != "" and norm(pred) == norm(gold)


def contains_match(pred: str, gold: str) -> bool:
    """Gold answer contained in a concise prediction (guards against answer dumping)."""
    p, g = norm(pred), norm(gold)
    if not g or not p:
        return False
    return g in p and len(p) <= max(3 * len(g), len(g) + 80)


def evidence_recall(cited: list, gold_docids: list) -> float:
    if not gold_docids:
        return 0.0
    cited_set = {str(c).strip() for c in cited}
    return sum(1 for d in gold_docids if str(d) in cited_set) / len(gold_docids)
