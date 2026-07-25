"""Run one scaffold cell over a query sample against local vLLM servers.

Evidence channel: stdout. Emits one `RESULT {json}` line per (query, seed)
task and a final `AGGREGATE {json}` line.
"""
import argparse
import asyncio
import json
import random
import sys
import time

from checker import contains_match, evidence_recall, exact_match
from scaffold import Scaffold
from tools import SearchEngine


def log(msg):
    print(f"[driver {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tasks(workdir, cfg):
    tasks = [json.loads(l) for l in open(f"{workdir}/tasks.jsonl")]
    tasks.sort(key=lambda t: int(t["query_id"]))
    rng = random.Random(cfg["query_sample_seed"])
    sample = rng.sample(tasks, cfg["n_queries"]) if cfg["n_queries"] < len(tasks) else tasks
    return sample


async def run_one(scaffold, task, seed, sem, cfg, results):
    async with sem:
        try:
            out = await asyncio.wait_for(scaffold.run_task(task, seed),
                                         timeout=cfg["task_timeout_s"])
        except Exception as e:
            out = {"final_answer": "", "final_evidence": [], "final_confidence": 0,
                   "finish_type": f"error:{type(e).__name__}", "rounds": [],
                   "rounds_used": 0, "tool_calls_used": 0, "updates": 0,
                   "truncations": 0, "parse_fails": 0, "forced_finish": "error",
                   "completion_tokens": 0, "ctx_max_tokens": 0, "update_sizes": [],
                   "wall_s": 0, "transcript": [], "error": str(e)[:300]}
        transcript = out.pop("transcript", [])
        rec = {
            "qid": task["query_id"], "seed": seed, "cell": cfg["cell"],
            "gold": task["answer"],
            "em": exact_match(out["final_answer"], task["answer"]),
            "correct": contains_match(out["final_answer"], task["answer"]),
            "ev_recall_gold": evidence_recall(out["final_evidence"], task["gold_docids"]),
            "ev_recall_evd": evidence_recall(out["final_evidence"], task["evidence_docids"]),
            "n_gold_docs": len(task["gold_docids"]),
            **{k: v for k, v in out.items() if k != "rounds"},
            "rounds": [{k: v for k, v in r.items() if k != "evidence_docids"}
                       for r in out.get("rounds", [])],
        }
        print("RESULT " + json.dumps(rec, ensure_ascii=False), flush=True)
        if cfg.get("smoke") and len(results) < 2:
            print("TRANSCRIPT " + json.dumps(
                {"qid": task["query_id"], "seed": seed, "turns": transcript},
                ensure_ascii=False)[:60000], flush=True)
        results.append(rec)
        done, total = len(results), cfg["_total"]
        acc = sum(r["correct"] for r in results) / done
        log(f"{done}/{total} done | running acc={acc:.3f}")


def aggregate(results, cfg):
    def mean(xs):
        xs = list(xs)
        return round(sum(xs) / len(xs), 4) if xs else 0.0
    agg = {
        "cell": cfg["cell"], "n_tasks": len(results),
        "n_queries": cfg["n_queries"], "seeds": cfg["seeds"],
        "acc_contain": mean(r["correct"] for r in results),
        "acc_em": mean(r["em"] for r in results),
        "ev_recall_gold": mean(r["ev_recall_gold"] for r in results),
        "ev_recall_evd": mean(r["ev_recall_evd"] for r in results),
        "mean_tool_calls": mean(r["tool_calls_used"] for r in results),
        "mean_updates": mean(r["updates"] for r in results),
        "mean_truncations": mean(r["truncations"] for r in results),
        "mean_rounds": mean(r["rounds_used"] for r in results),
        "mean_completion_tokens": mean(r["completion_tokens"] for r in results),
        "mean_ctx_max": mean(r["ctx_max_tokens"] for r in results),
        "mean_wall_s": mean(r["wall_s"] for r in results),
        "forced_finish_rate": mean(bool(r["forced_finish"]) for r in results),
        "error_rate": mean("error" in r for r in results),
        "by_seed": {},
    }
    for s in cfg["seeds"]:
        rs = [r for r in results if r["seed"] == s]
        if rs:
            agg["by_seed"][str(s)] = {
                "acc_contain": mean(r["correct"] for r in rs),
                "acc_em": mean(r["em"] for r in rs),
                "ev_recall_gold": mean(r["ev_recall_gold"] for r in rs),
            }
    return agg


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", default="/work")
    ap.add_argument("--ports", required=True, help="space-separated vLLM ports")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    ports = [int(p) for p in args.ports.split()]
    print("CONFIG " + json.dumps(cfg), flush=True)

    log("loading corpus + BM25 index")
    engine = SearchEngine(args.workdir, topk=cfg["search_topk"],
                          snippet_chars=cfg["snippet_chars"],
                          open_doc_chars=cfg["open_doc_chars"])
    log(f"engine ready: {len(engine.docs)} docs")
    tasks = load_tasks(args.workdir, cfg)
    log(f"{len(tasks)} queries x {len(cfg['seeds'])} seeds, cell={cfg['cell']}")

    from openai import AsyncOpenAI
    clients = [AsyncOpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="none",
                           timeout=600, max_retries=3) for p in ports]
    scaffolds = [Scaffold(cfg, engine, c) for c in clients]
    sems = [asyncio.Semaphore(cfg["concurrency_per_server"]) for _ in ports]

    jobs = []
    results = []
    i = 0
    for seed in cfg["seeds"]:
        for t in tasks:
            k = i % len(ports)
            jobs.append(run_one(scaffolds[k], t, seed, sems[k], cfg, results))
            i += 1
    cfg["_total"] = len(jobs)
    t0 = time.time()
    await asyncio.gather(*jobs)
    agg = aggregate(results, cfg)
    agg["total_wall_s"] = round(time.time() - t0, 1)
    print("AGGREGATE " + json.dumps(agg), flush=True)
    log("done")


if __name__ == "__main__":
    sys.path.insert(0, "repro")
    asyncio.run(main())
