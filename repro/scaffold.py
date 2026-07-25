"""AREX-style inference scaffolds around a fixed chat model.

Four matched cells (identical model, tools, budgets; only mechanisms differ):
  base : fixed context (drop-oldest truncation), single round
  acu  : + autonomous update_context tool
  outer: + constraint-wise audit -> accept/refine/restart controller
  full : both
"""
import asyncio
import json
import re
import time

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_system_prompt(cfg):
    acu = cfg["acu"]
    tools = [
        'search: {"tool": "search", "args": {"query": "<keywords>"}}\n'
        "   BM25 keyword search over a fixed offline web corpus. Returns top-"
        f'{cfg["search_topk"]} results as [docid] title + snippet. Use focused '
        "keyword queries and try several phrasings.",
        'open: {"tool": "open", "args": {"docid": "<docid>"}}\n'
        "   Returns the full text of a document from search results.",
        'finish: {"tool": "finish", "args": {"answer": "<short exact answer>", '
        '"evidence_docids": ["<docid>", ...], "confidence": <integer 0-100>}}\n'
        "   Submit your final answer, the docids of documents that support it, and a "
        "calibrated confidence (probability your answer is exactly correct).",
    ]
    if acu:
        tools.append(
            'update_context: {"tool": "update_context", "args": {"state": {'
            '"verified_findings": [{"finding": "<fact>", "docids": ["<docid>"]}], '
            '"current_candidates": ["..."], "rejected_candidates": ["..."], '
            '"unresolved_constraints": ["..."], "next_steps": ["..."]}}}\n'
            "   REPLACES your entire interaction history with this compact research "
            "state. Use it after resolving a subproblem, rejecting a hypothesis, or "
            "changing strategy, and whenever your context grows large: it frees space "
            "to continue researching. Anything not saved in the state is lost, so "
            "record verified findings WITH their docids."
        )
    tool_block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(tools))
    return f"""You are a research agent answering a multi-constraint research question using ONLY a fixed offline document corpus reachable through your tools.

Tools:
{tool_block}

Rules:
- Reason briefly, then end every message with exactly one tool call: a single JSON object in a ```json fenced code block.
- The correct answer satisfies ALL constraints in the question. Verify each constraint against document text before finishing; a candidate matching only some constraints is likely wrong.
- You have a budget of {cfg["tool_budget"]} search/open calls in total; remaining budget is shown after each tool result.
- Final answers are short: a name, title, place, date, or number. Give the exact answer only, no explanation inside the answer field."""


def find_tool_call(text: str):
    """Return (tool, args) from the last JSON object containing a 'tool' key."""
    text = THINK_RE.sub("", text or "")
    candidates = []
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        candidates.append(m.group(1))
    # fallback: last balanced {...} starting at '{"tool"' or '{ "tool"'
    for m in re.finditer(r"\{\s*\"tool\"", text):
        start = m.start()
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            return obj.get("tool"), obj.get("args") or {}
    return None, None


def est_tokens(messages) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages) // 3


class Budget:
    def __init__(self, cfg):
        self.tool_calls_left = cfg["tool_budget"]
        self.completion_used = 0
        self.completion_budget = cfg["completion_budget"]

    def exhausted(self):
        return self.tool_calls_left <= 0 or self.completion_used >= self.completion_budget


class Scaffold:
    def __init__(self, cfg, engine, client):
        self.cfg = cfg
        self.engine = engine
        self.client = client
        self.system = build_system_prompt(cfg)
        self.thinking_kwarg_ok = True

    async def chat(self, messages, seed, max_tokens, stats, budget):
        kwargs = dict(
            model=self.cfg["model"],
            messages=messages,
            temperature=self.cfg["temperature"],
            top_p=self.cfg["top_p"],
            max_tokens=max_tokens,
            seed=seed,
        )
        if self.thinking_kwarg_ok:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e)
            if self.thinking_kwarg_ok and ("enable_thinking" in msg or "chat_template" in msg or "template" in msg):
                self.thinking_kwarg_ok = False
                kwargs.pop("extra_body", None)
                resp = await self.client.chat.completions.create(**kwargs)
            else:
                raise
        u = resp.usage
        stats["completion_tokens"] = stats.get("completion_tokens", 0) + (u.completion_tokens or 0)
        budget.completion_used += u.completion_tokens or 0
        stats["ctx_max"] = max(stats.get("ctx_max", 0), u.prompt_tokens or 0)
        return resp.choices[0].message.content or "", u

    # ---------- context management ----------

    def truncate_oldest(self, messages):
        """Drop-oldest fixed-context fallback: blank out earliest big tool results."""
        n = 0
        for m in messages[2:-4]:
            if m["role"] == "user" and len(str(m["content"])) > 600:
                m["content"] = "[earlier tool result dropped: context limit reached]"
                n += 1
                if est_tokens(messages) < self.cfg["ctx_soft_limit_tokens"]:
                    break
        return n

    def apply_update_context(self, messages, state, question, objective):
        head = f"Research question:\n{question}\n"
        if objective:
            head += f"\nCurrent targeted objective:\n{objective}\n"
        head += (
            "\n## Refreshed research state (you wrote this via update_context; "
            "your earlier interaction history was cleared)\n"
            + json.dumps(state, ensure_ascii=False, indent=1)
            + "\n\nContinue the investigation from this state."
        )
        return [messages[0], {"role": "user", "content": head}]

    # ---------- inner research loop ----------

    async def inner_loop(self, question, objective, preserved, seed, budget, stats, transcript):
        cfg = self.cfg
        head = f"Research question:\n{question}\n"
        if objective:
            head += f"\nCurrent targeted objective (from the audit of your previous attempt):\n{objective}\n"
        if preserved:
            head += "\nPreserved findings from previous rounds:\n" + json.dumps(
                preserved, ensure_ascii=False, indent=1
            ) + "\n"
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": head},
        ]
        parse_fails = 0
        demand_pending = 0  # tool calls since an unmet update_context demand
        for step in range(cfg["max_steps_per_round"]):
            # fixed-context fallback (all cells): truncate before overflow
            if est_tokens(messages) > cfg["ctx_soft_limit_tokens"]:
                stats["truncations"] = stats.get("truncations", 0) + self.truncate_oldest(messages)
            if budget.completion_used >= budget.completion_budget:
                return await self.force_finish(messages, seed, stats, transcript, budget, "token_budget")
            try:
                text, u = await self.chat(messages, seed + step, cfg["step_max_tokens"], stats, budget)
            except Exception as e:
                if "maximum context length" in str(e) or "context length" in str(e):
                    stats["truncations"] = stats.get("truncations", 0) + self.truncate_oldest(messages)
                    try:
                        text, u = await self.chat(messages, seed + step, cfg["step_max_tokens"], stats, budget)
                    except Exception:
                        return await self.force_finish(messages, seed, stats, transcript, budget, "ctx_overflow")
                else:
                    raise
            transcript.append({"role": "assistant", "content": text})
            messages.append({"role": "assistant", "content": text})
            tool, args = find_tool_call(text)

            if tool is None:
                parse_fails += 1
                stats["parse_fails"] = stats.get("parse_fails", 0) + 1
                if parse_fails > 4:
                    return await self.force_finish(messages, seed, stats, transcript, budget, "parse_fails")
                reply = ("Could not parse a tool call. End your message with exactly one "
                         "JSON object in a ```json fenced block, e.g. "
                         '{"tool": "search", "args": {"query": "..."}}.')
            elif tool == "finish":
                ans = str(args.get("answer", "")).strip()
                ev = args.get("evidence_docids") or []
                if not isinstance(ev, list):
                    ev = [ev]
                try:
                    conf = float(args.get("confidence", 0))
                except Exception:
                    conf = 0.0
                return {"answer": ans, "evidence_docids": [str(x) for x in ev],
                        "confidence": conf, "finish_type": "model"}
            elif tool == "update_context":
                if not cfg["acu"]:
                    reply = "Tool 'update_context' is not available. Use search, open, or finish."
                else:
                    state = args.get("state") or {k: v for k, v in args.items()}
                    before = est_tokens(messages)
                    messages = self.apply_update_context(messages, state, question, objective)
                    after = est_tokens(messages)
                    demand_pending = 0
                    stats["updates"] = stats.get("updates", 0) + 1
                    stats.setdefault("update_sizes", []).append([before, after])
                    transcript.append({"role": "acu", "content": f"context {before}->{after} est.tokens"})
                    continue
            elif tool in ("search", "open"):
                if budget.tool_calls_left <= 0:
                    reply = ("Search/open budget exhausted. You must now call finish with "
                             "your best answer, evidence docids, and confidence.")
                else:
                    budget.tool_calls_left -= 1
                    loop = asyncio.get_event_loop()
                    if tool == "search":
                        q = str(args.get("query", ""))[:500]
                        out = await loop.run_in_executor(None, self.engine.search, q)
                    else:
                        out = self.engine.open(str(args.get("docid", "")))
                    reply = (f"Tool result ({budget.tool_calls_left} search/open calls "
                             f"remaining):\n{out}")
                    if cfg["acu"]:
                        est = est_tokens(messages) + len(reply) // 3
                        if est > cfg["acu_trigger_tokens"]:
                            if demand_pending == 0:
                                demand_pending = 1
                                reply += (
                                    "\n\n[CONTEXT ALERT: your interaction history is close to "
                                    "the context limit. You MUST call update_context NOW, "
                                    "before any further search/open calls. Save every verified "
                                    "finding WITH its docids, current and rejected candidates, "
                                    "unresolved constraints, and your next steps — anything not "
                                    "saved will be lost.]")
                            else:
                                demand_pending += 1
                                if demand_pending > 3:
                                    stats["acu_refusals"] = stats.get("acu_refusals", 0) + 1
                                    stats["truncations"] = stats.get("truncations", 0) + \
                                        self.truncate_oldest(messages)
                                    demand_pending = 0
            else:
                reply = f"Unknown tool '{tool}'. Available: search, open, finish" + (
                    ", update_context." if cfg["acu"] else ".")

            transcript.append({"role": "tool", "content": reply[:2000]})
            messages.append({"role": "user", "content": reply})
        return await self.force_finish(messages, seed, stats, transcript, budget, "max_steps")

    async def force_finish(self, messages, seed, stats, transcript, budget, reason):
        stats["forced_finish"] = reason
        messages = messages[:1] + messages[1:]  # keep as-is; append final demand
        messages.append({"role": "user", "content":
            "STOP. You must answer NOW. Reply with exactly one finish tool call in a "
            '```json block: {"tool": "finish", "args": {"answer": "...", '
            '"evidence_docids": ["..."], "confidence": <0-100>}}'})
        if est_tokens(messages) > self.cfg["ctx_soft_limit_tokens"]:
            self.truncate_oldest(messages)
        for attempt in range(2):
            try:
                text, _ = await self.chat(messages, seed + 900 + attempt, 600, stats, budget)
            except Exception:
                break
            transcript.append({"role": "assistant", "content": text})
            tool, args = find_tool_call(text)
            if tool == "finish":
                ev = args.get("evidence_docids") or []
                if not isinstance(ev, list):
                    ev = [ev]
                try:
                    conf = float(args.get("confidence", 0))
                except Exception:
                    conf = 0.0
                return {"answer": str(args.get("answer", "")).strip(),
                        "evidence_docids": [str(x) for x in ev],
                        "confidence": conf, "finish_type": f"forced:{reason}"}
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Reply with ONLY the finish tool call JSON."})
        return {"answer": "", "evidence_docids": [], "confidence": 0.0,
                "finish_type": f"forced_empty:{reason}"}

    # ---------- outer audit loop ----------

    async def audit(self, question, result, seed, stats, transcript, budget):
        cfg = self.cfg
        ev_lines = []
        for d in result["evidence_docids"][:8]:
            ev_lines.append(self.engine.excerpt(d, cfg["audit_excerpt_chars"]))
        ev_block = "\n\n".join(ev_lines) if ev_lines else "(no evidence cited)"
        prompt = f"""You are auditing a research agent's provisional answer to a multi-constraint question.

Question:
{question}

Provisional answer: {result['answer'] or '(none)'}
Agent's confidence: {result['confidence']}/100

Cited evidence excerpts:
{ev_block}

Do the following:
1. Break the question into its distinct constraints.
2. For EACH constraint, judge strictly from the cited evidence excerpts: SUPPORTED, UNSUPPORTED (no cited evidence addresses it), or CONTRADICTED.
3. Decide: "accept" (every constraint SUPPORTED and the answer is exact), "refine" (the trajectory is recoverable: keep verified findings and state a targeted objective that resolves the unsupported/contradicted constraints), or "restart" (the answer and evidence look misleading; start over).

Reply with exactly one JSON object in a ```json block:
{{"constraints": [{{"text": "<constraint>", "verdict": "SUPPORTED|UNSUPPORTED|CONTRADICTED"}}, ...],
 "decision": "accept|refine|restart",
 "targeted_objective": "<next-round research objective, if refine>",
 "preserved_findings": [{{"finding": "<verified fact worth keeping>", "docids": ["<docid>"]}}, ...]}}"""
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(2):
            try:
                text, _ = await self.chat(messages, seed + 7000 + attempt, 1500, stats, budget)
            except Exception:
                break
            transcript.append({"role": "audit", "content": text})
            m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", THINK_RE.sub("", text), re.DOTALL)
            raw = m.group(1) if m else None
            if raw is None:
                mm = re.search(r"\{.*\}", THINK_RE.sub("", text), re.DOTALL)
                raw = mm.group(0) if mm else None
            if raw:
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict) and "decision" in obj:
                        return obj
                except Exception:
                    pass
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Reply with ONLY the JSON object."})
        return {"constraints": [], "decision": "accept", "targeted_objective": "",
                "preserved_findings": []}

    # ---------- full task ----------

    async def run_task(self, task, seed):
        cfg = self.cfg
        t0 = time.time()
        budget = Budget(cfg)
        stats = {}
        transcript = []
        question = task["query"]
        rounds = []
        objective, preserved = None, []
        max_rounds = cfg["max_rounds"] if cfg["outer"] else 1

        for rnd in range(max_rounds):
            result = await self.inner_loop(question, objective, preserved,
                                           seed * 100000 + rnd * 1000, budget, stats, transcript)
            rec = {"round": rnd, **result}
            if not cfg["outer"]:
                rounds.append(rec)
                break
            audit = await self.audit(question, result,
                                     seed * 100000 + rnd * 1000, stats, transcript, budget)
            verdicts = [str(c.get("verdict", "")).upper() for c in audit.get("constraints", [])]
            n_sup = sum(v == "SUPPORTED" for v in verdicts)
            all_sup = len(verdicts) > 0 and n_sup == len(verdicts)
            rec["audit_decision"] = audit.get("decision")
            rec["n_constraints"] = len(verdicts)
            rec["n_supported"] = n_sup
            rounds.append(rec)
            accepted = (result["confidence"] >= cfg["confidence_tau"] and all_sup
                        and result["answer"] != "")
            rec["accepted"] = accepted
            if accepted or rnd == max_rounds - 1 or budget.exhausted():
                break
            if audit.get("decision") == "restart":
                objective, preserved = None, []
            else:
                objective = str(audit.get("targeted_objective") or "") or None
                pf = audit.get("preserved_findings") or []
                if isinstance(pf, list):
                    preserved = pf[:12]

        # final answer: accepted round, else best (most supported constraints, then confidence)
        final = None
        for r in rounds:
            if r.get("accepted"):
                final = r
                break
        if final is None:
            final = max(rounds, key=lambda r: (r.get("n_supported", 0),
                                               r.get("confidence", 0),
                                               r["answer"] != ""))
        return {
            "final_answer": final["answer"],
            "final_evidence": final["evidence_docids"],
            "final_confidence": final["confidence"],
            "finish_type": final["finish_type"],
            "rounds": rounds,
            "rounds_used": len(rounds),
            "tool_calls_used": cfg["tool_budget"] - budget.tool_calls_left,
            "updates": stats.get("updates", 0),
            "acu_refusals": stats.get("acu_refusals", 0),
            "truncations": stats.get("truncations", 0),
            "parse_fails": stats.get("parse_fails", 0),
            "forced_finish": stats.get("forced_finish"),
            "completion_tokens": stats.get("completion_tokens", 0),
            "ctx_max_tokens": stats.get("ctx_max", 0),
            "update_sizes": stats.get("update_sizes", []),
            "wall_s": round(time.time() - t0, 1),
            "transcript": transcript,
        }
