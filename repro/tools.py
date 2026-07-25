"""Frozen-corpus retrieval tools: BM25 search + document open."""
import json
import os


class SearchEngine:
    def __init__(self, workdir: str, topk: int = 10, snippet_chars: int = 400,
                 open_doc_chars: int = 12000):
        import bm25s
        import Stemmer

        self.retriever = bm25s.BM25.load(os.path.join(workdir, "bm25_index"))
        self.stemmer = Stemmer.Stemmer("english")
        self.bm25s = bm25s
        self.docids = json.load(open(os.path.join(workdir, "docids.json")))
        self.topk = topk
        self.snippet_chars = snippet_chars
        self.open_doc_chars = open_doc_chars
        self.docs = {}
        with open(os.path.join(workdir, "corpus.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                self.docs[d["docid"]] = d

    def search(self, query: str) -> str:
        toks = self.bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer,
                                   show_progress=False)
        ids, scores = self.retriever.retrieve(toks, k=self.topk, show_progress=False)
        lines = []
        for rank in range(ids.shape[1]):
            docid = self.docids[int(ids[0, rank])]
            d = self.docs[docid]
            snippet = d["text"][: self.snippet_chars].replace("\n", " ")
            lines.append(f"[{docid}] {d['title']}\n  {snippet}")
        return "\n".join(lines) if lines else "No results."

    def open(self, docid: str) -> str:
        d = self.docs.get(str(docid).strip())
        if d is None:
            return f"ERROR: no document with docid '{docid}'."
        text = d["text"][: self.open_doc_chars]
        trunc = " [truncated]" if len(d["text"]) > self.open_doc_chars else ""
        return f"[{d['docid']}] {d['title']}\nURL: {d['url']}\n\n{text}{trunc}"

    def excerpt(self, docid: str, chars: int) -> str:
        d = self.docs.get(str(docid).strip())
        if d is None:
            return f"[{docid}] (not found in corpus)"
        return f"[{d['docid']}] {d['title']}\n{d['text'][:chars]}"
