"""Measure retrieval against Grace's question set: the first answerability check.

A question lists its relevant passages (document, version, section). A search result is relevant when
it comes from one of them (same document and version, and it covers that section).

hit@k: 1 if at least one of the top k results is relevant, else 0. Its mean over the questions is the
share of answerable questions for which the evidence reached the top k: the number to beat.
MRR@k (Mean Reciprocal Rank): mean of 1 / rank of the first relevant result within the top k (0 if none),
so a relevant passage at rank 1 scores 1.0, rank 2 scores 0.5, and rank 5 scores 0.2.
Unanswerable questions have no relevant passage and are left out.
"""

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
QUESTIONS = PROJECT / "questions" / "questions.jsonl"


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    language: str
    kind: str
    as_of: str
    public_ok: bool
    abstain: bool
    reference_answer: str
    keys: list
    must_not: list
    relevant: list            # [{"doc_id", "version", "section", "access"}]
    relevant_chunks: list
    wrong_version: list
    tags: list
    notes: str


def load_questions(path: Path = QUESTIONS) -> dict[str, Question]:
    qs = [Question(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {q.id: q for q in qs}


def covers(result, passage: dict) -> bool:
    return result.doc_id == passage["doc_id"] and result.version == passage["version"] and passage["section"] in result.sections


def retrieval_scores(q: Question, ranked: list, k: int) -> dict:
    if not q.relevant:
        return {"id": q.id, "kind": q.kind, "scored": False}
    first = next((i for i, r in enumerate(ranked) if any(covers(r, p) for p in q.relevant)), None)
    hit = 1.0 if first is not None and first < k else 0.0
    rr = 1.0 / (first + 1) if hit else 0.0
    return {"id": q.id, "kind": q.kind, "scored": True, "hit": hit, "rr": rr,
            "first_rank": None if first is None else first + 1}


def summarise(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["scored"]]
    if not scored:
        return {"questions": 0, "hit": None, "mrr": None}
    return {"questions": len(scored),
            "hit": round(statistics.fmean(r["hit"] for r in scored), 3),
            "mrr": round(statistics.fmean(r["rr"] for r in scored), 3)}


def by_kind(rows: list[dict]) -> dict:
    return {kind: summarise([r for r in rows if r["kind"] == kind]) for kind in sorted({r["kind"] for r in rows})}