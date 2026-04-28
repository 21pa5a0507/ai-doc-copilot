import json
from pathlib import Path
from socket import timeout as socket_timeout
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = PROJECT_ROOT / "evaluation" / "dataset" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

BASE_URL = "http://localhost:8000"
DATASET_TO_RUN = "all"  # all, hexnode, keka, combined
REQUEST_TIMEOUT = 120
CHECK_DATASET_ONLY = False


def load_cases():
    with DATASET_FILE.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("Evaluation dataset must be a list of cases")

    return cases


def case_group(case):
    source = case.get("source", "default")
    if source == "default":
        return "hexnode"
    if source == "both":
        return "combined"
    return source


def selected_cases(cases):
    if DATASET_TO_RUN == "all":
        return cases
    return [case for case in cases if case_group(case) == DATASET_TO_RUN]


def ask_backend(question, source):
    payload = json.dumps({"question": question, "source": source}).encode("utf-8")
    api_request = request.Request(
        f"{BASE_URL.rstrip('/')}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(api_request, timeout=REQUEST_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def text(value):
    return "" if value is None else str(value).lower()


def all_chunks(result):
    chunks = list(result.get("chunks") or [])

    for source_result in (result.get("source_results") or {}).values():
        chunks.extend(source_result.get("chunks") or [])

    return chunks


def joined_chunks(result, include_content=True):
    parts = []
    for chunk in all_chunks(result):
        parts.append(str(chunk.get("title", "")))
        parts.append(str(chunk.get("source", "")))
        if include_content:
            parts.append(str(chunk.get("content", "")))
    return "\n".join(parts).lower()


def matches(expected_values, haystack):
    return [value for value in expected_values if text(value) in haystack]


def expected_sources_for(case):
    expected = case.get("expected_sources")
    if expected is not None:
        return expected

    source = case.get("source", "default")
    if source == "default":
        return ["hexnode"]
    if source == "both":
        return ["hexnode", "keka"]
    return [source]


def actual_sources(result):
    found = set()
    source = text(result.get("source")).strip()

    if source in {"default", "hexnode"}:
        found.add("hexnode")
    elif source in {"keka", "keka_rag"}:
        found.add("keka")
    elif source == "both":
        found.update({"hexnode", "keka"})

    for source_name in (result.get("source_results") or {}).keys():
        found.add(text(source_name).strip())

    for chunk in all_chunks(result):
        chunk_source = text(chunk.get("kb_source") or chunk.get("source")).strip()
        if "keka" in chunk_source:
            found.add("keka")
        elif chunk_source:
            found.add("hexnode")

    return sorted(source for source in found if source)


def is_fallback(answer):
    answer = text(answer)
    fallback_phrases = (
        "i don't know",
        "information not found",
        "not available",
        "not contain",
        "not supported",
    )
    return any(phrase in answer for phrase in fallback_phrases)


def score_case(case, result):
    answer = text(result.get("answer")).strip()
    chunks = all_chunks(result)
    chunk_text = joined_chunks(result, include_content=True)
    document_text = joined_chunks(result, include_content=False)

    expected_keywords = case.get("expected_keywords") or []
    required_keywords = case.get("required_keywords") or []
    expected_documents = case.get("expected_documents") or []
    expected_sources = expected_sources_for(case)

    matched_documents = matches(expected_documents, document_text)
    matched_chunk_keywords = matches(required_keywords + expected_keywords, chunk_text)
    matched_answer_keywords = matches(required_keywords + expected_keywords, answer)

    source_ok = set(expected_sources).issubset(set(actual_sources(result)))
    document_ok = bool(matched_documents) if expected_documents else True
    retrieval_hit = bool(chunks)
    retrieval_relevant = bool(matched_chunk_keywords)
    answer_covers_facts = bool(matched_answer_keywords)
    fallback = is_fallback(answer)
    hallucination_risk = bool(answer and not fallback and not retrieval_relevant)

    if case.get("out_of_scope"):
        passed = fallback or not answer
    else:
        passed = (
            source_ok
            and document_ok
            and retrieval_hit
            and retrieval_relevant
            and answer_covers_facts
            and not fallback
            and not hallucination_risk
        )

    return {
        "passed": passed,
        "source_ok": source_ok,
        "expected_sources": expected_sources,
        "actual_sources": actual_sources(result),
        "document_ok": document_ok,
        "expected_documents": expected_documents,
        "matched_documents": matched_documents,
        "retrieval_hit": retrieval_hit,
        "retrieval_relevant": retrieval_relevant,
        "chunk_count": len(chunks),
        "answer_covers_facts": answer_covers_facts,
        "matched_answer_keywords": matched_answer_keywords,
        "fallback": fallback,
        "hallucination_risk": hallucination_risk,
    }


def error_score(case, status, message):
    return {
        "passed": False,
        "source_ok": False,
        "expected_sources": expected_sources_for(case),
        "actual_sources": [],
        "document_ok": False,
        "expected_documents": case.get("expected_documents") or [],
        "matched_documents": [],
        "retrieval_hit": False,
        "retrieval_relevant": False,
        "chunk_count": 0,
        "answer_covers_facts": False,
        "matched_answer_keywords": [],
        "fallback": True,
        "hallucination_risk": False,
        "status": status,
        "error_message": message,
    }


def rate(rows, key):
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if row["scores"][key]) / len(rows), 2)


def summarize(rows):
    return {
        "total_questions": len(rows),
        "pass_rate": rate(rows, "passed"),
        "source_accuracy": rate(rows, "source_ok"),
        "document_match_rate": rate(rows, "document_ok"),
        "retrieval_hit_rate": rate(rows, "retrieval_hit"),
        "retrieval_relevance_rate": rate(rows, "retrieval_relevant"),
        "answer_fact_coverage_rate": rate(rows, "answer_covers_facts"),
        "fallback_rate": rate(rows, "fallback"),
        "hallucination_risk_rate": rate(rows, "hallucination_risk"),
    }


def failure_reason(scores):
    if scores.get("error_message"):
        return scores["error_message"]
    if not scores["source_ok"]:
        return "wrong source selected"
    if not scores["document_ok"]:
        return "expected document/title was not retrieved"
    if not scores["retrieval_hit"]:
        return "no chunks retrieved"
    if not scores["retrieval_relevant"]:
        return "retrieved chunks missed expected evidence"
    if scores["fallback"]:
        return "fallback answer"
    if not scores["answer_covers_facts"]:
        return "answer missed expected facts"
    if scores["hallucination_risk"]:
        return "answer was not supported by retrieved evidence"
    return "quality check failed"


def print_case(case, result, scores):
    print("")
    print(f"Case: {case.get('id', '-')}")
    print(f"Question: {case['question']}")
    print(f"Pass: {scores['passed']}")
    print(f"Source OK: {scores['source_ok']} {scores['actual_sources']}")
    print(f"Document OK: {scores['document_ok']} {scores['matched_documents']}")
    print(f"Retrieval: {scores['retrieval_hit']} ({scores['chunk_count']} chunks)")
    print(f"Answer facts OK: {scores['answer_covers_facts']}")

    if not scores["passed"]:
        print(f"Reason: {failure_reason(scores)}")

    chunks = result.get("chunks") or []
    if chunks:
        print(f"Top chunk: {chunks[0].get('title', 'Untitled')}")

    answer = " ".join(str(result.get("answer", "")).split())[:220]
    if answer:
        print(f"Answer: {answer}")


def print_summary(summary):
    print("")
    print(f"Dataset: {DATASET_TO_RUN}")
    print(f"File: {DATASET_FILE}")
    for key, value in summary.items():
        print(f"{key}: {value}")


def save_results(rows, summary):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{DATASET_TO_RUN}_latest.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": DATASET_TO_RUN,
                "dataset_file": str(DATASET_FILE),
                "summary": summary,
                "cases": rows,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved results: {output_path}")


def validate_dataset():
    cases = load_cases()
    counts = {}
    for case in cases:
        counts[case_group(case)] = counts.get(case_group(case), 0) + 1

    print(f"all: OK ({len(cases)} questions) -> {DATASET_FILE}")
    for name in sorted(counts):
        print(f"{name}: {counts[name]} questions")


def run():
    if CHECK_DATASET_ONLY:
        validate_dataset()
        return

    cases = selected_cases(load_cases())
    rows = []

    print("")
    print("=" * 72)
    print(f"Running {DATASET_TO_RUN} evaluation")
    print("=" * 72)

    for index, case in enumerate(cases, start=1):
        print("")
        print(f"[{index}/{len(cases)}] Asking source={case.get('source', 'default')}")

        try:
            result = ask_backend(case["question"], case.get("source", "default"))
            scores = score_case(case, result)
        except socket_timeout:
            result = {}
            scores = error_score(case, "timeout", f"request timed out after {REQUEST_TIMEOUT}s")
        except error.HTTPError as exc:
            result = {}
            body = exc.read().decode("utf-8", errors="replace")
            scores = error_score(case, "api_error", f"HTTP {exc.code}: {body}")
        except error.URLError as exc:
            result = {}
            scores = error_score(case, "api_error", f"could not reach backend at {BASE_URL}: {exc}")

        rows.append({"item": case, "result": result, "scores": scores})
        print_case(case, result, scores)

    summary = summarize(rows)
    print_summary(summary)
    save_results(rows, summary)


if __name__ == "__main__":
    run()
