import json
from pathlib import Path
from socket import timeout as socket_timeout
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "evaluation" / "dataset"
DATASET_FILES = {
    "hexnode": DATASET_DIR / "hexnode_questions.json",
    "keka": DATASET_DIR / "keka_questions.json",
    "combined": DATASET_DIR / "combined_questions.json",
}
BASE_URL = "http://localhost:8000"
DATASET_TO_RUN = "all"  # hexnode, keka, combined, all
REQUEST_TIMEOUT = 120
CHECK_DATASET_ONLY = False


def load_dataset(path: Path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Dataset file must contain a list of questions")

    return data


def call_ask_api(question: str, source: str):
    payload = json.dumps(
        {
            "question": question,
            "source": source,
        }
    ).encode("utf-8")

    api_request = request.Request(
        f"{BASE_URL.rstrip('/')}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(api_request, timeout=REQUEST_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_text(value):
    if value is None:
        return ""
    return str(value).lower()


def collect_search_text(result):
    parts = []

    answer = result.get("answer", "")
    if answer:
        parts.append(str(answer))

    for chunk in result.get("chunks", []):
        title = chunk.get("title", "")
        content = chunk.get("content", "")
        if title:
            parts.append(str(title))
        if content:
            parts.append(str(content))

    for source_name, source_result in (result.get("source_results") or {}).items():
        parts.append(source_name)
        for chunk in source_result.get("chunks", []):
            title = chunk.get("title", "")
            content = chunk.get("content", "")
            if title:
                parts.append(str(title))
            if content:
                parts.append(str(content))

    return "\n".join(parts).lower()


def score_result(item, result):
    search_text = collect_search_text(result)

    expected_keywords = item.get("expected_keywords") or []
    matched_keywords = [keyword for keyword in expected_keywords if keyword.lower() in search_text]
    required_keywords = item.get("required_keywords") or []
    matched_required_keywords = [keyword for keyword in required_keywords if keyword.lower() in search_text]

    answer_text = normalize_text(result.get("answer", "")).strip()
    has_real_answer = bool(answer_text)
    fallback_answer = (
        "i don't know" in answer_text
        or "information not found" in answer_text
        or "not available" in answer_text
    )
    chunk_count = len(result.get("chunks", []))
    retrieval_hit = chunk_count > 0
    answer_mentions_keyword = bool(matched_keywords) or bool(matched_required_keywords)

    if has_real_answer and not fallback_answer and retrieval_hit and answer_mentions_keyword:
        answer_quality = "good"
    elif has_real_answer and not fallback_answer:
        answer_quality = "partial"
    else:
        answer_quality = "weak"

    passed = retrieval_hit and has_real_answer and not fallback_answer and bool(matched_required_keywords or matched_keywords)

    return {
        "expected_keywords": expected_keywords,
        "matched_keywords": matched_keywords,
        "required_keywords": required_keywords,
        "matched_required_keywords": matched_required_keywords,
        "keyword_match_count": len(matched_keywords),
        "keyword_match_ratio": round(len(matched_keywords) / len(expected_keywords), 2) if expected_keywords else 0.0,
        "retrieval_hit": retrieval_hit,
        "chunk_count": chunk_count,
        "fallback_answer": fallback_answer,
        "answer_quality": answer_quality,
        "passed": passed,
    }


def score_error_result(item, status, message):
    return {
        "expected_keywords": item.get("expected_keywords") or [],
        "matched_keywords": [],
        "required_keywords": item.get("required_keywords") or [],
        "matched_required_keywords": [],
        "keyword_match_count": 0,
        "keyword_match_ratio": 0.0,
        "retrieval_hit": False,
        "chunk_count": 0,
        "fallback_answer": True,
        "answer_quality": "weak",
        "passed": False,
        "status": status,
        "error_message": message,
    }


def summarize_results(rows):
    total = len(rows)
    keyword_ratios = [row["scores"]["keyword_match_ratio"] for row in rows]
    good_answers = sum(1 for row in rows if row["scores"]["answer_quality"] == "good")
    partial_answers = sum(1 for row in rows if row["scores"]["answer_quality"] == "partial")
    weak_answers = sum(1 for row in rows if row["scores"]["answer_quality"] == "weak")
    passed = sum(1 for row in rows if row["scores"]["passed"])
    retrieval_hits = sum(1 for row in rows if row["scores"]["retrieval_hit"])
    fallback_answers = sum(1 for row in rows if row["scores"]["fallback_answer"])
    timeouts = sum(1 for row in rows if row["scores"].get("status") == "timeout")
    api_errors = sum(1 for row in rows if row["scores"].get("status") == "api_error")

    return {
        "total_questions": total,
        "pass_rate": round(passed / total, 2) if total else 0.0,
        "retrieval_hit_rate": round(retrieval_hits / total, 2) if total else 0.0,
        "average_keyword_match_ratio": round(sum(keyword_ratios) / total, 2) if total else 0.0,
        "fallback_answer_rate": round(fallback_answers / total, 2) if total else 0.0,
        "timeouts": timeouts,
        "api_errors": api_errors,
        "good_answers": good_answers,
        "partial_answers": partial_answers,
        "weak_answers": weak_answers,
    }


def print_one_result(item, result, scores):
    print("")
    print(f"Case: {item.get('id', '-')}")
    print(f"Question: {item['question']}")
    print(f"Pass: {scores['passed']}")

    status = scores.get("status", "ok")
    if status != "ok":
        print(f"Status: {status}")
        print(f"Error: {scores.get('error_message', '')}")
        return

    tool_result = result.get("tool_result") or {}
    print(f"Tool: {tool_result.get('tool_name')}")
    print(f"Retrieval: {scores['retrieval_hit']} ({scores['chunk_count']} chunks)")
    print(f"Fallback answer: {scores['fallback_answer']}")
    print(
        f"Required keywords matched: "
        f"{len(scores['matched_required_keywords'])}/{len(scores['required_keywords'])}"
    )
    print(f"Answer quality: {scores['answer_quality']}")

    chunks = result.get("chunks", [])
    if chunks:
        first_chunk = chunks[0]
        title = first_chunk.get("title", "Untitled")
        content = " ".join(str(first_chunk.get("content", "")).split())[:180]
        print(f"Top chunk: {title}")
        if content:
            print(f"Chunk preview: {content}")

    answer = " ".join(str(result.get("answer", "")).split())[:260]
    print(f"Answer: {answer}")


def print_summary(summary, dataset_name, dataset_path):
    print("")
    print(f"Dataset: {dataset_name}")
    print(f"File: {dataset_path}")
    print(f"Questions: {summary['total_questions']}")
    print(f"Pass rate: {summary['pass_rate']}")
    print(f"Retrieval hit rate: {summary['retrieval_hit_rate']}")
    print(f"Average keyword match ratio: {summary['average_keyword_match_ratio']}")
    print(f"Fallback answer rate: {summary['fallback_answer_rate']}")
    print(f"Timeouts: {summary['timeouts']}")
    print(f"API errors: {summary['api_errors']}")
    print(
        "Answer quality counts: "
        f"good={summary['good_answers']}, "
        f"partial={summary['partial_answers']}, "
        f"weak={summary['weak_answers']}"
    )


def print_failed_cases(rows):
    failed_rows = [row for row in rows if not row["scores"]["passed"]]

    if not failed_rows:
        return

    print("")
    print("Failed cases:")
    for row in failed_rows:
        item = row["item"]
        scores = row["scores"]
        status = scores.get("status", "ok")
        reason = scores.get("error_message")

        if not reason:
            if not scores["retrieval_hit"]:
                reason = "no chunks retrieved"
            elif scores["fallback_answer"]:
                reason = "fallback answer"
            elif not scores["matched_required_keywords"]:
                reason = "required keywords missing"
            else:
                reason = "quality check failed"

        print(f"- {item.get('id', '-')} [{status}]: {reason}")


def validate_datasets():
    for name, path in DATASET_FILES.items():
        dataset = load_dataset(path)
        print(f"{name}: OK ({len(dataset)} questions) -> {path}")


def main():
    if CHECK_DATASET_ONLY:
        validate_datasets()
        return

    selected_names = list(DATASET_FILES.keys()) if DATASET_TO_RUN == "all" else [DATASET_TO_RUN]

    for dataset_name in selected_names:
        dataset_path = DATASET_FILES[dataset_name]
        dataset = load_dataset(dataset_path)
        rows = []

        print("")
        print("=" * 72)
        print(f"Running {dataset_name} evaluation")
        print("=" * 72)

        for index, item in enumerate(dataset, start=1):
            question = item["question"]
            source = item.get("source", "default")
            print("")
            print(f"[{index}/{len(dataset)}] Asking source={source}")

            try:
                result = call_ask_api(question, source)
                scores = score_result(item, result)
            except socket_timeout:
                result = {}
                scores = score_error_result(item, "timeout", f"request timed out after {REQUEST_TIMEOUT}s")
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                result = {}
                scores = score_error_result(item, "api_error", f"HTTP {exc.code}: {body}")
            except error.URLError as exc:
                result = {}
                scores = score_error_result(
                    item,
                    "api_error",
                    f"could not reach backend at {BASE_URL}: {exc}",
                )

            rows.append({"item": item, "scores": scores})
            print_one_result(item, result, scores)

        summary = summarize_results(rows)
        print_summary(summary, dataset_name, dataset_path)
        print_failed_cases(rows)


if __name__ == "__main__":
    main()
