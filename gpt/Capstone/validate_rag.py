from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings

import embed

FALLBACK_ANSWER = "I cannot find that information in the lab manual."
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9]{2,}|\d+[a-z]?", re.IGNORECASE)
TASK_PATTERN = re.compile(r"task\s*(\d+)", re.IGNORECASE)
REFERENCE_PATTERN = re.compile(
    r"\b(?:figure|table)\.?\s*\d+(?:\s*\.\s*\d+)*(?:[a-z](?![a-z]))?(?=[^0-9]|$)",
    re.IGNORECASE,
)
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "using",
    "lab",
    "manual",
    "section",
    "task",
    "what",
    "when",
    "where",
    "which",
    "does",
    "about",
    "according",
    "students",
    "student",
    "should",
    "have",
    "must",
    "will",
    "each",
}
ACTION_HINTS = {
    "build",
    "take",
    "find",
    "make",
    "connect",
    "construct",
    "measure",
    "record",
    "repeat",
    "plot",
    "calculate",
    "compare",
    "simulate",
    "adjust",
    "design",
    "determine",
    "wire",
    "observe",
    "verify",
    "use",
}
MAX_REFERENCE_CASES_PER_CHUNK = 2

QUESTION_TYPE_LIMITS = {
    "objective": 1,
    "prelab": 1,
    "materials": 1,
    "theory": 1,
    "procedure": 2,
    "task": 6,
    "results": 2,
    "analysis": 1,
    "discussion": 2,
    "report": 2,
    "deliverable": 1,
    "checkoff": 1,
    "questions": 2,
    "conclusion": 1,
    "reference": 4,
    "section": 2,
}


@dataclass
class ValidationCase:
    lab_number: int
    lab_name: str
    manual_path: str
    question_type: str
    question: str
    expected_excerpt: str
    section_name: str
    heading: str
    page_num: Optional[int]
    keywords: list[str]
    references: list[str]


@dataclass
class ValidationResult:
    case: ValidationCase
    ok: bool
    status_code: int
    answer: str
    has_citation: bool
    is_fallback: bool
    keyword_hits: int
    keyword_recall: float
    semantic_similarity: Optional[float]
    passed: bool
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the /chat/{lab_number} RAG endpoint with questions built from the real lab manuals."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--manuals-dir", default=str(embed.MANUALS_DIR))
    parser.add_argument("--labs", nargs="*", type=int, default=None, help="Optional list of lab numbers to test.")
    parser.add_argument("--cases-per-lab", type=int, default=18)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--semantic-threshold", type=float, default=0.62)
    parser.add_argument("--keyword-threshold", type=float, default=0.35)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--show-passes", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Generate questions without calling the API.")
    return parser.parse_args()


def _extract_lab_number(lab_name: str) -> Optional[int]:
    match = re.search(r"\bLab\s+(\d+)\b", lab_name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text.lower()):
        token = match.group(0).lower()
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _normalize_answer(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _sentence_excerpt(text: str, max_chars: int = 340) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_excerpt = " ".join(lines[:3]).strip()
    if line_excerpt and len(line_excerpt) >= 40:
        return line_excerpt[:max_chars].rstrip()

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    excerpt_parts: list[str] = []
    total = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 25:
            continue
        projected = total + len(sentence) + (1 if excerpt_parts else 0)
        if excerpt_parts and projected > max_chars:
            break
        excerpt_parts.append(sentence)
        total = projected
        if total >= max_chars or len(excerpt_parts) >= 2:
            break
    if excerpt_parts:
        return " ".join(excerpt_parts)
    return text.strip()[:max_chars].rstrip()


def _build_keywords(section_name: str, heading: str, excerpt: str, references: list[str]) -> list[str]:
    ranked_terms: list[str] = []
    seen: set[str] = set()
    for ref in references:
        if ref.lower() not in seen:
            ranked_terms.append(ref.lower())
            seen.add(ref.lower())

    token_counts = Counter(_tokenize(" ".join([section_name, heading, excerpt])))
    for token, _ in token_counts.most_common(10):
        if len(token) < 4 or token in seen:
            continue
        ranked_terms.append(token)
        seen.add(token)
        if len(ranked_terms) >= 8:
            break
    return ranked_terms


def _normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" :-,;")


def _looks_noisy_label(text: str) -> bool:
    cleaned = _normalize_label(text)
    if not cleaned:
        return True
    words = cleaned.split()
    if len(words) > 16:
        return True
    if cleaned.count("?") or cleaned.count("!") or cleaned.count("|"):
        return True
    if re.search(r"\btask\s*#?\d+.*\btask\s*#?\d+", cleaned, re.IGNORECASE):
        return True
    alpha_chars = sum(char.isalpha() for char in cleaned)
    if alpha_chars and alpha_chars < len(cleaned) * 0.45:
        return True
    if re.search(r"[=<>]{2,}|[●•]", cleaned):
        return True
    return False


def _has_action_language(text: str) -> bool:
    lowered = f" {text.lower()} "
    return any(f" {verb} " in lowered for verb in ACTION_HINTS)


def _clean_reference_label(text: str) -> str:
    cleaned = _normalize_label(text)
    cleaned = re.sub(r"\b(Figure|Table)\.?\s*(\d)", r"\1 \2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\.\s*", ".", cleaned)
    return cleaned


def _task_label_for_question(section_name: str) -> str:
    normalized = _normalize_label(section_name)
    task_match = re.search(r"\bTask\s+(\d+)\b", normalized, re.IGNORECASE)
    if not task_match:
        return normalized

    prefix = f"Task {task_match.group(1)}"
    title_match = re.match(rf"{re.escape(prefix)}(?::\s*(.*))?$", normalized, re.IGNORECASE)
    title = _normalize_label(title_match.group(1) if title_match and title_match.group(1) else "")
    if not title or _looks_noisy_label(title) or len(title.split()) > 10:
        return prefix
    return f"{prefix}: {title}"


def _extract_reference_labels(text: str, references: list[str]) -> list[str]:
    candidates = [_clean_reference_label(ref) for ref in references]
    candidates.extend(_clean_reference_label(match.group(0)) for match in REFERENCE_PATTERN.finditer(text))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)

    filtered: list[str] = []
    parsed: list[tuple[str, str, str]] = []
    for candidate in deduped:
        match = re.match(r"^(Figure|Table)\s+(\d+(?:\.\d+)*)", candidate, re.IGNORECASE)
        if not match:
            continue
        parsed.append((candidate, match.group(1).lower(), match.group(2)))

    for candidate, kind, number in parsed:
        if any(other_kind == kind and other_number.startswith(f"{number}.") for _, other_kind, other_number in parsed):
            continue
        filtered.append(candidate)

    filtered.sort(key=lambda label: (-label.count("."), len(label)))
    return filtered


def _build_reference_question(lab_name: str, reference_label: str) -> str:
    if reference_label.lower().startswith("figure"):
        return f"What does {reference_label} describe in {lab_name}?"
    return f"What information is given in {reference_label} for {lab_name}?"


def _reference_matches_lab(reference_label: str, lab_number: int) -> bool:
    match = re.match(r"^(?:Figure|Table)\s+(\d+)", reference_label, re.IGNORECASE)
    if not match:
        return True
    return int(match.group(1)) == lab_number


def _build_case(
    *,
    lab_number: int,
    lab_name: str,
    pdf_path: Path,
    question_type: str,
    question: str,
    expected_excerpt: str,
    section_name: str,
    heading: str,
    page_num: Optional[int],
    references: list[str],
) -> ValidationCase:
    keywords = _build_keywords(section_name, heading, expected_excerpt, references)
    return ValidationCase(
        lab_number=lab_number,
        lab_name=lab_name,
        manual_path=str(pdf_path),
        question_type=question_type,
        question=question,
        expected_excerpt=expected_excerpt,
        section_name=section_name,
        heading=heading,
        page_num=page_num,
        keywords=keywords,
        references=references,
    )


def _candidate_sort_key(case: ValidationCase) -> tuple[int, int, int, int, int]:
    priority_group, priority_order = _question_priority(case.question_type, case.section_name)
    page_num = case.page_num if isinstance(case.page_num, int) else 999

    quality = 0
    if case.question_type == "task":
        if re.match(r"(?i)^task\s*#?\s*\d+", case.expected_excerpt):
            quality += 3
        if _task_label_for_question(case.section_name) != _normalize_label(case.section_name):
            quality += 2
        if _has_action_language(case.expected_excerpt):
            quality += 2
    if case.question_type == "theory":
        section_lower = case.section_name.lower()
        if section_lower.startswith("theory"):
            quality += 3
        elif "background" in section_lower:
            quality += 2
    if case.question_type == "results":
        for term in ("record", "reproduce", "table", "plot", "graph", "sample calculations", "raw", "measured"):
            if term in case.expected_excerpt.lower():
                quality += 1
    if case.question_type == "procedure" and _has_action_language(case.expected_excerpt):
        quality += 2
    if case.question_type == "reference" and case.references:
        quality += 1
        if case.references[0].count("."):
            quality += 2
        if case.expected_excerpt.lower().startswith(case.references[0].lower()):
            quality += 2
    if 60 <= len(case.expected_excerpt) <= 320:
        quality += 1

    return (
        priority_group,
        priority_order,
        -quality,
        page_num,
        len(_normalize_label(case.section_name).split()),
    )


def _question_priority(question_type: str, section_name: str) -> tuple[int, int]:
    if question_type == "objective":
        return (0, 0)
    if question_type == "prelab":
        return (1, 0)
    if question_type == "materials":
        return (2, 0)
    if question_type == "theory":
        return (3, 0)
    if question_type == "procedure":
        return (4, 0)
    if question_type == "task":
        task_match = TASK_PATTERN.search(section_name)
        return (5, int(task_match.group(1)) if task_match else 999)
    if question_type == "results":
        return (6, 0)
    if question_type == "analysis":
        return (7, 0)
    if question_type in {"discussion", "report", "deliverable", "checkoff", "questions", "conclusion"}:
        return (8, 0)
    if question_type == "reference":
        return (9, 0)
    return (10, 0)


def _question_from_chunk(
    lab_name: str,
    section_name: str,
    heading: str,
    excerpt: str,
) -> tuple[Optional[str], Optional[str]]:
    section_name = _normalize_label(embed.normalize_section_label(section_name))
    section_lower = section_name.lower()
    heading = _normalize_label(heading)
    heading_lower = heading.lower()
    excerpt_lower = excerpt.lower()

    if "objective" in section_lower or "goal" in section_lower:
        return "objective", f"What is the objective of {lab_name}?"
    if "pre-lab" in section_lower or "prelab" in section_lower:
        return "prelab", f"What does the pre-lab require for {lab_name}?"
    if "materials" in section_lower or "parts" in section_lower or "equipment" in section_lower:
        return "materials", f"What equipment or materials are needed for {lab_name}?"
    if "theory" in section_lower or "introduction" in section_lower or "background" in section_lower:
        return "theory", f"What theory or background does {lab_name} provide?"
    if "procedure" in section_lower:
        if (
            "lab report" in excerpt_lower
            or ("summarize" in excerpt_lower and "procedure" in excerpt_lower)
            or "write a detailed procedure" in excerpt_lower
            or "procedure section" in excerpt_lower
        ):
            return None, None
        return "procedure", f"What procedure does {lab_name} ask students to follow?"
    if section_name.startswith("Task "):
        task_label = _task_label_for_question(section_name)
        if _looks_noisy_label(task_label) and not _has_action_language(excerpt):
            return None, None
        return "task", f"What does {task_label} ask students to do in {lab_name}?"
    if "results" in section_lower:
        if not any(
            term in excerpt_lower
            for term in ("data", "record", "table", "plot", "graph", "raw", "sample calculations", "reproduce")
        ):
            return None, None
        return "results", f"What results or data does {lab_name} ask students to provide?"
    if "analysis" in section_lower:
        return "analysis", f"What analysis is required in {lab_name}?"
    if "discussion" in section_lower:
        if "discussion" not in excerpt_lower and "discuss" not in excerpt_lower:
            return None, None
        return "discussion", f"What should students discuss in {lab_name}?"
    if "deliverable" in section_lower:
        return "deliverable", f"What deliverables are required for {lab_name}?"
    if "report" in section_lower:
        return "report", f"What needs to be included in the report for {lab_name}?"
    if "checkoff" in section_lower:
        return "checkoff", f"What is required for checkoff in {lab_name}?"
    if "question" in section_lower:
        if "question" not in excerpt_lower and "(a)" not in excerpt_lower and "answer the following" not in excerpt_lower:
            return None, None
        return "questions", f"What questions does the manual ask students to answer for {lab_name}?"
    if "conclusion" in section_lower:
        return "conclusion", f"What conclusion is expected in {lab_name}?"

    return None, None


def generate_cases_for_manual(pdf_path: Path, cases_per_lab: int) -> list[ValidationCase]:
    lab_name = embed.extract_clean_lab_name(pdf_path.name)
    lab_number = _extract_lab_number(lab_name)
    if lab_number is None:
        return []

    pages = embed.preprocess_pages(PyPDFLoader(str(pdf_path)).load())
    chunks = embed.chunk_documents(pages)

    candidates: list[ValidationCase] = []
    last_section: Optional[str] = None

    for chunk in chunks:
        content = chunk.page_content.strip()
        if len(content) < 80:
            continue

        section_name = embed.normalize_section_label(embed.infer_section_name(content, fallback=last_section))
        last_section = section_name
        excerpt = _sentence_excerpt(content)
        heading = embed._extract_heading_candidate(content) or section_name
        references = _extract_reference_labels(content, embed.extract_reference_tags(content))
        page_num_raw = chunk.metadata.get("page") if chunk.metadata else None
        page_num = page_num_raw + 1 if isinstance(page_num_raw, int) else page_num_raw

        question_type, question = _question_from_chunk(lab_name, section_name, heading, excerpt)
        if question_type and question:
            if not (_looks_noisy_label(section_name) and question_type in {"section", "reference"}):
                if question_type != "task" or _has_action_language(excerpt):
                    candidates.append(
                        _build_case(
                            lab_number=lab_number,
                            lab_name=lab_name,
                            pdf_path=pdf_path,
                            question_type=question_type,
                            question=question,
                            expected_excerpt=excerpt,
                            section_name=section_name,
                            heading=heading,
                            page_num=page_num,
                            references=references,
                        )
                    )

        kept_references = [
            reference_label
            for reference_label in references
            if _reference_matches_lab(reference_label, lab_number)
        ]
        for reference_label in kept_references[:MAX_REFERENCE_CASES_PER_CHUNK]:
            candidates.append(
                _build_case(
                    lab_number=lab_number,
                    lab_name=lab_name,
                    pdf_path=pdf_path,
                    question_type="reference",
                    question=_build_reference_question(lab_name, reference_label),
                    expected_excerpt=excerpt,
                    section_name=reference_label,
                    heading=reference_label,
                    page_num=page_num,
                    references=[reference_label],
                )
            )

    unique_cases: list[ValidationCase] = []
    seen_questions: set[str] = set()
    seen_section_keys: set[tuple[Any, ...]] = set()
    type_counts: Counter[str] = Counter()
    ordered_candidates = sorted(candidates, key=_candidate_sort_key)

    for candidate in ordered_candidates:
        section_key = (candidate.question_type, candidate.section_name.lower())
        if candidate.question in seen_questions or section_key in seen_section_keys:
            continue
        limit = QUESTION_TYPE_LIMITS.get(candidate.question_type, 1)
        if type_counts[candidate.question_type] >= limit:
            continue
        seen_questions.add(candidate.question)
        seen_section_keys.add(section_key)
        type_counts[candidate.question_type] += 1
        unique_cases.append(candidate)
        if len(unique_cases) >= cases_per_lab:
            break

    if len(unique_cases) < cases_per_lab:
        for candidate in ordered_candidates:
            section_key = (candidate.question_type, candidate.section_name.lower(), candidate.page_num)
            if candidate.question in seen_questions or section_key in seen_section_keys:
                continue
            seen_questions.add(candidate.question)
            seen_section_keys.add(section_key)
            unique_cases.append(candidate)
            if len(unique_cases) >= cases_per_lab:
                break

    return unique_cases


def generate_cases(manuals_dir: Path, selected_labs: Optional[set[int]], cases_per_lab: int) -> list[ValidationCase]:
    cases: list[ValidationCase] = []
    for pdf_path in sorted(manuals_dir.glob("*.pdf")):
        lab_name = embed.extract_clean_lab_name(pdf_path.name)
        lab_number = _extract_lab_number(lab_name)
        if lab_number is None:
            continue
        if selected_labs and lab_number not in selected_labs:
            continue
        cases.extend(generate_cases_for_manual(pdf_path, cases_per_lab))
    return cases


def _ask_question(base_url: str, lab_number: int, question: str, timeout: int) -> tuple[bool, int, str]:
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/{lab_number}",
        json={"question": question},
        timeout=timeout,
    )
    try:
        payload = response.json()
    except Exception:
        payload = response.text

    if response.status_code >= 400:
        return False, response.status_code, str(payload)

    if not isinstance(payload, dict) or "answer" not in payload:
        return False, response.status_code, f"Unexpected response shape: {payload}"

    return True, response.status_code, str(payload["answer"])


def _evaluate_case(
    case: ValidationCase,
    answer: str,
    ok: bool,
    status_code: int,
    semantic_embedder: Optional[OllamaEmbeddings],
    semantic_threshold: float,
    keyword_threshold: float,
    error: Optional[str] = None,
) -> ValidationResult:
    normalized_answer = _normalize_answer(answer)
    has_citation = bool(re.search(r"\[[^\]]+\]", answer))
    is_fallback = normalized_answer.strip() == FALLBACK_ANSWER

    keyword_hits = 0
    answer_lower = normalized_answer.lower()
    for keyword in case.keywords:
        if keyword in answer_lower:
            keyword_hits += 1
    keyword_recall = keyword_hits / max(len(case.keywords), 1)

    semantic_similarity: Optional[float] = None
    if semantic_embedder and ok and not is_fallback:
        try:
            vectors = semantic_embedder.embed_documents([case.expected_excerpt, normalized_answer])
            semantic_similarity = _cosine_similarity(vectors[0], vectors[1])
        except Exception:
            semantic_similarity = None

    pass_by_similarity = semantic_similarity is not None and semantic_similarity >= semantic_threshold
    pass_by_keywords = keyword_recall >= keyword_threshold
    passed = ok and not is_fallback and (pass_by_similarity or pass_by_keywords)

    return ValidationResult(
        case=case,
        ok=ok,
        status_code=status_code,
        answer=answer,
        has_citation=has_citation,
        is_fallback=is_fallback,
        keyword_hits=keyword_hits,
        keyword_recall=keyword_recall,
        semantic_similarity=semantic_similarity,
        passed=passed,
        error=error,
    )


def _print_case(result: ValidationResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    semantic = f"{result.semantic_similarity:.3f}" if result.semantic_similarity is not None else "n/a"
    print(
        f"{status} | Lab {result.case.lab_number} | {result.case.section_name} | "
        f"sim={semantic} | keywords={result.keyword_hits}/{max(len(result.case.keywords), 1)}"
    )
    print(f"Q: {result.case.question}")
    print(f"A: {result.answer}")
    if not result.passed:
        print(f"Expected: {result.case.expected_excerpt}")
        if result.error:
            print(f"Error: {result.error}")
    print()


def _results_to_json(results: list[ValidationResult]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for result in results:
        row = asdict(result)
        output.append(row)
    return output


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    manuals_dir = Path(args.manuals_dir)
    selected_labs = set(args.labs) if args.labs else None

    cases = generate_cases(manuals_dir, selected_labs, args.cases_per_lab)
    if not cases:
        print("No validation cases were generated from the selected lab manuals.")
        return 1

    print(f"Generated {len(cases)} validation questions from the lab manuals.")
    if args.dry_run:
        for case in cases:
            print(f"Lab {case.lab_number}: {case.question}")
            print(f"Expected section: {case.section_name} | p.{case.page_num or '?'}")
            print(f"Expected excerpt: {case.expected_excerpt}")
            print()
        return 0

    semantic_embedder: Optional[OllamaEmbeddings] = None
    try:
        semantic_embedder = OllamaEmbeddings(model=embed.EMBED_MODEL)
    except Exception:
        semantic_embedder = None

    results: list[ValidationResult] = []
    for case in cases:
        try:
            ok, status_code, answer = _ask_question(args.base_url, case.lab_number, case.question, args.timeout)
            result = _evaluate_case(
                case=case,
                answer=answer,
                ok=ok,
                status_code=status_code,
                semantic_embedder=semantic_embedder,
                semantic_threshold=args.semantic_threshold,
                keyword_threshold=args.keyword_threshold,
                error=None if ok else answer,
            )
        except requests.RequestException as e:
            result = _evaluate_case(
                case=case,
                answer=str(e),
                ok=False,
                status_code=0,
                semantic_embedder=None,
                semantic_threshold=args.semantic_threshold,
                keyword_threshold=args.keyword_threshold,
                error=str(e),
            )
        results.append(result)
        if args.show_passes or not result.passed:
            _print_case(result)

    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    semantic_scores = [result.semantic_similarity for result in results if result.semantic_similarity is not None]
    avg_semantic = sum(semantic_scores) / len(semantic_scores) if semantic_scores else None
    avg_keyword = sum(result.keyword_recall for result in results) / len(results)

    print("Summary")
    print(f"Cases: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {passed / len(results):.1%}")
    print(f"Average keyword recall: {avg_keyword:.3f}")
    if avg_semantic is not None:
        print(f"Average semantic similarity: {avg_semantic:.3f}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(_results_to_json(results), indent=2), encoding="utf-8")
        print(f"Saved detailed results to {output_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
