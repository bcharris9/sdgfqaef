from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client

load_dotenv()
SUPABASE_URL = "https://mvyumvpmzcrrcwcppcea.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im12eXVtdnBtemNycmN3Y3BwY2VhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE2Njk2MDQsImV4cCI6MjA3NzI0NTYwNH0.WfjqQowIt9lxKPdnWSGEOP_u7MKmetWgIPFOASuzeBw"

MANUALS_DIR = Path(os.getenv("LAB_MANUALS_DIR", "./MANUALS"))
CHUNK_SIZE = int(os.getenv("LAB_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("LAB_CHUNK_OVERLAP", "180"))
MANUAL_VERSION = os.getenv("LAB_MANUAL_VERSION", "v2")
MAX_REFERENCE_TAGS = int(os.getenv("LAB_MAX_REFERENCE_TAGS", "6"))
EMBED_MODEL = os.getenv("LAB_EMBED_MODEL", "mxbai-embed-large")

supabase = None
embedder = None

try:
    if SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        embedder = OllamaEmbeddings(model=EMBED_MODEL)
    else:
        print("WARNING: SUPABASE_KEY missing.")
except Exception as e:
    print(f"Startup Error: {e}")


SECTION_HINTS = [
    (r"goals? for lab", "Goals"),
    (r"\bobjective", "Objectives"),
    (r"\btheory\b", "Theory"),
    (r"\bintroduction\b", "Introduction"),
    (r"\bbackground\b", "Background"),
    (r"\bpre[- ]?lab\b", "Pre-Lab"),
    (r"parts needed|equipment|materials", "Materials / Parts"),
    (r"\bprocedure\b", "Procedure"),
    (r"tables? and results", "Tables & Results"),
    (r"\bresults?\b", "Results"),
    (r"\banalysis\b", "Analysis"),
    (r"\bdiscussion\b", "Discussion"),
    (r"\bquestions?\b", "Questions"),
    (r"\breport\b", "Report"),
    (r"\bdeliverables?\b", "Deliverables"),
    (r"\bcheck[- ]?off\b", "Checkoff"),
    (r"\bconclusion\b", "Conclusion"),
]

SECTION_TITLE_KEYWORDS = [
    "goal",
    "objective",
    "introduction",
    "background",
    "theory",
    "prelab",
    "procedure",
    "task",
    "table",
    "result",
    "analysis",
    "discussion",
    "question",
    "report",
    "deliverable",
    "checkoff",
    "conclusion",
    "appendix",
    "material",
    "equipment",
    "part",
    "figure",
]

REFERENCE_PATTERNS = [
    re.compile(r"\btask\s*#?\s*\d+[a-z]?\b", re.IGNORECASE),
    re.compile(r"\bfigure\.?\s*\d+(?:\s*\.\s*\d+)*(?:[a-z](?![a-z]))?(?=[^0-9]|$)", re.IGNORECASE),
    re.compile(r"\btable\.?\s*\d+(?:\s*\.\s*\d+)*(?:[a-z](?![a-z]))?(?=[^0-9]|$)", re.IGNORECASE),
    re.compile(r"\bappendix\s*[a-z0-9]+\b", re.IGNORECASE),
    re.compile(r"\beq(?:uation)?\.?\s*\(?\d+(?:\s*\.\s*\d+)*\)?(?:[a-z](?![a-z]))?(?=[^0-9]|$)", re.IGNORECASE),
]

SECTION_NORMALIZATIONS = [
    (r"\bgoals?\b|\bobjectives?\b|\baim\b|\bpurpose\b", "Goals"),
    (r"\bpre[- ]?lab\b", "Pre-Lab"),
    (r"\btheory\b.*\bintroduction\b|\bintroduction\b.*\btheory\b", "Theory and Introduction"),
    (r"\btheory\b", "Theory"),
    (r"\bintroduction\b", "Introduction"),
    (r"\bbackground\b", "Background"),
    (r"\bparts used\b|\bmaterials?\b|\bequipment\b|\bparts needed\b|\bparts list\b", "Materials / Parts"),
    (r"\bprocedure\b", "Procedure"),
    (r"\btables?\b.*\bresults?\b|\bdata tables?\b|\bresults?\b", "Results"),
    (r"\banalysis\b", "Analysis"),
    (r"\bdiscussion\b", "Discussion"),
    (r"\bquestions?\b", "Questions"),
    (r"\breport\b", "Report"),
    (r"\bdeliverables?\b", "Deliverables"),
    (r"\bcheck[- ]?off\b", "Checkoff"),
    (r"\bconclusion\b", "Conclusion"),
]

INLINE_HEADING_PATTERNS = [
    r"Goals?\s+for\s+Lab\b",
    r"Pre[- ]?lab\b",
    r"Theory(?:\s*(?:&|and)\s*Introduction)?\b",
    r"Introduction\b",
    r"Background\b",
    r"Procedure\b",
    r"Parts?\s+(?:used|needed|list)\b",
    r"Equipment(?:\s+and\s+parts\s+needed)?\b",
    r"Materials?\b",
    r"Task\s*#?\s*\d+\b",
    r"Tables?\s+and\s+Results\b",
    r"Results?\b",
    r"Analysis\b",
    r"Discussion\b",
    r"Questions?\b",
    r"Report(?:\s+Requirements)?\b",
    r"Deliverables?\b",
    r"Check[- ]?off\b",
]


def extract_clean_lab_name(filename: str) -> str:
    """Normalize filenames to 'Lab X' labels."""
    match = re.search(r"lab\s*0?(\d+)", filename, re.IGNORECASE)
    if match:
        return f"Lab {match.group(1)}"
    return Path(filename).stem


def extract_manual_title(filename: str) -> str:
    stem = Path(filename).stem.replace("_", " ")
    stem = re.sub(r"\s+", " ", stem).strip(" -")
    stem = re.sub(r"^lab\s*0?\d+\s*", "", stem, flags=re.IGNORECASE).strip(" -:")
    return stem or Path(filename).stem


def _normalize_unicode_text(text: str) -> str:
    return (
        text.replace("\u00a0", " ")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def _normalize_section_text(text: str) -> str:
    return _normalize_unicode_text(text).lower()


def _insert_inline_heading_breaks(text: str) -> str:
    updated = text
    for pattern in INLINE_HEADING_PATTERNS:
        updated = re.sub(
            rf"(?<!\n)\s+({pattern})",
            r"\n\1",
            updated,
            flags=re.IGNORECASE,
        )
    return updated


def clean_page_text(text: str) -> str:
    """Strip headers/footers and normalize whitespace while preserving references."""
    text = _normalize_unicode_text(text)
    text = re.sub(r"(?m)^\s*revised[^\n]*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*page\s+\d+\b[^\n]*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*page\s+\d+\s+of\s+\d+\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = text.replace("\xad", "")
    text = text.replace("\u2022", "- ").replace("\uf0b7", "- ")
    text = re.sub(r"\b(Lab)(\d+)\b", r"\1 \2", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s*&\s*", " & ", text)
    text = _insert_inline_heading_breaks(text)
    text = re.sub(r"(\w)-\s*\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_table_header(line: str) -> bool:
    tokens = line.split()
    if len(tokens) < 4:
        return False
    short_tokens = 0
    long_alpha_tokens = 0
    for token in tokens:
        letters = re.sub(r"[^A-Za-z]", "", token)
        if len(letters) <= 2:
            short_tokens += 1
        if len(letters) >= 4:
            long_alpha_tokens += 1
    return short_tokens >= max(3, len(tokens) - 2) and long_alpha_tokens <= 2


def normalize_section_label(label: str | None) -> str:
    if not label:
        return "General"
    cleaned = re.sub(r"\s+", " ", _normalize_unicode_text(label)).strip(" :-")
    normalized = cleaned.lower()

    task_match = re.search(r"task\s*#?\s*(\d+)[\s:-]*([^\n\.]+)?", cleaned, re.IGNORECASE)
    if task_match:
        title = (task_match.group(2) or "").strip(" :-#")
        return f"Task {task_match.group(1)}" + (f": {title}" if title else "")

    for pattern, canonical in SECTION_NORMALIZATIONS:
        if re.search(pattern, normalized):
            return canonical

    if cleaned.lower().startswith(("figure ", "table ", "appendix ")):
        return cleaned

    return cleaned or "General"


def _extract_heading_candidate(text: str) -> str | None:
    """Try to grab a short heading-like line near the top of the chunk."""
    for raw_line in text.splitlines():
        line = raw_line.strip(" :-\t")
        if not line:
            continue
        if re.match(r"revised\b", line, re.IGNORECASE):
            continue
        if re.match(r"lab\s*\d+[: ]", line, re.IGNORECASE):
            continue
        if line.endswith("."):
            continue
        words = line.split()
        if len(words) <= 1 or len(words) > 14:
            continue
        if _looks_like_table_header(line):
            continue
        if re.match(r"(task|figure|table|appendix)\b", line, re.IGNORECASE):
            return line
        if not line[0].isalpha():
            continue
        if line.isupper() or line[0].isupper():
            return line
    return None


def infer_section_name(text: str, fallback: str | None = None) -> str:
    """Create a descriptive section label from chunk content."""
    normalized = _normalize_section_text(text)
    window = normalized[:700]
    lead_window = window[:220]

    for pattern, label in SECTION_HINTS:
        if re.search(pattern, lead_window):
            return normalize_section_label(label)

    task_match = re.search(r"task\s*#?\s*(\d+)[\s:-]*([^\n\.]+)?", window)
    if task_match:
        title = (task_match.group(2) or "").strip(" :-#")
        return normalize_section_label(
            f"Task {task_match.group(1)}" + (f": {title}" if title else "")
        )

    for pattern, label in SECTION_HINTS:
        if re.search(pattern, window):
            return normalize_section_label(label)

    heading = _extract_heading_candidate(text)
    if heading:
        if any(keyword in heading.lower() for keyword in SECTION_TITLE_KEYWORDS):
            return normalize_section_label(heading)
        if fallback:
            return normalize_section_label(fallback)
        return normalize_section_label(heading)

    if fallback:
        return normalize_section_label(fallback)

    return "General"


def preprocess_pages(pages: list[Document]) -> list[Document]:
    cleaned: list[Document] = []
    for doc in pages:
        cleaned_text = clean_page_text(doc.page_content)
        if not cleaned_text:
            continue
        metadata = dict(doc.metadata or {})
        cleaned.append(Document(page_content=cleaned_text, metadata=metadata))
    return cleaned


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    normalized = stripped.lower()
    if re.match(r"(task|figure|table|appendix)\b", normalized):
        return True
    if stripped.isupper() and len(stripped.split()) <= 14:
        return True
    return len(stripped.split()) <= 14 and any(
        keyword in normalized for keyword in SECTION_TITLE_KEYWORDS
    )


def _split_by_headings(text: str) -> list[str]:
    """
    Split a page into heading-led sections so each chunk keeps a stronger local topic.
    """
    sections: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if _is_heading_line(stripped) and current:
            sections.append("\n".join(current).strip())
            current = [line]
            continue
        current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    return [section for section in sections if section]


def chunk_documents(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ": ", " "],
    )
    pre_split_docs: list[Document] = []
    for doc in pages:
        sections = _split_by_headings(doc.page_content) or [doc.page_content]
        for index, section_text in enumerate(sections, start=1):
            metadata = dict(doc.metadata or {})
            metadata["page_section"] = index
            pre_split_docs.append(Document(page_content=section_text, metadata=metadata))
    return splitter.split_documents(pre_split_docs)


def extract_reference_tags(text: str) -> list[str]:
    references: list[str] = []
    seen: set[str] = set()
    for pattern in REFERENCE_PATTERNS:
        for match in pattern.finditer(text):
            tag = re.sub(r"\s+", " ", match.group(0)).strip()
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            references.append(tag)
            if len(references) >= MAX_REFERENCE_TAGS:
                return references
    return references


def build_embedding_text(
    *,
    lab_name: str,
    manual_title: str,
    section_name: str,
    heading: str,
    page_num: int | str | None,
    references: list[str],
    content: str,
) -> str:
    parts = [f"Manual: {lab_name}"]
    if manual_title and manual_title != lab_name:
        parts.append(f"Title: {manual_title}")
    parts.append(f"Section: {section_name}")
    if heading and heading.lower() != section_name.lower():
        parts.append(f"Heading: {heading}")
    if page_num not in (None, ""):
        parts.append(f"Page: {page_num}")
    if references:
        parts.append(f"References: {', '.join(references)}")
    parts.append("Content:")
    parts.append(content)
    return "\n".join(parts)


def _raise_with_ollama_hint(error: Exception, model_name: str) -> None:
    message = str(error)
    if "not found" in message.lower() and model_name in message:
        raise RuntimeError(
            f'Ollama embedding model "{model_name}" is not installed. '
            f'Run: ollama pull {model_name}'
        ) from error
    raise error


def main() -> None:
    print("Loading lab manuals...")
    if not MANUALS_DIR.is_dir():
        print(f"Manuals directory not found: {MANUALS_DIR}. Create it and add PDFs.")
        return
    if not supabase or not embedder:
        print("Supabase client or embedder not initialized. Exiting.")
        return

    all_pdfs = sorted(path for path in MANUALS_DIR.iterdir() if path.suffix.lower() == ".pdf")
    if not all_pdfs:
        print(f"No PDFs found in {MANUALS_DIR}")
        return

    seen_hashes: set[str] = set()

    for pdf_path in all_pdfs:
        clean_lab_name = extract_clean_lab_name(pdf_path.name)
        manual_title = extract_manual_title(pdf_path.name)
        print(f"Processing: {pdf_path.name} -> ID: {clean_lab_name}")

        loader = PyPDFLoader(str(pdf_path))
        pages = preprocess_pages(loader.load())
        chunks = chunk_documents(pages)

        print(f"   - Uploading {len(chunks)} chunks for {clean_lab_name}...")
        last_section = None
        chunk_payloads: list[dict[str, object]] = []
        chunk_texts: list[str] = []
        chunk_counter = 0

        for chunk in chunks:
            content = chunk.page_content.strip()
            if not content or len(content) < 40:
                continue

            page_num_raw = chunk.metadata.get("page") if chunk.metadata else None
            page_num = page_num_raw + 1 if isinstance(page_num_raw, int) else page_num_raw
            hash_basis = f"{MANUAL_VERSION}|{clean_lab_name}|{page_num}|{content}"
            chunk_hash = hashlib.sha256(hash_basis.encode("utf-8")).hexdigest()
            if chunk_hash in seen_hashes:
                continue
            seen_hashes.add(chunk_hash)

            chunk_counter += 1
            section_name = normalize_section_label(infer_section_name(content, fallback=last_section))
            last_section = section_name
            heading = _extract_heading_candidate(content) or section_name
            references = extract_reference_tags(content)
            token_count = len(re.findall(r"\w+", content))
            embedding_text = build_embedding_text(
                lab_name=clean_lab_name,
                manual_title=manual_title,
                section_name=section_name,
                heading=heading,
                page_num=page_num,
                references=references,
                content=content,
            )

            data: dict[str, object] = {
                "lab_name": clean_lab_name,
                "manual_version": MANUAL_VERSION,
                "section_name": section_name,
                "heading": heading,
                "content": content,
                "chunk_hash": chunk_hash,
                "chunk_order": chunk_counter,
                "token_count": token_count,
            }
            if page_num is not None:
                data["page_num"] = page_num

            chunk_payloads.append(data)
            chunk_texts.append(embedding_text)

        if not chunk_payloads:
            continue

        try:
            vectors = embedder.embed_documents(chunk_texts)
        except Exception as e:
            _raise_with_ollama_hint(e, EMBED_MODEL)
        for record, vector in zip(chunk_payloads, vectors):
            record["embedding"] = vector

        supabase.table("lab_sections").upsert(
            chunk_payloads,
            on_conflict="chunk_hash",
        ).execute()

    print("Database refreshed with cleaner chunk metadata and embeddings.")


if __name__ == "__main__":
    if not SUPABASE_KEY:
        print("Supabase key missing. Exiting.")
        sys.exit(1)
    main()
