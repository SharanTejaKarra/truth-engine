"""Document parsers for the three Truth Engine source types."""

import csv
import json
import logging
import re
from pathlib import Path

import pdfplumber

from app.models import DocumentChunk

logger = logging.getLogger(__name__)


def _read_text_file(path: Path) -> str:
    """Read a text file with encoding fallback."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def parse_pdf(file_path: str) -> list[DocumentChunk]:
    """Parse PDF and plain-text files into DocumentChunks.

    PDFs: extracts text page-by-page using pdfplumber, including tables as
    formatted text.
    TXT: reads plain text with section detection (ALL-CAPS lines or '## '
    prefixes treated as section headers).
    """
    path = Path(file_path)
    filename = path.name
    chunks: list[DocumentChunk] = []

    if path.suffix.lower() == ".txt":
        text = _read_text_file(path)
        if not text.strip():
            logger.warning("Empty text file: %s", file_path)
            return []

        sections = _split_txt_sections(text)
        for idx, (title, body) in enumerate(sections):
            if not body.strip():
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=f"manual_{filename}_{idx}",
                    content=body.strip(),
                    source_type="manual",
                    source_file=filename,
                    metadata={
                        "section_title": title,
                    },
                )
            )
        return chunks

    # PDF path
    try:
        pdf = pdfplumber.open(file_path)
    except Exception as e:
        logger.error("Failed to open PDF %s: %s", file_path, e)
        return []

    with pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text_parts: list[str] = []

            text = page.extract_text() or ""
            if text.strip():
                page_text_parts.append(text)

            tables = page.extract_tables()
            for table in tables:
                formatted = _format_table(table)
                if formatted:
                    page_text_parts.append(formatted)

            page_content = "\n\n".join(page_text_parts).strip()
            if not page_content:
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=f"manual_{filename}_{page_num}",
                    content=page_content,
                    source_type="manual",
                    source_file=filename,
                    metadata={
                        "page_number": page_num,
                        "section_title": "",
                    },
                )
            )

    return chunks


def parse_json_csv(file_path: str) -> list[DocumentChunk]:
    """Parse JSON arrays and CSV files from support logs.

    JSON: each ticket object becomes a chunk.
    CSV: each row becomes a chunk.
    Content is formatted as readable text.
    """
    path = Path(file_path)
    filename = path.name
    chunks: list[DocumentChunk] = []

    if path.suffix.lower() == ".json":
        raw = _read_text_file(path)
        if not raw.strip():
            logger.warning("Empty JSON file: %s", file_path)
            return []
        data = json.loads(raw)
        if isinstance(data, dict):
            # Handle nested JSON: recursively find the first array of dicts
            found = _find_records_array(data)
            if found is not None:
                data = found
            else:
                data = [data]
        elif not isinstance(data, list):
            data = [data]

        for idx, ticket in enumerate(data):
            # Flexible field name resolution with broad fallbacks
            ticket_id = (
                ticket.get("ticket_id") or ticket.get("id")
                or ticket.get("ticket") or ticket.get("case_id")
                or ticket.get("incident_id") or ticket.get("ref")
                or ticket.get("reference") or f"unknown-{idx}"
            )
            issue = (
                ticket.get("issue") or ticket.get("summary")
                or ticket.get("description") or ticket.get("problem")
                or ticket.get("title") or ticket.get("subject") or ""
            )
            description = ticket.get("description", "")
            resolution = (
                ticket.get("resolution") or ticket.get("fix")
                or ticket.get("solution") or ticket.get("fix_applied")
                or ticket.get("workaround") or ""
            )
            status = (
                ticket.get("resolution_status") or ticket.get("status")
                or ticket.get("state") or ticket.get("outcome") or ""
            )
            engineer = (
                ticket.get("engineer") or ticket.get("assigned_to")
                or ticket.get("resolved_by") or ticket.get("assignee")
                or ticket.get("owner") or ticket.get("technician")
                or ticket.get("handled_by") or ticket.get("fixed_by") or ""
            )
            timestamp = (
                ticket.get("timestamp") or ticket.get("date")
                or ticket.get("created_at") or ticket.get("updated_at")
                or ticket.get("resolved_at") or ""
            )
            error_code = (
                ticket.get("related_error_code") or ticket.get("error_code")
                or ticket.get("error") or ""
            )
            category = (
                ticket.get("category") or ticket.get("type")
                or ticket.get("component") or ""
            )

            # Build content with ALL relevant fields so they are searchable
            # and visible to the LLM
            content = f"Ticket {ticket_id}"
            if engineer:
                content += f" (Engineer: {engineer})"
            if timestamp:
                content += f" [{timestamp}]"
            content += f": {issue}"
            if description and description != issue:
                content += f". Details: {description}"
            if resolution:
                content += f". Resolution: {resolution}"
            if status:
                content += f". Status: {status}"
            if error_code:
                content += f". Error Code: {error_code}"
            if category:
                content += f". Category: {category}"

            # Preserve ALL original fields as metadata (like CSV parser)
            base_metadata = {}
            for k, v in ticket.items():
                if v is not None:
                    base_metadata[str(k)] = str(v)
                else:
                    base_metadata[str(k)] = ""
            # Ensure canonical keys are always present
            base_metadata["ticket_id"] = str(ticket_id)
            base_metadata["timestamp"] = str(timestamp)
            base_metadata["engineer"] = str(engineer)
            base_metadata["resolution_status"] = str(status)

            chunks.append(
                DocumentChunk(
                    chunk_id=f"support_log_{filename}_{idx}",
                    content=content,
                    source_type="support_log",
                    source_file=filename,
                    metadata=base_metadata,
                )
            )

    elif path.suffix.lower() == ".csv":
        raw = _read_text_file(path)
        if not raw.strip():
            logger.warning("Empty CSV file: %s", file_path)
            return []

        reader = csv.DictReader(raw.splitlines())
        for idx, row in enumerate(reader):
            ticket_id = (
                row.get("ticket_id") or row.get("id") or f"row-{idx}"
            )
            issue = (
                row.get("issue") or row.get("summary")
                or row.get("description") or ""
            )
            description = row.get("description", "")
            resolution = (
                row.get("resolution") or row.get("fix_applied")
                or row.get("fix") or row.get("solution") or ""
            )
            status = (
                row.get("resolution_status") or row.get("status") or ""
            )
            engineer = (
                row.get("engineer") or row.get("assigned_to")
                or row.get("resolved_by") or ""
            )
            timestamp = row.get("date") or row.get("timestamp") or ""
            error_code = row.get("related_error_code") or row.get("error_code") or ""
            category = row.get("category") or row.get("type") or ""

            # Build content with ALL relevant fields
            content = f"Ticket {ticket_id}"
            if engineer:
                content += f" (Engineer: {engineer})"
            if timestamp:
                content += f" [{timestamp}]"
            content += f": {issue}"
            if description and description != issue:
                content += f". Details: {description}"
            if resolution:
                content += f". Resolution: {resolution}"
            if status:
                content += f". Status: {status}"
            if error_code:
                content += f". Error Code: {error_code}"
            if category:
                content += f". Category: {category}"

            metadata = {k: str(v) for k, v in row.items()}
            chunks.append(
                DocumentChunk(
                    chunk_id=f"support_log_{filename}_{idx}",
                    content=content,
                    source_type="support_log",
                    source_file=filename,
                    metadata=metadata,
                )
            )

    return chunks


def parse_markdown(file_path: str) -> list[DocumentChunk]:
    """Parse markdown files, splitting by headers (## and ###).

    Each section becomes a chunk with the section title and header hierarchy
    preserved in metadata.
    """
    path = Path(file_path)
    filename = path.name
    text = _read_text_file(path)
    if not text.strip():
        logger.warning("Empty markdown file: %s", file_path)
        return []

    sections = _split_md_sections(text)
    chunks: list[DocumentChunk] = []

    for idx, (hierarchy, title, body) in enumerate(sections):
        if not body.strip():
            continue
        chunks.append(
            DocumentChunk(
                chunk_id=f"wiki_{filename}_{idx}",
                content=body.strip(),
                source_type="wiki",
                source_file=filename,
                metadata={
                    "section_title": title,
                    "header_hierarchy": " > ".join(hierarchy),
                },
            )
        )

    return chunks


# ── Helpers ────────────────────────────────────────────────────────────


def _find_records_array(data) -> list[dict] | None:
    """Recursively find the first array of dicts in a nested JSON structure."""
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data
    if isinstance(data, dict):
        for val in data.values():
            result = _find_records_array(val)
            if result is not None:
                return result
    return None


def _format_table(table: list[list[str | None]]) -> str:
    """Convert a pdfplumber table to a readable text representation."""
    if not table:
        return ""
    rows = []
    for row in table:
        cells = [str(c).strip() if c else "" for c in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _is_separator_line(stripped: str) -> bool:
    """Check if a line is a visual separator (===, ---, etc.)."""
    if not stripped or len(stripped) < 3:
        return False
    return all(c in "=-_*~" for c in stripped)


def _is_section_header(stripped: str) -> bool:
    """Check if a line looks like a section header.

    Detects ALL-CAPS headers that may contain spaces, numbers, colons, etc.
    (e.g. 'SECTION 1: PRODUCT OVERVIEW', 'TABLE OF CONTENTS').
    """
    if not stripped or len(stripped) <= 3:
        return False
    if _is_separator_line(stripped):
        return False
    alpha_chars = [c for c in stripped if c.isalpha()]
    if (
        len(alpha_chars) >= 3
        and stripped == stripped.upper()
        and len(alpha_chars) / max(len(stripped), 1) > 0.3
    ):
        return True
    return False


def _split_txt_sections(text: str) -> list[tuple[str, str]]:
    """Split plain text into sections based on ALL-CAPS lines or '## ' prefixes.

    Also skips visual separator lines (===, ---, etc.) to avoid noise.
    """
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_title = "Introduction"
    current_body: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if current_body:
                sections.append((current_title, "\n".join(current_body)))
            current_title = stripped[3:].strip()
            current_body = []
        elif _is_section_header(stripped):
            if current_body:
                sections.append((current_title, "\n".join(current_body)))
            current_title = stripped
            current_body = []
        elif _is_separator_line(stripped):
            # Skip decorator lines — don't add to body
            continue
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_title, "\n".join(current_body)))

    return sections


def _split_md_sections(text: str) -> list[tuple[list[str], str, str]]:
    """Split markdown by headers, returning (hierarchy, title, body) tuples."""
    header_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[list[str], str, str]] = []
    hierarchy: list[str] = []
    last_pos = 0
    last_title = "Introduction"

    for match in header_re.finditer(text):
        body = text[last_pos : match.start()]
        if body.strip() or sections:
            sections.append((list(hierarchy), last_title, body))

        level = len(match.group(1))
        title = match.group(2).strip()

        # Maintain hierarchy: trim to current level then append
        hierarchy = [h for h in hierarchy if not h.startswith(f"h{level}:")]
        # Also trim deeper headers
        hierarchy = [
            h for h in hierarchy if int(h.split(":")[0][1:]) < level
        ]
        hierarchy.append(f"h{level}:{title}")

        last_title = title
        last_pos = match.end()

    # Final section
    trailing = text[last_pos:]
    if trailing.strip():
        sections.append((list(hierarchy), last_title, trailing))

    return sections
