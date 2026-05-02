"""
IRS PDF → hierarchical parent-child text chunks.

Chunking strategy: semantic (paragraph-aware) parent chunks, word-count child chunks.

  Parent chunks: split at paragraph / section boundaries so IRS rules are
  never cut mid-sentence. Capped at PARENT_MAX words to prevent runaway chunks.

  Child chunks: fixed-size word windows inside each parent — small and precise
  for dense/sparse vector matching.

  Each child stores its parent's full text in metadata["parent_text"] so the
  Critic and answer generator see the wider context without a separate lookup.
"""

import re
from pathlib import Path

from llama_index.core.schema import TextNode

SECTION_PATTERN = re.compile(
    r"^(Chapter \d+|Part [IVX]+|Section \d+[\.\d]*)", re.IGNORECASE
)
# Cross-reference patterns in IRS text: "See Publication 596", "see chapter 7"
XREF_PATTERN = re.compile(
    r"[Ss]ee\s+(?:Publication|Pub\.?|Chapter|Form|Schedule)\s+([\w\s\d\-]+?)(?=[,.\n]|$)",
    re.IGNORECASE,
)

# Hierarchical sizing — parent smaller than before to avoid topic bleed
PARENT_MAX     = 500   # hard cap on parent chunk words
PARENT_OVERLAP = 60
CHILD_SIZE     = 150   # retrieval unit — precise enough for vectors
CHILD_OVERLAP  = 30

# Legacy flat sizing (kept for rollback)
CHUNK_SIZE = 400
OVERLAP    = 60


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_pages(pdf_path: str | Path) -> list[tuple[int, str, str]]:
    """[(page_num, current_section_heading, page_text), ...]"""
    import pdfplumber

    pages: list[tuple[int, str, str]] = []
    current_section = "General"

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.split("\n"):
                if SECTION_PATTERN.match(line.strip()):
                    current_section = line.strip()
            pages.append((page.page_number, current_section, text))

    return pages


def _semantic_paragraphs(text: str, max_words: int) -> list[str]:
    """
    Split text into paragraph-aware chunks capped at max_words.

    IRS publications use blank lines or indented lines to delimit paragraphs.
    We accumulate paragraphs until the word budget is exhausted, then flush.
    """
    raw_paras = re.split(r"\n{2,}", text.strip())
    chunks: list[str] = []
    buffer: list[str] = []
    buf_words = 0

    for para in raw_paras:
        para = para.strip()
        if not para:
            continue
        pw = len(para.split())
        if buf_words + pw > max_words and buffer:
            chunks.append("\n\n".join(buffer))
            # overlap: keep last paragraph in next chunk
            buffer = [buffer[-1]] if buffer else []
            buf_words = len(buffer[0].split()) if buffer else 0
        buffer.append(para)
        buf_words += pw

    if buffer:
        chunks.append("\n\n".join(buffer))

    return chunks


def _slide_words(words: list[str], size: int, overlap: int):
    """Yield (offset, word_list) fixed-size windows."""
    start = 0
    while start < len(words):
        yield start, words[start : start + size]
        if start + size >= len(words):
            break
        start += size - overlap


def _extract_xrefs(text: str) -> list[str]:
    """Pull IRS cross-reference strings from text (e.g. 'Publication 596')."""
    return [m.group(1).strip() for m in XREF_PATTERN.finditer(text)]


# ---------------------------------------------------------------------------
# Public: semantic hierarchical parser (recommended)
# ---------------------------------------------------------------------------

def parse_irs_pdf_hierarchical(
    pdf_path: str | Path,
    source: str,
    tax_year: int,
) -> list[TextNode]:
    """
    Semantic parent-child chunking.

    Parent: paragraph-aware, capped at PARENT_MAX words.
    Child:  fixed word-window inside parent.

    Metadata per child:
      text, parent_text, source, tax_year, page, section,
      parent_page, parent_section, cross_refs
    """
    pages = _extract_pages(pdf_path)

    nodes: list[TextNode] = []
    node_id = 0

    for page_num, section, page_text in pages:
        parent_chunks = _semantic_paragraphs(page_text, PARENT_MAX)

        for parent_text in parent_chunks:
            if not parent_text.strip():
                continue

            xrefs = _extract_xrefs(parent_text)
            parent_words = parent_text.split()

            for offset, child_words in _slide_words(
                parent_words, CHILD_SIZE, CHILD_OVERLAP
            ):
                if not child_words:
                    continue
                nodes.append(TextNode(
                    id_=str(node_id),
                    text=" ".join(child_words),
                    metadata={
                        "source":         source,
                        "tax_year":       tax_year,
                        "page":           page_num,
                        "section":        section,
                        "parent_text":    parent_text,
                        "parent_page":    page_num,
                        "parent_section": section,
                        "cross_refs":     xrefs,   # for GraphRAG expansion
                    },
                ))
                node_id += 1

    return nodes


# ---------------------------------------------------------------------------
# Public: legacy flat parser (rollback / comparison)
# ---------------------------------------------------------------------------

def parse_irs_pdf(pdf_path: str | Path, source: str, tax_year: int) -> list[TextNode]:
    """Original flat fixed-size chunking."""
    import pdfplumber

    nodes: list[TextNode] = []
    node_id = 0

    with pdfplumber.open(pdf_path) as pdf:
        current_section = "General"
        buffer: list[str] = []
        buf_page_start = 1

        for page in pdf.pages:
            page_num = page.page_number
            text = page.extract_text() or ""
            for line in text.split("\n"):
                if SECTION_PATTERN.match(line.strip()):
                    current_section = line.strip()
            words = text.split()
            if not buffer:
                buf_page_start = page_num
            buffer.extend(words)
            while len(buffer) >= CHUNK_SIZE:
                nodes.append(TextNode(
                    id_=str(node_id),
                    text=" ".join(buffer[:CHUNK_SIZE]),
                    metadata={
                        "source": source, "tax_year": tax_year,
                        "page": buf_page_start, "section": current_section,
                    },
                ))
                node_id += 1
                buffer = buffer[CHUNK_SIZE - OVERLAP:]
                buf_page_start = page_num
        if buffer:
            nodes.append(TextNode(
                id_=str(node_id),
                text=" ".join(buffer),
                metadata={
                    "source": source, "tax_year": tax_year,
                    "page": buf_page_start, "section": current_section,
                },
            ))
    return nodes
