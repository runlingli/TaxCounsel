"""
Run full ingestion pipeline: PDF → chunks → Qdrant.

Two parsers are used:
  - parse_irs_pdf_hierarchical: for text-heavy PDFs (Pub 17, 525, 531 …)
  - parse_irs_pdf_unstructured: for table-heavy PDFs (Pub 596, 501 …)
    uses the `unstructured` library which understands HTML tables in PDFs.

Usage (from backend/):
    python scripts/ingest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core.schema import TextNode

from app.ingestion.indexer import build_and_upload
from app.ingestion.parser import (
    CHILD_OVERLAP,
    CHILD_SIZE,
    PARENT_MAX,
    _extract_xrefs,
    _semantic_paragraphs,
    _slide_words,
    parse_irs_pdf_hierarchical,
)


def parse_irs_pdf_unstructured(
    pdf_path: str | Path,
    source: str,
    tax_year: int,
) -> list[TextNode]:
    """
    Parse table-heavy IRS PDFs using unstructured's partition_pdf.
    Tables are converted to plain text rows, then fed into the same
    semantic parent-child chunking pipeline.
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        print("  WARNING: unstructured not available, falling back to pdfplumber")
        return parse_irs_pdf_hierarchical(pdf_path, source, tax_year)

    print(f"  Using unstructured parser for {source}...")
    elements = partition_pdf(str(pdf_path), strategy="fast")

    # Group elements by page, build page text
    page_texts: dict[int, list[str]] = {}
    for el in elements:
        pn = getattr(el.metadata, "page_number", 1) or 1
        page_texts.setdefault(pn, []).append(el.text or "")

    nodes: list[TextNode] = []
    node_id = 0

    for page_num in sorted(page_texts):
        page_text = "\n\n".join(t for t in page_texts[page_num] if t.strip())
        if not page_text.strip():
            continue

        for parent_text in _semantic_paragraphs(page_text, PARENT_MAX):
            if not parent_text.strip():
                continue
            xrefs = _extract_xrefs(parent_text)
            parent_words = parent_text.split()
            for offset, child_words in _slide_words(parent_words, CHILD_SIZE, CHILD_OVERLAP):
                if not child_words:
                    continue
                nodes.append(TextNode(
                    id_=str(node_id),
                    text=" ".join(child_words),
                    metadata={
                        "source":         source,
                        "tax_year":       tax_year,
                        "page":           page_num,
                        "section":        "General",
                        "parent_text":    parent_text,
                        "parent_page":    page_num,
                        "parent_section": "General",
                        "cross_refs":     xrefs,
                    },
                ))
                node_id += 1

    return nodes


# PDFs and which parser to use
# table=True → unstructured;  table=False → pdfplumber hierarchical
PDFS = [
    ("data/irs_pdfs/p17.pdf",  "IRS Pub 17",  2025, False),  # general guide
    ("data/irs_pdfs/p596.pdf", "IRS Pub 596", 2025, False),  # EIC
    ("data/irs_pdfs/p501.pdf", "IRS Pub 501", 2025, False),  # Dependents / CTC
    ("data/irs_pdfs/p502.pdf", "IRS Pub 502", 2025, False),  # Medical / dental expenses
    ("data/irs_pdfs/p525.pdf", "IRS Pub 525", 2025, False),  # Taxable income
    ("data/irs_pdfs/p463.pdf", "IRS Pub 463", 2025, False),  # Mileage / travel
    ("data/irs_pdfs/p531.pdf", "IRS Pub 531", 2025, False),  # Tip income
    ("data/irs_pdfs/p334.pdf", "IRS Pub 334", 2025, False),  # Self-employed / Schedule C
    ("data/irs_pdfs/p550.pdf", "IRS Pub 550", 2025, False),  # Investment income / capital gains
]

if __name__ == "__main__":
    all_nodes: list[TextNode] = []
    for pdf_path, source, tax_year, use_unstructured in PDFS:
        path = Path(pdf_path)
        if not path.exists():
            print(f"  SKIP {source} — not found: {pdf_path}")
            continue
        print(f"\nParsing {source} ...")
        if use_unstructured:
            nodes = parse_irs_pdf_unstructured(path, source=source, tax_year=tax_year)
        else:
            nodes = parse_irs_pdf_hierarchical(path, source=source, tax_year=tax_year)
        print(f"  → {len(nodes)} child chunks")
        all_nodes.extend(nodes)

    print(f"\nTotal chunks: {len(all_nodes)}")
    build_and_upload(all_nodes, tfidf_save_path="data/tfidf.pkl")
    print("\nIngestion complete.")
