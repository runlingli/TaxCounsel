"""Quick smoke-test for the PDF parser. Run from backend/: python scripts/test_parser.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.parser import parse_irs_pdf

PDF_PATH = Path("data/irs_pdfs/p17.pdf")

nodes = parse_irs_pdf(PDF_PATH, source="IRS Pub 17", tax_year=2024)

print(f"Total chunks: {len(nodes)}")
print("\n--- First chunk ---")
print(f"  page={nodes[0].metadata['page']}  section={nodes[0].metadata['section']}")
print(f"  text preview: {nodes[0].text[:200]}")
print("\n--- Last chunk ---")
print(f"  page={nodes[-1].metadata['page']}  section={nodes[-1].metadata['section']}")
print(f"  text preview: {nodes[-1].text[:200]}")
