"""
End-to-end test for the LangGraph Critic agent.
Usage (from backend/):
    python scripts/test_agent.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.critic import ask

QUESTIONS = [
    ("What is the standard deduction for single filers?", 2025),
    ("Can I deduct mortgage interest on my taxes?", 2025),
    ("What is the capital gains tax rate for collectibles?", 2025),
]

for question, year in QUESTIONS:
    print(f"\n{'='*70}")
    response = ask(question, tax_year=year)
    print(f"\nAnswer:\n{response.answer.answer}")
    print(f"\nCitations: {response.answer.page_citations}")
    print(f"Confidence: {response.answer.confidence:.3f}")
    print(f"Attempts:   {response.attempts}")
    print(f"Rewritten:  {response.rewritten_query}")
