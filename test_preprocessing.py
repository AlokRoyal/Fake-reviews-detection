import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()

test_reviews = [
    "This product is amazing!",
    "Love it!",
    "Good",
    "!!!???",
    "123",
    "The. Quick. Brown. Fox.",
    "Don't like it",
    "5 stars! Excellent quality.",
    "Really great product, highly recommend!",
    "OK",
]

print("="*70)
print("PREPROCESSING TEST")
print("="*70)

for review in test_reviews:
    processed = preprocessor.preprocess(review)
    empty = "EMPTY" if not processed else f"{len(processed)} chars"
    print(f"\n'{review[:40]:40s}' → [{empty}]")
    if processed:
        print(f"  Result: {processed[:60]}")

print("\n" + "="*70)
