import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_preprocessing import TextPreprocessor

pp = TextPreprocessor()

tests = [
    "This product is amazing!",
    "Love it!",
    "Good",
    "5 stars excellent",
    "Best purchase ever",
    "Not good at all",
    "!!!",
    "Amazing product highly recommend 5 stars",
]

print("PREPROCESSING TEST")
print("="*70)

for text in tests:
    result = pp.preprocess(text)
    print(f"\n'{text}'")
    print(f"→ '{result}'")
    print(f"  Length: {len(result)} chars, Empty: {len(result) == 0}")

print("\n" + "="*70)
