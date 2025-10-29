# test_imports.py
import sys
from pathlib import Path

print("="*60)
print("IMPORT DIAGNOSTIC TEST")
print("="*60)

# Add src to path
src_path = str(Path(__file__).parent / "src")
print(f"\n1. Adding to path: {src_path}")
sys.path.insert(0, src_path)

# Test individual imports
print("\n2. Testing individual module imports:")

try:
    from language_detection import LanguageDetector
    print("   ✓ language_detection imported")
except Exception as e:
    print(f"   ✗ language_detection failed: {e}")

try:
    from translation_module import ReviewTranslator
    print("   ✓ translation_module imported")
except Exception as e:
    print(f"   ✗ translation_module failed: {e}")

try:
    from data_preprocessing import TextPreprocessor
    print("   ✓ data_preprocessing imported")
except Exception as e:
    print(f"   ✗ data_preprocessing failed: {e}")

try:
    from feature_extraction import FeatureExtractor
    print("   ✓ feature_extraction imported")
except Exception as e:
    print(f"   ✗ feature_extraction failed: {e}")

try:
    from model_training import ModelTrainer
    print("   ✓ model_training imported")
except Exception as e:
    print(f"   ✗ model_training failed: {e}")

# Test prediction module
print("\n3. Testing prediction module:")
try:
    from prediction import ReviewAnalyzer
    print("   ✓ prediction.ReviewAnalyzer imported")
except Exception as e:
    print(f"   ✗ prediction.ReviewAnalyzer failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Import diagnostic complete!")
print("="*60)
