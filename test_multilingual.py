import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from language_detection import LanguageDetector
from translation_module import ReviewTranslator

# Test language detection
detector = LanguageDetector()

test_texts = {
    'en': "This product is amazing!",
    'hi': "यह उत्पाद अद्भुत है!",
    'es': "¡Este producto es increíble!",
    'fr': "Ce produit est incroyable!",
    'de': "Dieses Produkt ist fantastisch!",
    'ja': "この製品は素晴らしいです！",
    'ar': "هذا المنتج رائع جدا!",
}

print("=" * 60)
print("LANGUAGE DETECTION TEST")
print("=" * 60)

for expected_lang, text in test_texts.items():
    detected = detector.detect_language(text)
    status = "✓" if detected == expected_lang else "✗"
    print(f"{status} {expected_lang.upper()}: {text[:40]}... → Detected: {detected}")

print("\n" + "=" * 60)
print("TRANSLATION TEST")
print("=" * 60)

translator = ReviewTranslator()

for lang_code, text in test_texts.items():
    if lang_code != 'en':
        try:
            translated = translator.translate_to_english(text, lang_code)
            print(f"\n{lang_code.upper()}: {text}")
            print(f"→ EN: {translated}")
        except Exception as e:
            print(f"\n{lang_code.upper()}: ERROR - {e}")
