from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewTranslator:
    def __init__(self):
        # Cache for loaded translators (avoid reloading)
        self.translators = {}
        
        # Language pair mapping to MarianMT models
        # Format: 'source_target': 'model_name'
        self.supported_pairs = {
            # Hindi to English
            'hi_en': 'Helsinki-NLP/opus-mt-hi-en',
            
            # Spanish to English
            'es_en': 'Helsinki-NLP/opus-mt-es-en',
            
            # French to English
            'fr_en': 'Helsinki-NLP/opus-mt-fr-en',
            
            # German to English
            'de_en': 'Helsinki-NLP/opus-mt-de-en',
            
            # Japanese to English
            'ja_en': 'Helsinki-NLP/opus-mt-ja-en',
            
            # Portuguese to English
            'pt_en': 'Helsinki-NLP/opus-mt-pt-en',
            
            # Russian to English
            'ru_en': 'Helsinki-NLP/opus-mt-ru-en',
            
            # Arabic to English
            'ar_en': 'Helsinki-NLP/opus-mt-ar-en',
            
            # Italian to English
            'it_en': 'Helsinki-NLP/opus-mt-it-en',
            
            # Turkish to English
            'tr_en': 'Helsinki-NLP/opus-mt-tr-en',
            
            # Korean to English
            'ko_en': 'Helsinki-NLP/opus-mt-ko-en',
            
            # Polish to English
            'pl_en': 'Helsinki-NLP/opus-mt-pl-en',
            
            # Dutch to English
            'nl_en': 'Helsinki-NLP/opus-mt-nl-en',
        }
    
    def get_translator(self, lang_code):
        """Load and cache translator for specific language"""
        
        pair_key = f"{lang_code}_en"
        
        # Return cached translator if available
        if pair_key in self.translators:
            return self.translators[pair_key]
        
        # Check if language pair is supported
        if pair_key not in self.supported_pairs:
            logger.warning(f"No translator available for {lang_code}->en")
            return None
        
        try:
            model_name = self.supported_pairs[pair_key]
            logger.info(f"Loading translator for {lang_code}->en: {model_name}")
            
            # Load translator pipeline
            translator = pipeline(
                "translation",
                model=model_name,
                device=0 if _cuda_available() else -1  # Use GPU if available
            )
            
            # Cache it
            self.translators[pair_key] = translator
            logger.info(f"✓ Loaded translator for {lang_code}->en")
            
            return translator
            
        except Exception as e:
            logger.error(f"Failed to load translator for {lang_code}: {e}")
            return None
    
    def translate_to_english(self, text, source_lang):
        """Translate review to English"""
        
        # If already English, return as-is
        if source_lang == 'en':
            return text
        
        # Get translator
        translator = self.get_translator(source_lang)
        
        if not translator:
            logger.warning(f"No translator for {source_lang}, returning original text")
            return text
        
        try:
            # Limit text length to avoid token limit issues
            max_length = min(len(text.split()) + 50, 512)
            
            # Translate
            result = translator(text, max_length=max_length)
            
            if result and len(result) > 0:
                translated = result['translation_text']
                logger.info(f"✓ Translated {source_lang} text ({len(text)} chars → {len(translated)} chars)")
                return translated
            else:
                logger.warning("Translation returned empty result")
                return text
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            logger.warning("Returning original text")
            return text


def _cuda_available():
    """Check if CUDA (GPU) is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False
