from langdetect import detect, detect_langs, DetectorFactory, LangDetectException
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed for consistent results
DetectorFactory.seed = 42

class LanguageDetector:
    def __init__(self):
        # Supported language codes for translation
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'ja': 'Japanese',
            'zh': 'Chinese',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ar': 'Arabic',
            'it': 'Italian',
            'tr': 'Turkish',
            'ko': 'Korean',
            'pl': 'Polish',
            'nl': 'Dutch',
        }
        
        # langdetect code mapping (some differences)
        self.langdetect_map = {
            'zh-cn': 'zh',
            'zh-tw': 'zh',
        }
    
    def detect_language(self, text):
        """Detect the language of the input text with fallback"""
        try:
            # Skip detection for very short text
            if not text or len(text.strip()) < 5:
                logger.warning("Text too short for reliable detection, defaulting to English")
                return 'en'
            
            # Detect language
            lang = detect(text)
            
            # Map code if needed
            if lang in self.langdetect_map:
                lang = self.langdetect_map[lang]
            
            # Check if language is supported for translation
            if lang in self.supported_languages:
                logger.info(f"Detected language: {lang} ({self.supported_languages[lang]})")
                return lang
            else:
                logger.warning(f"Detected language {lang} not in supported list, defaulting to English")
                return 'en'
                
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}, defaulting to English")
            return 'en'
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {e}")
            return 'en'
    
    def get_language_probabilities(self, text):
        """Get probability scores for top languages"""
        try:
            if not text or len(text.strip()) < 5:
                return []
            
            return detect_langs(text)
        except Exception as e:
            logger.error(f"Error getting language probabilities: {e}")
            return []
    
    def get_language_name(self, lang_code):
        """Get readable language name"""
        return self.supported_languages.get(lang_code, 'Unknown')
