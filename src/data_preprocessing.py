import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Ultra-simple text preprocessor that won't delete everything"""
    
    def __init__(self):
        # Basic English stopwords
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'in', 'on', 'at', 'to', 'from', 'of', 'with', 'by', 'for', 'as'
        }
        logger.info("TextPreprocessor initialized")
    
    def clean_text(self, text):
        """Minimal cleaning - just lowercase and remove extra spaces"""
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # KEEP most characters - only remove HTML tags and control characters
        text = re.sub(r'<[^>]+>', '', text)  # HTML tags
        text = re.sub(r'[^\w\s\.\!\?\-\']', '', text)  # Keep word chars, spaces, punctuation
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """Simple word tokenization"""
        # Just split by spaces
        words = text.split()
        return words
    
    def remove_stopwords(self, tokens):
        """Remove common stopwords"""
        filtered = [w for w in tokens if w.lower() not in self.stop_words and len(w) > 1]
        
        # If we removed everything, return at least some tokens
        if not filtered:
            return tokens
        
        return filtered
    
    def lemmatize(self, tokens):
        """Skip lemmatization - it's causing issues"""
        return tokens
    
    def preprocess(self, text):
        """
        Simple preprocessing:
        1. Clean
        2. Lowercase
        3. Remove stopwords
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Clean
        text = self.clean_text(text)
        
        if not text:
            logger.warning("Text became empty after cleaning")
            return ""
        
        # Tokenize
        tokens = self.tokenize(text)
        
        if not tokens:
            logger.warning("No tokens after tokenization")
            return text  # Return original cleaned text
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        if not tokens:
            logger.warning("No tokens after stopword removal, returning cleaned text")
            return text
        
        # Rejoin
        processed = ' '.join(tokens)
        
        logger.info(f"✓ Preprocessed: '{text[:50]}...' → '{processed[:50]}...'")
        
        return processed
    
    def batch_preprocess(self, texts):
        """Preprocess multiple texts"""
        return [self.preprocess(text) for text in texts]
