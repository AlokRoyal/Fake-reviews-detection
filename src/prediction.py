import logging
import sys
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path for imports
src_path = str(Path(__file__).parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import required modules
try:
    from language_detection import LanguageDetector
    logger.info("✓ Imported LanguageDetector")
except ImportError as e:
    logger.error(f"Failed to import LanguageDetector: {e}")
    raise

try:
    from translation_module import ReviewTranslator
    logger.info("✓ Imported ReviewTranslator")
except ImportError as e:
    logger.error(f"Failed to import ReviewTranslator: {e}")
    raise

try:
    from data_preprocessing import TextPreprocessor
    logger.info("✓ Imported TextPreprocessor")
except ImportError as e:
    logger.error(f"Failed to import TextPreprocessor: {e}")
    raise

try:
    from feature_extraction import FeatureExtractor
    logger.info("✓ Imported FeatureExtractor")
except ImportError as e:
    logger.error(f"Failed to import FeatureExtractor: {e}")
    raise


class ReviewAnalyzer:
    """
    Complete review analysis pipeline:
    1. Language Detection
    2. Translation to English
    3. Text Preprocessing
    4. Feature Extraction (TF-IDF)
    5. ML Model Prediction
    """
    
    def __init__(self, model_path, vectorizer_path):
        """Initialize the ReviewAnalyzer with model and vectorizer"""
        try:
            logger.info("="*60)
            logger.info("Initializing ReviewAnalyzer...")
            logger.info("="*60)
            
            # Initialize language components
            self.lang_detector = LanguageDetector()
            logger.info("✓ LanguageDetector initialized")
            
            self.translator = ReviewTranslator()
            logger.info("✓ ReviewTranslator initialized")
            
            # Initialize text processing
            self.preprocessor = TextPreprocessor()
            logger.info("✓ TextPreprocessor initialized")
            
            self.feature_extractor = FeatureExtractor()
            logger.info("✓ FeatureExtractor initialized")
            
            # Load the vectorizer
            self.feature_extractor.load(vectorizer_path)
            logger.info(f"✓ TF-IDF Vectorizer loaded from {vectorizer_path}")
            
            # Load trained model
            import joblib
            self.model = joblib.load(model_path)
            logger.info(f"✓ ML Model loaded from {model_path}")
            
            logger.info("="*60)
            logger.info("✓ ReviewAnalyzer fully initialized!")
            logger.info("="*60)
        
        except Exception as e:
            logger.error(f"Error initializing ReviewAnalyzer: {e}", exc_info=True)
            raise
    
    def analyze_review(self, review_text):
        """
        Complete pipeline for review analysis
        
        Args:
            review_text (str): The review to analyze
        
        Returns:
            dict: Analysis result with prediction and confidence
        """
        try:
            # Validate input
            if not review_text or not isinstance(review_text, str):
                return {'error': 'Invalid review text'}
            
            # ===== STEP 1: LANGUAGE DETECTION =====
            logger.info("\n[Step 1/5] Detecting language...")
            try:
                detected_lang = self.lang_detector.detect_language(review_text)
                logger.info(f"Detected language: {detected_lang}")
            except Exception as e:
                logger.error(f"Language detection failed: {e}")
                detected_lang = 'en'
                logger.warning("Defaulting to English")
            
            # ===== STEP 2: TRANSLATION =====
            logger.info("[Step 2/5] Translating if needed...")
            try:
                if detected_lang == 'en':
                    english_review = review_text
                    logger.info("Already in English, skipping translation")
                else:
                    logger.info(f"Translating from {detected_lang} to English...")
                    english_review = self.translator.translate_to_english(review_text, detected_lang)
                    logger.info(f"Translation complete ({len(english_review)} chars)")
            except Exception as e:
                logger.warning(f"Translation failed: {e}, using original text")
                english_review = review_text
            
            # ===== STEP 3: TEXT PREPROCESSING =====
            logger.info("[Step 3/5] Preprocessing text...")
            try:
                processed_review = self.preprocessor.preprocess(english_review)
                
                if not processed_review or len(processed_review.strip()) == 0:
                    return {'error': 'Review is empty after preprocessing'}
                
                logger.info(f"Preprocessing complete ({len(processed_review)} chars)")
            except Exception as e:
                logger.error(f"Preprocessing failed: {e}")
                return {'error': f'Preprocessing failed: {e}'}
            
            # ===== STEP 4: FEATURE EXTRACTION =====
            logger.info("[Step 4/5] Extracting TF-IDF features...")
            try:
                features = self.feature_extractor.transform([processed_review])
                logger.info(f"Feature extraction complete (shape: {features.shape})")
            except Exception as e:
                logger.error(f"Feature extraction failed: {e}")
                return {'error': f'Feature extraction failed: {e}'}
            
            # ===== STEP 5: PREDICTION =====
            logger.info("[Step 5/5] Making prediction...")
            try:
                # Get prediction
                prediction = int(self.model.predict(features))
                logger.info(f"Raw prediction: {prediction}")
                
                # Get confidence scores
                try:
                    if hasattr(self.model, 'predict_proba'):
                        proba = self.model.predict_proba(features)
                        confidence = float(np.max(proba))
                        probability = float(proba[prediction])
                    else:
                        logger.warning("Model doesn't support predict_proba, using default confidence")
                        confidence = 0.5
                        probability = 0.5
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}, using default")
                    confidence = 0.5
                    probability = 0.5
                
                logger.info(f"Prediction: {prediction} (Fake=1, Genuine=0)")
                logger.info(f"Confidence: {confidence:.2%}")
                
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return {'error': f'Prediction failed: {e}'}
            
            # ===== FORMAT RESULT =====
            result = {
                'original_review': review_text,
                'detected_language': detected_lang,
                'translated_review': english_review if detected_lang != 'en' else review_text,
                'preprocessed_review': processed_review,
                'prediction': 'FAKE' if prediction == 1 else 'GENUINE',
                'confidence': float(confidence),
                'probability': float(probability),
                'label': int(prediction)
            }
            
            logger.info(f"\n✓ ANALYSIS COMPLETE: {result['prediction']} ({confidence:.2%})\n")
            return result
        
        except Exception as e:
            logger.error(f"\n✗ Analysis pipeline failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'original_review': review_text
            }
