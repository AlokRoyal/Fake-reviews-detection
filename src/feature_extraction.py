from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging

class FeatureExtractor:
    def __init__(self, max_features=3000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True,
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english'
        )
    
    def fit(self, texts):
        """Fit TF-IDF vectorizer on texts"""
        self.vectorizer.fit(texts)
        return self
    
    def transform(self, texts):
        """Transform texts to TF-IDF features"""
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform in one step"""
        return self.vectorizer.fit_transform(texts)
    
    def save(self, filepath):
        """Save fitted vectorizer"""
        joblib.dump(self.vectorizer, filepath)
        logging.info(f"Vectorizer saved to {filepath}")
    
    def load(self, filepath):
        """Load fitted vectorizer"""
        self.vectorizer = joblib.load(filepath)
        logging.info(f"Vectorizer loaded from {filepath}")
        return self
    
    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()
