from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import logging

class ModelTrainer:
    def __init__(self, model_name='random_forest'):
        self.model_name = model_name
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified model"""
        models = {
            'random_forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
            'svm': SVC(kernel='linear', probability=True, random_state=42),
            'naive_bayes': MultinomialNB(alpha=1.0),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        return models.get(self.model_name, models['random_forest'])
    
    def train(self, X_train, y_train):
        """Train the model"""
        logging.info(f"Training {self.model_name} model...")
        self.model.fit(X_train, y_train)
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        return metrics, y_pred, confusion_matrix(y_test, y_pred)
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1_weighted')
        return scores
    
    def save(self, filepath):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        logging.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        logging.info(f"Model loaded from {filepath}")
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            logging.warning(f"{self.model_name} does not support probability predictions")
            return None
