import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from model_training import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_datasets():
    """Load and combine datasets"""
    
    # Option 1: Load from CSV file (YOUR FORMAT)
    try:
        logger.info("Loading CSV file...")
        df = pd.read_csv('data/raw/fake_reviews.csv')
        
        # Your CSV has: category, rating, text, label
        # We only need: text and label
        
        logger.info(f"  Found columns: {df.columns.tolist()}")
        
        # Rename 'text' to 'review' to match our code
        df = df.rename(columns={'text': 'review'})
        
        # Keep only review and label columns
        df = df[['review', 'label']].dropna()
        
        logger.info(f"✓ Loaded {len(df)} reviews from CSV")
        logger.info(f"  Fake: {(df['label'] == 1).sum()}, Genuine: {(df['label'] == 0).sum()}")
        
        return df
        
    except FileNotFoundError:
        logger.warning("CSV file not found. Trying Hugging Face...")
    except KeyError as e:
        logger.error(f"Column not found: {e}")
        logger.error("Expected columns: text, label")
        raise
    except Exception as e:
        logger.warning(f"Error loading CSV: {e}")
    
    # Option 2: Load from Hugging Face
    try:
        logger.info("Loading from Hugging Face...")
        from datasets import load_dataset
        dataset = load_dataset("theArijitDas/Fake-Reviews-Dataset", split='train')
        df = pd.DataFrame(dataset)
        
        # Rename if needed
        if 'text' in df.columns:
            df = df.rename(columns={'text': 'review'})
        
        df = df[['review', 'label']].dropna()
        
        if len(df) > 100000:
            df = df.sample(n=100000, random_state=42)
        
        logger.info(f"✓ Loaded {len(df)} reviews from Hugging Face")
        return df
        
    except Exception as e:
        logger.warning(f"Hugging Face load failed: {e}")
    
    # Option 3: Create sample dataset
    logger.warning("Creating sample dataset for testing...")
    return create_sample_dataset()


def create_sample_dataset():
    """Create a small sample dataset for testing"""
    
    fake_reviews = [
        "This product is amazing! Best purchase ever! 5 stars!",
        "Excellent quality! Highly recommend to everyone!",
        "Love it! Perfect! Great! Wonderful! Fantastic!",
        "Best thing I ever bought! You must buy this!",
        "Amazing product! Changed my life! Buy it now!",
        "Absolutely incredible! Worth every penny!",
        "Best value for money! Highly satisfied!",
        "Can't believe how good this is! Recommend!",
        "Outstanding quality! Very impressed!",
        "Perfect product! Exactly what I wanted!",
    ] * 200
    
    genuine_reviews = [
        "The product arrived on time and works as expected. Good value for money.",
        "Decent quality for the price. Some minor issues but overall satisfied.",
        "It's okay. Does what it's supposed to do. Nothing special.",
        "Good product but the packaging was damaged during shipping.",
        "Works fine but I wish it had more features. Still recommend.",
        "Average product. Not great, not terrible. Fair price.",
        "Arrived quickly. Quality is acceptable for the price point.",
        "Does the job. Would have preferred better instructions.",
        "Okay product. Some features don't work as advertised.",
        "Decent but there are better alternatives available.",
    ] * 200
    
    df = pd.DataFrame({
        'review': fake_reviews + genuine_reviews,
        'label': [1] * len(fake_reviews) + [0] * len(genuine_reviews)

    })
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"✓ Created sample dataset with {len(df)} reviews")
    return df


def train_pipeline():
    """Complete training pipeline"""
    
    logger.info("="*50)
    logger.info("FAKE REVIEW DETECTION - MODEL TRAINING")
    logger.info("="*50)
    
    # Step 1: Load datasets
    logger.info("\n[1/6] Loading datasets...")
    df = load_datasets()
    
    logger.info(f"\nDataset Summary:")
    logger.info(f"  Total reviews: {len(df)}")
    logger.info(f"  Fake reviews: {(df['label'] == 1).sum()} ({(df['label'] == 1).sum()/len(df)*100:.1f}%)")
    logger.info(f"  Genuine reviews: {(df['label'] == 0).sum()} ({(df['label'] == 0).sum()/len(df)*100:.1f}%)")
    
    # Step 2: Preprocess
    logger.info("\n[2/6] Preprocessing text...")
    preprocessor = TextPreprocessor()
    df['processed_review'] = df['review'].apply(preprocessor.preprocess)
    
    # Remove empty reviews after preprocessing
    df = df[df['processed_review'].str.len() > 0].reset_index(drop=True)
    logger.info(f"  Retained {len(df)} reviews after preprocessing")
    
    # Step 3: Feature extraction
    logger.info("\n[3/6] Extracting TF-IDF features...")
    feature_extractor = FeatureExtractor(max_features=3000, ngram_range=(1, 2))
    X = feature_extractor.fit_transform(df['processed_review'])
    y = df['label'].values
    
    logger.info(f"  Feature matrix shape: {X.shape}")
    
    # Save vectorizer
    Path("models").mkdir(exist_ok=True)
    feature_extractor.save('models/tfidf_vectorizer.pkl')
    logger.info("  ✓ Saved TF-IDF vectorizer")
    
    # Step 4: Train-test split
    logger.info("\n[4/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"  Training set: {X_train.shape} samples")
    logger.info(f"  Test set: {X_test.shape} samples")
    
    # Step 5: Train models
    logger.info("\n[5/6] Training machine learning models...")
    models = ['random_forest', 'svm', 'naive_bayes', 'logistic_regression']
    results = {}
    
    for model_name in models:
        logger.info(f"\n  Training {model_name}...")
        trainer = ModelTrainer(model_name)
        trainer.train(X_train, y_train)
        
        # Evaluate
        metrics, y_pred, conf_matrix = trainer.evaluate(X_test, y_test)
        results[model_name] = metrics
        
        # Save model
        trainer.save(f'models/{model_name}_model.pkl')
        
        # Print metrics
        logger.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"    Precision: {metrics['precision']:.4f}")
        logger.info(f"    Recall:    {metrics['recall']:.4f}")
        logger.info(f"    F1-Score:  {metrics['f1']:.4f}")
    
    # Step 6: Summary
    logger.info("\n[6/6] Training Summary")
    logger.info("="*50)
    
    best_model = max(results, key=lambda x: results[x]['f1'])
    logger.info(f"\n🏆 Best Model: {best_model}")
    logger.info(f"   F1-Score: {results[best_model]['f1']:.4f}")
    logger.info(f"   Accuracy: {results[best_model]['accuracy']:.4f}")
    
    logger.info("\n✓ All models trained and saved successfully!")
    logger.info("\nSaved files:")
    logger.info("  - models/tfidf_vectorizer.pkl")
    for model_name in models:
        logger.info(f"  - models/{model_name}_model.pkl")
    
    logger.info("\n" + "="*50)
    logger.info("Next step: Run 'streamlit run app.py'")
    logger.info("="*50)
    
    return results


if __name__ == "__main__":
    try:
        results = train_pipeline()
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
