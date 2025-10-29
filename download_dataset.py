from datasets import load_dataset
dataset = load_dataset("theArijitDas/Fake-Reviews-Dataset")
dataset['train'].to_csv('data/raw/fake_reviews.csv', index=False)