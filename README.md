# Fake Review Detection

![Python](https://img.shields.io/badge/Python-ML-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

A Streamlit application for analyzing product reviews and classifying them as FAKE or GENUINE. It supports individual reviews and CSV batches, shows confidence and language details, and provides charts and an in-session history view.

## Analysis flow

1. The app accepts review text directly or from a CSV file.
2. The prediction modules detect language and prepare the text.
3. Translation support handles non-English input.
4. Text is transformed into TF-IDF features and passed to the configured classifier.
5. The app presents a label, confidence information, and visual summaries.

The application loads trained model and vectorizer artifacts from the models/ directory.

## Architecture

Streamlit UI → src prediction pipeline → language detection / translation / preprocessing → TF-IDF → trained classifier → result and charts

## Technology

Python, Streamlit, pandas, NumPy, Plotly, scikit-learn, joblib, NLTK, langdetect, Hugging Face Transformers, and PyTorch. Pinned versions are listed in requirements.txt.

## Start locally

Create and activate a virtual environment, install dependencies, then start Streamlit:

    python -m venv .venv
    pip install -r requirements.txt
    streamlit run app.py

Windows PowerShell activation: .venv\Scripts\Activate.ps1. macOS/Linux activation: source .venv/bin/activate.

## Data and model requirements

The app expects models/random_forest_model.pkl and models/tfidf_vectorizer.pkl. These artifacts are not listed in the repository root, so supply compatible trained artifacts before launching the prediction interface. train_model.py is the training entry point and requires the expected training data. CSV batch uploads must include a review-text column named review, text, text_, or reviewText.

## Project layout

- app.py — Streamlit UI and batch workflow
- src/ — prediction, preprocessing, language detection, translation, feature extraction, and model training
- train_model.py — training workflow
- download_dataset.py — dataset helper
- requirements.txt — dependency pins
- project_guide.md — extended implementation notes

## Limitations

Model predictions are estimates, not proof of fraudulent behavior. Evaluate the classifier with representative labeled data and review false positives before relying on results. Model artifacts and datasets are not included in the repository listing.