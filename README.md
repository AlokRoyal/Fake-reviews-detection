# Fake Review Detection

A Streamlit application for examining product reviews and classifying them as FAKE or GENUINE. It includes single-review analysis, CSV batch analysis, visual summaries, and an in-session history view.

## Analysis flow

The modules in src/ combine language detection and text preprocessing with feature extraction and a trained classifier. Translation support is included for non-English input. The application displays the predicted label, confidence information, and language details; batch results can be downloaded as CSV.

## Technology

Python, Streamlit, pandas, NumPy, Plotly, scikit-learn, joblib, NLTK, langdetect, Hugging Face Transformers, and PyTorch. The pinned versions are in requirements.txt.

Create a virtual environment, activate it, then install dependencies:

    python -m venv .venv
    pip install -r requirements.txt

On Windows PowerShell, activate with .venv\Scripts\Activate.ps1; on macOS/Linux use source .venv/bin/activate.

## Run

    streamlit run app.py

The app expects models/random_forest_model.pkl and models/tfidf_vectorizer.pkl. Those artifacts are not included in the repository listing, so provide trained models before launching the prediction interface. train_model.py is the training entry point and requires the expected training data.

For CSV batch analysis, include a review-text column named review, text, text_, or reviewText.

## Repository map

app.py is the Streamlit interface; src/ contains preprocessing, language detection, translation, feature extraction, training, and prediction modules; train_model.py contains the training workflow; download_dataset.py is a dataset helper; requirements.txt lists dependencies; project_guide.md contains extended notes.

Predictions are model outputs, not proof of fraud. Evaluate on representative labeled data before relying on them.