# Fake Review Detection

A Streamlit application for analyzing product reviews as fake or genuine. It supports single-review and CSV batch workflows, keeps an in-session analysis history, and visualizes predictions and confidence scores. The prediction pipeline includes language detection, optional translation, text preprocessing, TF-IDF features, and a scikit-learn classifier.

## Stack

- Python
- Streamlit for the web interface
- pandas, NumPy, and Plotly for data handling and charts
- scikit-learn and joblib for feature extraction and model loading
- NLTK for text preprocessing
- langdetect for language detection
- Hugging Face Transformers and PyTorch for translation

Install the pinned dependencies with:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app expects trained artifacts at `models/random_forest_model.pkl` and `models/tfidf_vectorizer.pkl`. These artifacts and the training dataset are not included in the repository; provide the required model files before launching the analyzer. `train_model.py` contains the training entry point, and `download_dataset.py` is available for dataset setup.

## Project layout

- `app.py` — Streamlit interface for single and batch analysis
- `src/` — prediction, preprocessing, language detection, translation, model, and feature extraction modules
- `train_model.py` — model training workflow
- `requirements.txt` — Python dependencies
- `project_guide.md` — extended implementation notes

CSV batch uploads must include a review text column (`review`, `text`, `text_`, or `reviewText`).

## Important

Predictions are model estimates, not proof that a review is fraudulent. Validate model behavior on representative data before relying on it.
