import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path FIRST
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Now import your modules
from prediction import ReviewAnalyzer

# Page configuration
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_analyzer():
    """Load the model analyzer"""
    model_path = Path("models/random_forest_model.pkl")
    vectorizer_path = Path("models/tfidf_vectorizer.pkl")
    
    if not model_path.exists() or not vectorizer_path.exists():
        st.error("❌ Model files not found!")
        st.error(f"Looking for:")
        st.error(f"  - {model_path}")
        st.error(f"  - {vectorizer_path}")
        st.error("\nPlease run 'python train_model.py' first")
        st.stop()
    
    try:
        return ReviewAnalyzer(str(model_path), str(vectorizer_path))
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
col1, col2 = st.columns([1, 4])
with col1:
    st.write("🔍")
with col2:
    st.title("Fake Product Review Detection System")
    st.markdown("*AI-Powered Review Authenticity Detector with Multilingual Support*")

st.divider()

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select a page:",
        ["🔍 Single Review Analysis", "📊 Batch Analysis", "📈 Statistics & Insights", "ℹ️ About"]
    )

# Load analyzer
analyzer = load_analyzer()

# PAGE 1: Single Review Analysis
if page == "🔍 Single Review Analysis":
    st.header("Single Review Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Review")
        review_text = st.text_area(
            "Enter a product review to analyze:",
            height=150,
            placeholder="Paste a review here... (supports English, Hindi, Spanish, French, etc.)"
        )
        
        if st.button("🔍 Analyze Review", use_container_width=True):
            if review_text.strip():
                with st.spinner("Analyzing review..."):
                    try:
                        result = analyzer.analyze_review(review_text)
                        
                        if 'error' in result:
                            st.error(f"❌ Analysis Error: {result['error']}")
                        else:
                            st.session_state.history.append(result)
                            
                            with col2:
                                st.subheader("Analysis Results")
                                
                                # Prediction badge
                                pred_color = "🔴" if result['prediction'] == 'FAKE' else "🟢"
                                st.markdown(f"### {pred_color} {result['prediction']}")
                                
                                # Confidence metric
                                confidence = result['confidence']
                                st.metric("Confidence", f"{confidence:.2%}")
                                
                                # Details in expander
                                with st.expander("📋 Detailed Information"):
                                    st.write(f"**Original Language:** {result['detected_language']}")
                                    st.write(f"**Translated Review:** {result['translated_review']}")
                                    st.write(f"**Preprocessed Text:** {result['preprocessed_review'][:200]}...")
                                
                                # Confidence visualization
                                fig = go.Figure(data=[
                                    go.Bar(
                                        y=['Confidence'],
                                        x=[confidence],
                                        orientation='h',
                                        marker=dict(
                                            color='#ef553b' if result['prediction'] == 'FAKE' else '#00cc96'
                                        )
                                    )
                                ])
                                fig.update_layout(
                                    xaxis=dict(range=[0, 1]),
                                    height=100,
                                    showlegend=False,
                                    margin=dict(l=0, r=0, t=0, b=0)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.success("✓ Analysis complete!")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
            else:
                st.warning("Please enter a review to analyze.")

# PAGE 2: Batch Analysis
elif page == "📊 Batch Analysis":
    st.header("Batch Review Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file with reviews (must have 'review' or 'text' column):",
        type=['csv']
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Try to find review column
        review_col = None
        for col in ['review', 'text', 'text_', 'reviewText']:
            if col in df.columns:
                review_col = col
                break
        
        if review_col is None:
            st.error(f"CSV must contain a review column. Found: {df.columns.tolist()}")
        else:
            st.info(f"Processing {len(df)} reviews...")
            
            progress_bar = st.progress(0)
            results = []
            
            for idx, row in df.iterrows():
                result = analyzer.analyze_review(row[review_col])
                results.append(result)
                progress_bar.progress((idx + 1) / len(df))
            
            results_df = pd.DataFrame(results)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                fake_count = (results_df['prediction'] == 'FAKE').sum()
                st.metric("Fake Reviews", fake_count)
            with col2:
                genuine_count = (results_df['prediction'] == 'GENUINE').sum()
                st.metric("Genuine Reviews", genuine_count)
            with col3:
                avg_conf = results_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.2%}")
            with col4:
                fake_pct = (fake_count / len(results_df)) * 100
                st.metric("Fake Percentage", f"{fake_pct:.1f}%")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                pred_counts = results_df['prediction'].value_counts()
                fig = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title="Review Distribution",
                    color_discrete_map={'FAKE': '#ef553b', 'GENUINE': '#00cc96'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(
                    results_df,
                    x='confidence',
                    nbins=20,
                    title="Confidence Distribution",
                    color_discrete_sequence=['#636EFA']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("Detailed Results")
            display_df = results_df[['original_review', 'detected_language', 'prediction', 'confidence']].copy()
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
            st.dataframe(display_df, use_container_width=True)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="analysis_results.csv",
                mime="text/csv"
            )

# PAGE 3: Statistics & Insights
elif page == "📈 Statistics & Insights":
    st.header("Analysis Statistics & Insights")
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Reviews Analyzed", len(history_df))
        with col2:
            fake_pct = (history_df['prediction'] == 'FAKE').sum() / len(history_df) * 100
            st.metric("Fake Reviews %", f"{fake_pct:.1f}%")
        with col3:
            st.metric("Avg Confidence", f"{history_df['confidence'].mean():.2%}")
        
        # Language distribution
        lang_counts = history_df['detected_language'].value_counts()
        fig = px.bar(
            x=lang_counts.index,
            y=lang_counts.values,
            title="Reviews by Language",
            labels={'x': 'Language', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction distribution
        pred_counts = history_df['prediction'].value_counts()
        fig = px.pie(
            values=pred_counts.values,
            names=pred_counts.index,
            title="Prediction Distribution",
            color_discrete_map={'FAKE': '#ef553b', 'GENUINE': '#00cc96'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No analysis history yet. Start analyzing reviews to see statistics.")

# PAGE 4: About
elif page == "ℹ️ About":
    st.header("About This System")
    
    st.markdown("""
    ### Project Overview
    
    This system leverages **Machine Learning** and **Natural Language Processing** to detect fake 
    product reviews from e-commerce platforms. It features:
    
    - **Multilingual Support**: Analyzes reviews in 10+ languages
    - **Automatic Translation**: Converts non-English reviews to English using MarianMT
    - **Advanced NLP**: Uses TF-IDF vectorization with multiple ML models
    - **High Accuracy**: Trained on 40,000+ reviews for robust detection
    
    ### Supported Languages
    
    - English (en)
    - [translate:हिंदी] (hi)
    - [translate:Español] (es)
    - [translate:Français] (fr)
    - [translate:Deutsch] (de)
    - [translate:日本語] (ja)
    - [translate:中文] (zh-cn)
    - [translate:Português] (pt)
    - [translate:Русский] (ru)
    - [translate:العربية] (ar)
    
    ### ML Models Used
    
    - **Random Forest**: Ensemble method for robust classification
    - **Support Vector Machine (SVM)**: Effective for text classification
    - **Naive Bayes**: Probabilistic classifier
    - **Logistic Regression**: Linear classification baseline
    
    ### Technical Stack
    
    - **Framework**: Streamlit for web interface
    - **ML/NLP**: scikit-learn, NLTK, Hugging Face Transformers
    - **Translation**: MarianMT (Helsinki-NLP)
    - **Language Detection**: langdetect
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly
    
    ### How It Works
    
    1. **Language Detection**: Identifies the language of the review
    2. **Translation**: Converts non-English reviews to English
    3. **Preprocessing**: Cleans text, removes stopwords, lemmatizes
    4. **Feature Extraction**: Converts text to TF-IDF vectors
    5. **Classification**: Uses trained ML model to predict (Fake/Genuine)
    6. **Confidence**: Shows probability score for the prediction
    """)

st.divider()
st.markdown("*Fake Review Detection System v1.0 | Last Updated: 2025*")
