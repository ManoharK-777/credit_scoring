# CodeAlpha_CreditScoringModel 💳

**Live Application:** [https://creditscoring-7cuvkhntrc4qsdht5kbaeh.streamlit.app/](https://creditscoring-7cuvkhntrc4qsdht5kbaeh.streamlit.app/)

An end-to-end machine learning application to predict creditworthiness using the German Credit Dataset. 

## Features
- **Data Preprocessing Pipeline** (`data_preprocessing.py`) handles missing values, categorical encoding, and numerical scaling.
- **Model Training** (`model_training.py`) trains multiple classifiers (Logistic Regression, Decision Trees, Random Forest, Gradient Boosting).
- **Evaluation Dashboard** (`evaluation.py`) generates Model comparisons, ROC curves, and Feature Importance graphs.
- **Interactive Streamlit UI** (`app.py`) for real-time predictions and dashboard performance visualizations.

## Prerequisites
⚠️ **Important for Windows Users:** You must have a **64-bit version of Python** installed (preferably 3.10 or newer) to easily install data science libraries like `pandas`, `scikit-learn`, and `streamlit`. 32-bit Python will fail to install these packages without a C++ build environment.

## Setup Instructions

1. **Clone the repository or navigate to the project directory:**
   ```bash
   cd credit-scoring-system
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   ```bash
   python download_data.py
   ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

The application will launch in your default web browser, allowing you to explore model performance and make interactive credit score predictions.
