import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data, preprocess_data, preprocess_single_input
from model_training import train_and_evaluate_models
from evaluation import get_all_models_metrics, find_best_model, plot_model_comparison, plot_roc_curve, plot_confusion_matrix, plot_feature_importance
from download_data import download_german_credit_data

st.set_page_config(page_title="Credit Scoring System", page_icon="💳", layout="wide")

# Apply some custom CSS for a modern look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- CACHED FUNCTIONS -----------------
@st.cache_resource
def fetch_and_prepare_data(filepath):
    if not os.path.exists(filepath):
        with st.spinner("Dataset not found locally. Downloading from OpenML..."):
            download_german_credit_data()
        if not os.path.exists(filepath):
            return None, None, None, None, None, None
            
    raw_df = load_data(filepath)
    X, y, target_col, feature_names, preprocessors = preprocess_data(raw_df)
    return raw_df, X, y, target_col, feature_names, preprocessors

@st.cache_resource
def train_and_evaluate(_X, _y):
    X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.2, random_state=42)
    trained_models, predictions, probabilities, training_times = train_and_evaluate_models(X_train, y_train, X_test)
    metrics_df = get_all_models_metrics(y_test, predictions, probabilities)
    best_model_name, best_metrics = find_best_model(metrics_df, primary_metric="ROC-AUC")
    return trained_models, metrics_df, best_model_name, best_metrics, y_test, predictions, probabilities

# ----------------- APP LOGIC -----------------

st.title("💳 Credit Scoring System")
st.markdown("A Machine Learning Web Application to predict creditworthiness using financial data.")

data_path = "data/german_credit_data.csv"
raw_df, X, y, target_col, feature_names, preprocessors = fetch_and_prepare_data(data_path)

if raw_df is None:
    st.error(f"Dataset not found at `{data_path}`. Please ensure the data download step was successful.")
    st.stop()

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Dashboard & Model Performance", "Make a Prediction"])

if page == "Dashboard & Model Performance":
    st.header("📊 Model Dashboard")
    
    with st.spinner("Training Models and Evaluating..."):
        trained_models, metrics_df, best_model_name, best_metrics, y_test, predictions, probabilities = train_and_evaluate(X, y)
        
    st.success(f"Models trained successfully! Best Model: **{best_model_name}**")
    
    # Overview Metrics
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Best Model Accuracy</div><div class='metric-value'>{best_metrics['Accuracy']:.2%}</div></div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>ROC-AUC Score</div><div class='metric-value'>{best_metrics['ROC-AUC']:.2%}</div></div>", unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Dataset Size</div><div class='metric-value'>{len(raw_df)}</div></div>", unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Features</div><div class='metric-value'>{len(X.columns)}</div></div>", unsafe_allow_html=True)
        
    st.subheader("Model Performance Comparison")
    st.write("Compare the performance of multiple classification models:")
    fig_comp = plot_model_comparison(metrics_df)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve")
        fig_roc = plot_roc_curve(y_test, probabilities)
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with col2:
        st.subheader(f"Confusion Matrix ({best_model_name})")
        fig_cm = plot_confusion_matrix(y_test, predictions[best_model_name], best_model_name)
        st.pyplot(fig_cm)
        
    st.subheader(f"Feature Importance ({best_model_name})")
    best_model = trained_models[best_model_name]
    fig_imp = plot_feature_importance(best_model, feature_names, top_n=10)
    if fig_imp:
        st.plotly_chart(fig_imp, use_container_width=True)
        st.info("💡 **Explanation**: Features at the top have the most significant impact on the model's decision to approve or reject credit. Understanding these can help explain financial decisions to applicants.")
    else:
        st.write("Feature importance not supported for this model.")

elif page == "Make a Prediction":
    st.header("🔮 Credit Risk Prediction")
    st.write("Enter the applicant's financial details below to predict their creditworthiness. The system automatically selects the best performing model.")
    
    # We must ensure models are trained
    trained_models, metrics_df, best_model_name, best_metrics, *_ = train_and_evaluate(X, y)
    best_model = trained_models[best_model_name]
    
    raw_features = raw_df.drop(columns=[target_col])
    
    st.markdown("### Applicant Details Form")
    with st.form("prediction_form"):
        input_data = {}
        cols = st.columns(3)
        col_idx = 0
        
        for feature in raw_features.columns:
            col = cols[col_idx % 3]
            dtype = raw_features[feature].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                # Suggest median as default
                default_val = float(raw_features[feature].median())
                input_data[feature] = col.number_input(f"{feature}", value=default_val)
            else:
                # Select box for categorical
                options = raw_features[feature].dropna().unique().tolist()
                input_data[feature] = col.selectbox(f"{feature}", options=options)
            
            col_idx += 1
            
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("Predict Credit Score 🚀", use_container_width=True)
        
    if submit_button:
        with st.spinner("Analyzing applicant profile..."):
            # Preprocess the input
            try:
                processed_input = preprocess_single_input(input_data, preprocessors)
                
                # Ensure feature order matches the model training exactly
                missing_cols = set(feature_names) - set(processed_input.columns)
                for c in missing_cols:
                    processed_input[c] = 0
                processed_input = processed_input[feature_names]
                
                # Predict
                prediction = best_model.predict(processed_input)[0]
                
                if hasattr(best_model, "predict_proba"):
                    probs = best_model.predict_proba(processed_input)[0]
                    confidence = np.max(probs) * 100
                else:
                    confidence = 100.0
                
                # Get the string representation of the prediction
                labels = preprocessors.get('target_classes', ["Bad", "Good"])
                # Usually: 0 -> Bad, 1 -> Good. But let's check what LabelEncoder produced.
                result_str = str(labels[prediction]) if len(labels) > prediction else str(prediction)
                
                # Map to standard Good/Bad if possible
                is_good = str(result_str).lower() == 'good' or str(result_str) == '1'
                
                st.markdown("---")
                st.subheader("Prediction Result")
                
                res_col1, res_col2 = st.columns(2)
                
                if is_good:
                    res_col1.success(f"### 🎉 Approved: {result_str} Credit Risk")
                    res_col2.metric("Model Confidence", f"{confidence:.1f}%")
                    st.balloons()
                else:
                    res_col1.error(f"### ⚠️ Rejected: {result_str} Credit Risk")
                    res_col2.metric("Model Confidence", f"{confidence:.1f}%")
                    
                st.info(f"**Explanation:** The applicant was evaluated using the **{best_model_name}** model, which currently has an accuracy of {best_metrics['Accuracy']:.1%} on our validation data.")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
