import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculates classification metrics.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

def get_all_models_metrics(y_true, predictions, probabilities):
    """
    Collects metrics for all models into a DataFrame.
    """
    results = []
    for name in predictions.keys():
        metrics = calculate_metrics(y_true, predictions[name], probabilities[name])
        metrics["Model"] = name
        results.append(metrics)
    
    return pd.DataFrame(results).set_index("Model")

def find_best_model(metrics_df, primary_metric="ROC-AUC"):
    """
    Automatically selects the best-performing model based on the primary metric.
    """
    best_model_name = metrics_df[primary_metric].idxmax()
    return best_model_name, metrics_df.loc[best_model_name].to_dict()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Returns a matplotlib figure for the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix: {model_name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, probabilities):
    """
    Returns a plotly figure for the ROC curves of all models.
    """
    fig = go.Figure()
    
    # Add random guessing line
    fig.add_shape(
        type='line', line=dict(dash='dash', color='gray'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    for name, probs in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_score = roc_auc_score(y_true, probs)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc_score:.3f})", mode='lines'))
        
    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    return fig

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Returns a plotly figure showing top N feature importances.
    Supported for Tree-based models and Logistic Regression (using absolute coefficients).
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None
        
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=True).tail(top_n)
    
    fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                 title=f'Top {top_n} Feature Importances', color='Importance', color_continuous_scale='Viridis')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_model_comparison(metrics_df):
    """
    Plots a comparison of models across all metrics using Plotly.
    """
    metrics_df_reset = metrics_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(metrics_df_reset, x="Model", y="Score", color="Metric", barmode="group",
                 title="Model Performance Comparison across all Metrics")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    return fig
