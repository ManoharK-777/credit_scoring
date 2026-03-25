from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time
import joblib
import os

def get_models():
    """
    Returns a dictionary of models to train with optimized parameters and n_jobs=-1 where applicable.
    """
    return {
        "Logistic Regression": LogisticRegression(n_jobs=-1, max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
    }

def train_and_evaluate_models(X_train, y_train, X_test):
    """
    Trains multiple models, measures execution time, and returns trained models and their predictions.
    Avoids unnecessary loops where possible.
    """
    models = get_models()
    trained_models = {}
    predictions = {}
    probabilities = {}
    training_times = {}
    
    for name, model in models.items():
        start_time = time.time()
        
        # Train model
        model.fit(X_train, y_train)
        
        # execution time
        end_time = time.time()
        training_times[name] = end_time - start_time
        
        # Predict
        preds = model.predict(X_test)
        
        # Predict probabilities (for ROC)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            # Fallback for models without predict_proba
            probs = preds
            
        trained_models[name] = model
        predictions[name] = preds
        probabilities[name] = probs
        
    return trained_models, predictions, probabilities, training_times

def save_model(model, filepath):
    """
    Save the trained model to disk for later use.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load a trained model from disk.
    """
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None
