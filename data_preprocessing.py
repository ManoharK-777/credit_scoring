import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    return pd.read_csv(filepath)

def detect_target_column(df):
    """
    Automatically detects the target column. 
    Usually, it's the last column or named 'class', 'target', 'risk', 'credit', 'credit_risk'.
    """
    potential_targets = ['class', 'target', 'risk', 'credit', 'credit_risk']
    for col in df.columns:
        if col.lower() in potential_targets and df[col].nunique() == 2:
            return col
    
    # Fallback to the last column if it has 2 unique values
    last_col = df.columns[-1]
    if df[last_col].nunique() == 2:
        return last_col
        
    return df.columns[-1]

def preprocess_data(df, target_col=None):
    """
    Preprocesses the dataset: handles missing values, encodes categoricals, and scales numericals.
    Returns:
    - X (preprocessed feature dataframe)
    - y (target series)
    - target_col (the name of the detected/selected target column)
    - feature_names (list of valid feature names after encoding)
    - preprocessor_objects (dict containing the fitted scalers/encoders for single prediction)
    """
    df = df.copy()
    
    # Handle missing values using forward and then backward fill
    df = df.ffill().bfill()
    
    if target_col is None:
        target_col = detect_target_column(df)
        
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    
    preprocessors = {
        'label_encoder': None,
        'scaler': None,
        'categorical_cols': [],
        'numerical_cols': [],
        'dummy_columns': []
    }
    
    # Encode Target if categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_col)
        preprocessors['label_encoder'] = le
        # Store inverse mapping to display correct string classes later
        preprocessors['target_classes'] = le.classes_
    else:
        preprocessors['target_classes'] = np.array(["Bad", "Good"]) # Assumption if already numeric (0/1)
        
    # Process features
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessors['categorical_cols'] = categorical_cols
    preprocessors['numerical_cols'] = numerical_cols
    
    # Ensure categorical columns are strictly treated as strings before encoding to avoid issues
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        
    # One-Hot Encoding for categoricals
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        # Store the exact columns created by get_dummies for future alignment
        preprocessors['dummy_columns'] = X.columns.tolist()
        
    # Normalize numerical features
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        preprocessors['scaler'] = scaler
        
    return X, y, target_col, list(X.columns), preprocessors

def preprocess_single_input(input_dict, preprocessors):
    """
    Preprocesses a single user input from the UI to match the training data shape and scaling.
    """
    df = pd.DataFrame([input_dict])
    
    # Ensure categorical columns are strings
    for col in preprocessors['categorical_cols']:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    # Apply One-Hot Encoding
    if len(preprocessors['categorical_cols']) > 0:
        df = pd.get_dummies(df, columns=preprocessors['categorical_cols'], drop_first=True)
        
    # Align columns with training data (add missing dummies, drop extra)
    # This is critical for predicting single inputs dynamically!
    if 'dummy_columns' in preprocessors and len(preprocessors['dummy_columns']) > 0:
        missing_cols = set(preprocessors['dummy_columns']) - set(df.columns)
        for c in missing_cols:
            df[c] = False  # Set missing OHE columns to False/0
        
        # Reorder to match training precisely
        df = df[preprocessors['dummy_columns']]
        
    # Scale numericals
    scaler = preprocessors.get('scaler')
    if scaler and len(preprocessors['numerical_cols']) > 0:
        # We need to make sure the numerical columns are scaled correctly
        # We will extract just the numerical ones for scaling, then put them back
        num_cols = preprocessors['numerical_cols']
        # Intercept columns that might have been rearranged
        df_nums = df[num_cols].copy() if all(c in df.columns for c in num_cols) else None
        
        # If the df columns got strictly transformed to dummy_columns array, some numerical cols might be lost or changed.
        # Actually, get_dummies keeps other columns. The `dummy_columns` list from training HAS all the final features.
        
        # Since we reordered `df` to exactly match `preprocessors['dummy_columns']` (which includes numericals),
        # we can safely transform the numerical cols in-place
        if num_cols and all(nc in df.columns for nc in num_cols):
             df[num_cols] = scaler.transform(df[num_cols])
             
    # Ensure to return correct order
    return df
