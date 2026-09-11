"""
Student Performance Prediction - Data Preprocessing Module
Author: Computer Engineering Student
Project: Student Performance Prediction Using Machine Learning
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw student performance CSV dataset.
    
    Parameters:
        filepath (str): Path to the raw CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at path: {filepath}")
    return pd.read_csv(filepath)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in the dataset using mode imputation for categorical columns.
    
    Rationale:
    - Only 3 categorical columns contain missing values (all < 1.5% of data):
      'Teacher_Quality' (78), 'Parental_Education_Level' (90), 'Distance_from_Home' (67).
    - Mode imputation replaces NaNs with the most frequent category, preserving all 6,607 rows
      without distorting class distributions.
      
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    df_clean = df.copy()
    
    # Columns with missing values identified in EDA
    cols_with_missing = ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']
    
    for col in cols_with_missing:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            
    return df_clean


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows if any exist in the dataset.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: Deduplicated DataFrame.
    """
    initial_rows = len(df)
    df_dedup = df.drop_duplicates().copy()
    removed = initial_rows - len(df_dedup)
    if removed > 0:
        print(f"Removed {removed} duplicate rows.")
    return df_dedup


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features using domain-appropriate Ordinal and Binary mappings.
    
    Rationale:
    - Ordinal features (e.g. Low -> 0, Medium -> 1, High -> 2) have a natural inherent hierarchy.
      Mapping them preserves this ranking and allows decision trees to make clean split decisions.
    - Binary features (e.g. No -> 0, Yes -> 1) are mapped to 0/1 indicator variables.
    - Tree-based models (Decision Tree & Random Forest) efficiently split numerical encodings without
      the dimensionality explosion caused by extensive one-hot encoding.
      
    Parameters:
        df (pd.DataFrame): DataFrame with clean categorical features.
        
    Returns:
        pd.DataFrame: DataFrame with encoded features.
    """
    df_encoded = df.copy()
    
    # Ordinal mappings (natural scale)
    ordinal_mappings = {
        'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
        'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
        'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
        'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
        'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
        'Parental_Education_Level': {'High School': 0, 'College': 1, 'Postgraduate': 2},
        'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2},
        'Peer_Influence': {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    }
    
    # Binary mappings
    binary_mappings = {
        'Extracurricular_Activities': {'No': 0, 'Yes': 1},
        'Internet_Access': {'No': 0, 'Yes': 1},
        'Learning_Disabilities': {'No': 0, 'Yes': 1},
        'School_Type': {'Public': 0, 'Private': 1},
        'Gender': {'Female': 0, 'Male': 1}
    }
    
    all_mappings = {**ordinal_mappings, **binary_mappings}
    
    for col, mapping in all_mappings.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
            
    return df_encoded


def create_binary_target(df: pd.DataFrame, threshold: float = 60.0) -> tuple[pd.DataFrame, pd.Series]:
    """
    Creates the binary classification target variable (Above 60% vs Below 60%)
    and separates feature matrix X from target vector y.
    
    Rationale for Data Leakage Prevention:
    - 'Exam_Score' directly defines the target. It MUST BE REMOVED from the feature matrix X
      so that models do not cheat by looking at the outcome during training.
      
    Parameters:
        df (pd.DataFrame): Processed DataFrame containing 'Exam_Score'.
        threshold (float): Score threshold for binary target (default 60.0).
        
    Returns:
        tuple[pd.DataFrame, pd.Series]: (X, y)
        - X: Feature matrix (all 19 input factors)
        - y: Binary target (1 = Above 60%, 0 = Below 60%)
    """
    if 'Exam_Score' not in df.columns:
        raise ValueError("Target column 'Exam_Score' not found in DataFrame.")
    
    # Create binary target: 1 = Above 60%, 0 = Below 60%
    y = (df['Exam_Score'] >= threshold).astype(int)
    y.name = 'Performance_Target'  # 1: Above 60%, 0: Below 60%
    
    # Drop target column to prevent data leakage
    X = df.drop(columns=['Exam_Score'])
    
    return X, y


def preprocess_pipeline(raw_filepath: str, threshold: float = 60.0) -> tuple[pd.DataFrame, pd.Series]:
    """
    Complete end-to-end preprocessing pipeline.
    
    Pipeline Steps:
    1. Load raw dataset (unmodified).
    2. Impute missing values (mode imputation for categorical columns).
    3. Drop duplicate records if any exist.
    4. Encode categorical and binary features.
    5. Construct binary target variable (Above 60% vs Below 60%).
    6. Return clean feature matrix X and target vector y.
    
    Parameters:
        raw_filepath (str): Path to raw CSV file.
        threshold (float): Target threshold (default 60.0).
        
    Returns:
        tuple[pd.DataFrame, pd.Series]: (X, y) ready for model training & evaluation.
    """
    raw_df = load_raw_data(raw_filepath)
    clean_df = handle_missing_values(raw_df)
    dedup_df = handle_duplicates(clean_df)
    encoded_df = encode_categorical_features(dedup_df)
    X, y = create_binary_target(encoded_df, threshold=threshold)
    
    return X, y


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "..", "dataset", "StudentPerformanceFactors.csv")
    
    print("Executing Preprocessing Pipeline Test...")
    X, y = preprocess_pipeline(data_path)
    
    print("\n" + "=" * 50)
    print("PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"Number of Samples     : {len(X)}")
    print(f"Number of Features    : {X.shape[1]}")
    print(f"Final Feature Columns :\n{list(X.columns)}")
    print(f"\nTarget Column Name    : {y.name}")
    print(f"Class Distribution (Counts):\n{y.value_counts().to_dict()}")
    print(f"Class Distribution (%):\n{(y.value_counts(normalize=True) * 100).round(2).to_dict()}")
    print(f"Missing Values in X   : {X.isnull().sum().sum()}")
    print("=" * 50)