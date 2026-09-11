"""
Student Performance Prediction - Interactive Command-Line Program
Author: Fourth Year Computer Engineering Student
Project: Student Performance Prediction Using Machine Learning
"""

import os
import sys
import joblib
import pandas as pd


def load_saved_model(model_path: str = None) -> dict:
    """
    Loads the saved model and preprocessing artifact.
    """
    if model_path is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, "model", "student_model.pkl")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at: {model_path}. "
            f"Please run 'python src/train.py' first."
        )
        
    artifact = joblib.load(model_path)
    return artifact


def get_numeric_input(prompt: str, min_val: float, max_val: float, default: float = None) -> float:
    """
    Prompts the user for a numeric input with validation.
    """
    while True:
        try:
            default_str = f" [default: {default}]" if default is not None else ""
            user_val = input(f"{prompt} ({min_val}-{max_val}){default_str}: ").strip()
            
            if user_val == "" and default is not None:
                return float(default)
                
            val = float(user_val)
            if min_val <= val <= max_val:
                return val
            else:
                print(f"  [!] Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  [!] Invalid input. Please enter a valid number.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting input.")
            sys.exit(0)


def get_choice_input(prompt: str, options: list, default: str) -> str:
    """
    Prompts the user for a categorical choice with validation.
    """
    options_str = "/".join(options)
    while True:
        try:
            user_val = input(f"{prompt} ({options_str}) [default: {default}]: ").strip()
            if user_val == "":
                return default
            # Case-insensitive matching
            for opt in options:
                if user_val.lower() == opt.lower():
                    return opt
            print(f"  [!] Invalid choice. Please select from: {options_str}")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting input.")
            sys.exit(0)


def preprocess_student_input(student_data: dict, artifact: dict) -> pd.DataFrame:
    """
    Applies exact encoding, imputation defaults, and feature ordering as training.
    """
    feature_names = artifact['feature_names']
    mappings = artifact['categorical_mappings']
    defaults = artifact['imputation_defaults']
    
    data = student_data.copy()
    
    # Fill any missing keys with defaults
    for col, default_val in defaults.items():
        if col not in data or data[col] is None:
            data[col] = default_val
            
    # Apply categorical mappings
    for col, mapping in mappings.items():
        if col in data:
            val = data[col]
            if val in mapping:
                data[col] = mapping[val]
            else:
                data[col] = mapping.get(defaults.get(col, list(mapping.keys())[0]), 0)
                
    # Create DataFrame and enforce training column order
    df_input = pd.DataFrame([data])
    for col in feature_names:
        if col not in df_input.columns:
            # Fallback default
            df_input[col] = 0
            
    df_input = df_input[feature_names]
    return df_input


def predict_performance(student_data: dict, artifact: dict) -> tuple[str, float]:
    """
    Runs model inference and returns predicted label and confidence.
    """
    model = artifact['model']
    target_labels = artifact['target_labels']
    
    df_processed = preprocess_student_input(student_data, artifact)
    pred_class = int(model.predict(df_processed)[0])
    pred_proba = model.predict_proba(df_processed)[0]
    
    predicted_label = target_labels.get(pred_class, "Unknown")
    confidence = pred_proba[pred_class] * 100
    return predicted_label, confidence

def predict_student(student_data: dict, artifact: dict = None) -> tuple[str, float]:
    """
    Predict student performance using the saved ML model.
    """
    if artifact is None:
        artifact = load_saved_model()

    return predict_performance(student_data, artifact)



def interactive_cli():
    """
    Interactive command-line interface for student performance prediction.
    """
    print("=" * 60)
    print("   STUDENT PERFORMANCE PREDICTION SYSTEM (ML MINI PROJECT)   ")
    print("=" * 60)
    print("Loading trained machine learning model...")
    
    try:
        artifact = load_saved_model()
        print(f"Loaded Model: {artifact.get('model_name', 'Random Forest Classifier')}")
        print(f"Model Test Accuracy: {artifact.get('test_accuracy', 0.9894)*100:.2f}%\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Please enter the student's details below:")
    print("-" * 60)
    
    # Core academic inputs
    hours_studied = get_numeric_input("Study Hours per week", 0, 60, default=20)
    attendance = get_numeric_input("Attendance Percentage (%)", 0, 100, default=80)
    previous_scores = get_numeric_input("Previous Marks / Score (0-100)", 0, 100, default=70)
    tutoring_sessions = get_numeric_input("Monthly Tutoring Sessions Attended", 0, 15, default=1)
    sleep_hours = get_numeric_input("Average Sleep Hours per night", 3, 14, default=7)
    physical_activity = get_numeric_input("Physical Activity (hours/week)", 0, 15, default=3)
    
    # Optional academic support inputs
    print("\nAcademic Support & Environment (Press Enter to use defaults):")
    parental_involvement = get_choice_input("Parental Involvement", ["Low", "Medium", "High"], default="Medium")
    access_resources = get_choice_input("Access to Learning Resources", ["Low", "Medium", "High"], default="Medium")
    motivation_level = get_choice_input("Student Motivation Level", ["Low", "Medium", "High"], default="Medium")
    teacher_quality = get_choice_input("Teacher Quality", ["Low", "Medium", "High"], default="Medium")
    internet_access = get_choice_input("Internet Access at Home", ["Yes", "No"], default="Yes")
    
    # Assemble input dictionary
    student_record = {
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Parental_Involvement': parental_involvement,
        'Access_to_Resources': access_resources,
        'Extracurricular_Activities': 'No',
        'Sleep_Hours': sleep_hours,
        'Previous_Scores': previous_scores,
        'Motivation_Level': motivation_level,
        'Internet_Access': internet_access,
        'Tutoring_Sessions': tutoring_sessions,
        'Family_Income': 'Medium',
        'Teacher_Quality': teacher_quality,
        'School_Type': 'Public',
        'Peer_Influence': 'Neutral',
        'Physical_Activity': physical_activity,
        'Learning_Disabilities': 'No',
        'Parental_Education_Level': 'High School',
        'Distance_from_Home': 'Near',
        'Gender': 'Male'
    }
    
    # Predict
    result_label, confidence = predict_performance(student_record, artifact)
    
    print("\n" + "=" * 60)
    print("                  PREDICTION RESULT                         ")
    print("=" * 60)
    print(f"Predicted Result : {result_label}")
    print(f"Model Confidence : {confidence:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    interactive_cli()