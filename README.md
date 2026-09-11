# Student Performance Prediction Using Machine Learning

## 📌 Project Overview

Student Performance Prediction is a Machine Learning project that predicts whether a student is likely to score above or below a predefined percentage based on academic and study-related factors.

The project compares two Machine Learning classification algorithms and selects the model with the better performance.

## 🎯 Project Objective

The main objective of this project is to develop a Machine Learning model that can predict student performance and identify students who may be at risk of achieving lower scores.

## 📊 Dataset

The project uses the `StudentPerformanceFactors.csv` dataset.

The dataset contains academic, study, attendance, and other student-related factors used for predicting student performance.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Git & GitHub

## 🤖 Machine Learning Algorithms

The following classification algorithms were implemented and compared:

1. Decision Tree Classifier
2. Random Forest Classifier

## 📈 Model Performance

| Model | Accuracy |
|---|---:|
| Decision Tree | 98.26% |
| Random Forest | 98.94% |

### Best Model

The **Random Forest Classifier** achieved the highest accuracy of **98.94%** and was selected as the best-performing model.

## 🔍 Prediction Testing

Two new student profiles were tested using the selected model.

- Test Case 1: **Above 60%** — Confidence: **100.00%**
- Test Case 2: **Below 60%** — Confidence: **75.00%**

## 📁 Project Structure

```text
Student_performance/
│
├── project.ipynb
├── preprocessing.py
├── predict.py
├── StudentPerformanceFactors.csv
├── .gitignore
└── README.md
