# Pima Indians Diabetes — Classification Project

A machine learning project that predicts whether a patient has diabetes based on 8 medical features. Built as part of my self-directed data science learning journey before starting my BCA degree.

---

## 📋 Project Overview

**Goal:** Build and compare two ML models (Logistic Regression and Random Forest) to predict diabetes from medical data, and evaluate which is more suitable for medical screening.

**Dataset:** Pima Indians Diabetes Database — 768 patient records with 8 medical features and a binary outcome (diabetic / non-diabetic).

**Tools:** Python, pandas, NumPy, scikit-learn, matplotlib, Google Colab

---

## 🔧 Methods

1. **EDA** — Explored data distributions; identified biologically impossible zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI as hidden missing values.
2. **Data Cleaning** — Replaced impossible zeros with NaN, then imputed with median (more robust to skewed distributions than mean).
3. **Train/Test Split** — Stratified 80/20 split with `random_state=42` to preserve class balance.
4. **Modeling** — Trained two classifiers with default parameters:
   - Logistic Regression (baseline)
   - Random Forest (comparison model)
5. **Evaluation** — Compared models using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.

---

## 📊 Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.70 | **0.78** |
| ROC-AUC | 0.81 | **0.82** |
| Diabetes Recall | 0.50 | **0.59** |

**Random Forest outperformed Logistic Regression on all key metrics.** The most important difference was Diabetes Recall — Random Forest caught 59% of real diabetic patients vs Logistic Regression's 50%. In a medical screening context, recall on the positive class matters more than overall accuracy because missing a diabetic patient (false negative) is much worse than wrongly flagging a healthy one (false positive).

---

## 🎓 What I Learned

- **Accuracy alone is misleading** for imbalanced classification problems. A 70% accurate model can still miss half the positive cases.
- **Domain context matters** when choosing metrics. In medical screening, recall trumps accuracy.
- **Data cleaning is the foundation** — properly handling hidden missing values (zeros that should be NaN) significantly impacts model quality.
- **Compare multiple models** with the same default settings before tuning anything. The simpler baseline isn't always the best starting point.
- **Stratified train/test splits** preserve class distributions and lead to more reliable evaluation.

---

## 📁 Repository Contents

- `EDA_pima_diabetes.ipynb` — Full Jupyter notebook with EDA, cleaning, modeling, and evaluation
- `README.md` — This file

---

## 🚀 Future Improvements

- Tune hyperparameters with GridSearchCV
- Try additional models (XGBoost, SVM)
- Address class imbalance using SMOTE or class weights
- Use stratified k-fold cross-validation for more robust evaluation
- Feature engineering (e.g., BMI categories, age groups)

---

## 👩‍💻 Author

**Riya Kumari**  
Aspiring data scientist | BCA student | Learning ML one project at a time

May 2026
