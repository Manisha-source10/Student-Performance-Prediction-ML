# Student Performance Prediction Using Machine Learning

A small educational machine-learning demonstration that estimates a broad performance category from six academic indicators.

> **Important:** This repository uses a synthetically generated dataset. It does not contain ANITS student records, and its results are not validated for real-world educational decisions. It is an academic demonstration only—not an official grading, admissions, scholarship, or disciplinary tool.

## Features

- Reproducible synthetic dataset generation
- Data inspection and preprocessing
- Logistic Regression and Random Forest model comparison
- Accuracy, precision, recall, F1-score and confusion matrix
- Interactive Streamlit application
- Input-range validation
- Saved model pipeline and generated evaluation reports

## Project structure

```text
student-performance-prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── student_performance_synthetic.csv
├── src/
│   ├── generate_data.py
│   └── train_model.py
├── models/                 # generated after training; ignored by Git
├── reports/
│   └── figures/            # generated evaluation charts
└── tests/
    └── test_data.py
```

## 1. Set up the environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Generate the demonstration dataset

```bash
python src/generate_data.py
```

This creates `data/student_performance_synthetic.csv` with 1,800 synthetic records. The target labels are generated from a transparent weighted score plus noise.

## 3. Train and evaluate the models

```bash
python src/train_model.py
```

This trains Logistic Regression and Random Forest models, compares their test-set metrics, saves the best pipeline to `models/best_model.joblib`, and writes evaluation files to `reports/`.

The test set is held out from training. Metrics describe only this synthetic experiment and should not be treated as expected performance on real students.

## 4. Run the interactive application

```bash
streamlit run app.py
```

A browser tab will open. Enter the six indicators and select **Predict performance**.

## Input features

| Feature | Meaning | Allowed range |
|---|---|---:|
| Attendance (%) | Attendance percentage | 0–100 |
| Previous marks (%) | Previous examination result | 0–100 |
| Assignment completion (%) | Completed assignment percentage | 0–100 |
| Quiz average (%) | Average quiz/test score | 0–100 |
| Study hours/day | Approximate daily study hours | 0–16 |
| Recent marks trend | Approximate change in marks | -30 to +30 |

## Performance categories

- **Needs Support**
- **Moderate**
- **Good**

These are broad demonstration labels, not permanent descriptions of a student.

## Method overview

1. Generate synthetic tabular records with a fixed random seed.
2. Split the records into stratified training and test sets.
3. Fit preprocessing only on the training data.
4. Train Logistic Regression and Random Forest classifiers.
5. Compare weighted precision, recall, F1-score and accuracy.
6. Save the selected scikit-learn pipeline.
7. Load the pipeline in Streamlit for interactive inference.

## Responsible-use note

A real educational application would require institutionally approved data, informed and appropriate governance, privacy protection, bias and fairness evaluation, independent validation, and meaningful human oversight. No automated prediction should determine a student's grades, progression, admission, scholarship, or disciplinary outcome.

## Repository owner

Add your own GitHub username and repository URL here after creating the repository.
