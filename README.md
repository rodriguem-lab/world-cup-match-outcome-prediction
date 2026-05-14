# World Cup Match Outcome Prediction

This project predicts international football match outcomes using historical
match data and machine learning.

## Problem

The goal is to predict one of three match outcomes:

- `0`: home team loses
- `1`: draw
- `2`: home team wins

## Dataset

The project uses the Kaggle international football results dataset. The main
raw file is:

```text
results.csv
```

The pipeline keeps modern matches from 1990 onward, creates form-based features
from past matches only, and splits the data chronologically:

- Train: matches before 2022
- Test: matches from 2022 onward

## Models

Champion model:

- Logistic Regression

Challenger models:

- Decision Tree
- Random Forest
- Gradient Boosting Classifier

`xgboost` was not available in the local environment, so scikit-learn's
`GradientBoostingClassifier` was used as the boosting fallback.

## Main Results

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Logistic Regression | 53.10% | 0.3739 |
| Decision Tree | 52.89% | 0.3569 |
| Random Forest | 52.93% | 0.3641 |
| Gradient Boosting | 52.93% | 0.3697 |

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the processed dataset:

```bash
python3 person1_data_pipeline.py
```

Train and evaluate all models:

```bash
python3 train_and_evaluate_models.py
```

## Key Files

```text
person1_data_pipeline.py
train_and_evaluate_models.py
final_report.md
model_comparison_table.csv
confusion_matrices.png
accuracy_comparison.png
feature_importance.png
```
