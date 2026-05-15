import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier


RANDOM_STATE = 42
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["Home loss", "Draw", "Home win"]


def load_inputs():
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")
    feature_names = [
        line.strip() for line in Path("feature_names.txt").read_text().splitlines() if line.strip()
    ]
    return X_train, X_test, y_train, y_test, feature_names


def chronological_validation_split(X_train, y_train, validation_ratio=0.2):
    split_index = int(len(X_train) * (1 - validation_ratio))
    return (
        X_train[:split_index],
        X_train[split_index:],
        y_train[:split_index],
        y_train[split_index:],
    )


def metric_payload(model_name, best_params, y_true, y_pred, validation_accuracy=None):
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    return {
        "model_name": model_name,
        "best_params": best_params,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "test_recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "test_f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "test_precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "test_recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "test_f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2))


def tune_model(model_factory, param_grid, X_train, y_train):
    X_inner, X_val, y_inner, y_val = chronological_validation_split(X_train, y_train)
    best_model = None
    best_params = None
    best_accuracy = -1.0

    for params in param_grid:
        model = model_factory(params)
        model.fit(X_inner, y_inner)

        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_model = model

    final_model = model_factory(best_params)
    final_model.fit(X_train, y_train)

    return final_model, best_params, best_accuracy

def train_most_frequent_baseline(X_train, y_train):
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model, {"strategy": "most_frequent"}, None

def train_logistic_regression(X_train, y_train):
    param_grid = [
        {"C": 0.1},
        {"C": 0.3},
        {"C": 1.0},
        {"C": 3.0},
    ]

    def factory(params):
        return LogisticRegression(
            C=params["C"],
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )

    return tune_model(factory, param_grid, X_train, y_train)


def train_random_forest(X_train, y_train):
    param_grid = [
        {"n_estimators": 250, "max_depth": 8, "min_samples_leaf": 1},
        {"n_estimators": 250, "max_depth": 12, "min_samples_leaf": 1},
        {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 2},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 3},
    ]

    def factory(params):
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    return tune_model(factory, param_grid, X_train, y_train)


def train_decision_tree(X_train, y_train):
    param_grid = [
        {"max_depth": 4, "min_samples_leaf": 10},
        {"max_depth": 6, "min_samples_leaf": 10},
        {"max_depth": 8, "min_samples_leaf": 20},
        {"max_depth": 10, "min_samples_leaf": 30},
        {"max_depth": None, "min_samples_leaf": 50},
    ]

    def factory(params):
        return DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=RANDOM_STATE,
        )

    return tune_model(factory, param_grid, X_train, y_train)


def train_boosting_model(X_train, y_train):
    param_grid = [
        {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 180, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 180, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 220, "learning_rate": 0.03, "max_depth": 3},
        {"n_estimators": 160, "learning_rate": 0.1, "max_depth": 2},
    ]

    def factory(params):
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=RANDOM_STATE,
        )

    return tune_model(factory, param_grid, X_train, y_train)


def plot_confusion_matrices(metric_by_key):
    n_models = len(metric_by_key)
    n_cols = 2
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 4.5 * n_rows),
        constrained_layout=True,
    )

    axes = np.array(axes).reshape(-1)

    for axis in axes[n_models:]:
        axis.axis("off")

    for axis, (title, metrics) in zip(axes, metric_by_key.items()):
        matrix = np.array(metrics["confusion_matrix"])

        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(title)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=25, ha="right")
        axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)

        threshold = matrix.max() / 2

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                color = "white" if matrix[row, col] > threshold else "black"
                axis.text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                    color=color,
                )

        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrices on Test Set", fontsize=14)
    fig.savefig("confusion_matrices.png", dpi=180)
    plt.close(fig)


def plot_accuracy_comparison(metric_by_key):
    names = list(metric_by_key.keys())
    accuracies = [metric_by_key[name]["test_accuracy"] for name in names]

    fig, axis = plt.subplots(figsize=(9, 5))

    bars = axis.bar(names, accuracies)

    axis.set_ylim(0, 1)
    axis.set_ylabel("Test accuracy")
    axis.set_title("Model Accuracy Comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=25)

    for bar, accuracy in zip(bars, accuracies):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{accuracy:.3f}",
            ha="center",
            va="bottom",
        )

    fig.savefig("accuracy_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_feature_importance(model, feature_names):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=True)

    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(importance_df["feature"], importance_df["importance"], color="#4c78a8")
    axis.set_xlabel("Importance")
    axis.set_title("Boosting Model Feature Importance")
    axis.grid(axis="x", alpha=0.25)
    fig.savefig("feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_model_comparison(metric_by_key):
    rows = []
    for name, metrics in metric_by_key.items():
        rows.append(
            {
                "model": name,
                "accuracy": metrics["test_accuracy"],
                "precision_macro": metrics["test_precision_macro"],
                "recall_macro": metrics["test_recall_macro"],
                "f1_macro": metrics["test_f1_macro"],
                "f1_weighted": metrics["test_f1_weighted"],
                "validation_accuracy": metrics["validation_accuracy"],
            }
        )

    comparison = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    comparison.to_csv("model_comparison_table.csv", index=False)
def write_classification_reports(metric_by_key):
    rows = []

    for model_name, metrics in metric_by_key.items():
        report = metrics["classification_report"]

        for class_name in CLASS_NAMES:
            class_metrics = report[class_name]

            rows.append(
                {
                    "model": model_name,
                    "class": class_name,
                    "precision": class_metrics["precision"],
                    "recall": class_metrics["recall"],
                    "f1_score": class_metrics["f1-score"],
                    "support": class_metrics["support"],
                }
            )

    report_table = pd.DataFrame(rows)
    report_table.to_csv("classification_report_by_class.csv", index=False)

def main():
    X_train, X_test, y_train, y_test, feature_names = load_inputs()

    baseline_model, baseline_params, baseline_validation_accuracy = train_most_frequent_baseline(
        X_train,
        y_train,
    )
    baseline_predictions = baseline_model.predict(X_test)
    baseline_metrics = metric_payload(
        "Baseline: Most Frequent Class",
        baseline_params,
        y_test,
        baseline_predictions,
        baseline_validation_accuracy,
    )
    np.save("baseline_predictions.npy", baseline_predictions)
    write_json("baseline_metrics.json", baseline_metrics)

    lr_model, lr_params, lr_validation_accuracy = train_logistic_regression(
        X_train,
        y_train,
    )
    lr_predictions = lr_model.predict(X_test)
    lr_metrics = metric_payload(
        "Logistic Regression",
        lr_params,
        y_test,
        lr_predictions,
        lr_validation_accuracy,
    )
    np.save("lr_predictions.npy", lr_predictions)
    write_json("lr_metrics.json", lr_metrics)

    rf_model, rf_params, rf_validation_accuracy = train_random_forest(
        X_train,
        y_train,
    )
    rf_predictions = rf_model.predict(X_test)
    rf_metrics = metric_payload(
        "Random Forest",
        rf_params,
        y_test,
        rf_predictions,
        rf_validation_accuracy,
    )
    np.save("rf_predictions.npy", rf_predictions)
    write_json("rf_metrics.json", rf_metrics)

    dt_model, dt_params, dt_validation_accuracy = train_decision_tree(
        X_train,
        y_train,
    )
    dt_predictions = dt_model.predict(X_test)
    dt_metrics = metric_payload(
        "Decision Tree",
        dt_params,
        y_test,
        dt_predictions,
        dt_validation_accuracy,
    )
    np.save("dt_predictions.npy", dt_predictions)
    write_json("dt_metrics.json", dt_metrics)

    boosting_model, boosting_params, boosting_validation_accuracy = train_boosting_model(
        X_train,
        y_train,
    )
    boosting_predictions = boosting_model.predict(X_test)
    boosting_metrics = metric_payload(
        "Gradient Boosting Classifier (XGBoost fallback)",
        boosting_params,
        y_test,
        boosting_predictions,
        boosting_validation_accuracy,
    )
    boosting_metrics["note"] = (
        "xgboost was not available in the local Python environment, so sklearn "
        "GradientBoostingClassifier was used as the planned fallback boosting model."
    )
    np.save("xgb_predictions.npy", boosting_predictions)
    write_json("xgb_metrics.json", boosting_metrics)

    metric_by_key = {
        "Baseline": baseline_metrics,
        "Logistic Regression": lr_metrics,
        "Decision Tree": dt_metrics,
        "Random Forest": rf_metrics,
        "Gradient Boosting": boosting_metrics,
    }

    plot_confusion_matrices(metric_by_key)
    plot_accuracy_comparison(metric_by_key)
    plot_feature_importance(boosting_model, feature_names)
    write_model_comparison(metric_by_key)
    write_classification_reports(metric_by_key)

    print("Model training and evaluation complete.")
    for name, metrics in metric_by_key.items():
        print(
            f"{name}: accuracy={metrics['test_accuracy']:.4f}, "
            f"macro_f1={metrics['test_f1_macro']:.4f}"
        )

if __name__ == "__main__":
    main()
