import json
import joblib
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
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
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_STATE = 42
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["Home loss", "Draw", "Home win"]

# Walk-forward CV fold boundaries: train on data before each cutpoint,
# test on that year, final evaluation on 2022+ (the held-out test set).
WF_CUTPOINTS = ["2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_inputs():
    X_train = np.load("data/processed/X_train.npy")
    X_test  = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test  = np.load("data/processed/y_test.npy")
    feature_names = [
        line.strip()
        for line in Path("data/processed/feature_names.txt").read_text().splitlines()
        if line.strip()
    ]
    return X_train, X_test, y_train, y_test, feature_names


def load_clean_df(feature_names):
    """Load raw (unscaled) features + dates from clean_matches.csv for walk-forward CV."""
    df = pd.read_csv("data/processed/clean_matches.csv", parse_dates=["date"])
    X_raw   = df[feature_names].values
    y_all   = df["target"].values
    dates   = df["date"].values
    return X_raw, y_all, dates


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metric_payload(model_name, best_params, y_true, y_pred, validation_macro_f1=None):
    report = classification_report(
        y_true, y_pred,
        labels=CLASS_LABELS, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    return {
        "model_name":            model_name,
        "best_params":           best_params,
        "validation_macro_f1":   validation_macro_f1,
        "test_accuracy":         accuracy_score(y_true, y_pred),
        "test_precision_macro":  precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "test_recall_macro":     recall_score(   y_true, y_pred, average="macro",    zero_division=0),
        "test_f1_macro":         f1_score(       y_true, y_pred, average="macro",    zero_division=0),
        "test_precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "test_recall_weighted":    recall_score(   y_true, y_pred, average="weighted", zero_division=0),
        "test_f1_weighted":        f1_score(       y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix":      matrix.tolist(),
        "classification_report": report,
    }


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_model(model_factory, param_grid, X_train, y_train, sample_weight=None):
    """
    Chronological 80/20 inner split for validation.
    Selects best params by macro F1 (not accuracy) to avoid majority-class bias.
    For GB, sample_weight is recomputed from y_inner to match the inner distribution.
    """
    split_index = int(len(X_train) * 0.8)
    X_inner, X_val = X_train[:split_index], X_train[split_index:]
    y_inner, y_val = y_train[:split_index], y_train[split_index:]

    # Fix: recompute weights from y_inner rather than slicing full-set weights
    sw_inner = compute_sample_weight("balanced", y_inner) if sample_weight is not None else None

    best_params = None
    best_score  = -1.0

    for params in param_grid:
        model = model_factory(params)
        fit_kwargs = {"sample_weight": sw_inner} if sw_inner is not None else {}
        model.fit(X_inner, y_inner, **fit_kwargs)
        score = f1_score(y_val, model.predict(X_val), average="macro", zero_division=0)
        if score > best_score:
            best_score  = score
            best_params = params

    final_model = model_factory(best_params)
    fit_kwargs  = {"sample_weight": sample_weight} if sample_weight is not None else {}
    final_model.fit(X_train, y_train, **fit_kwargs)

    return final_model, best_params, best_score


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_most_frequent_baseline(X_train, y_train):
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model, {"strategy": "most_frequent"}, None


def train_logistic_regression(X_train, y_train):
    param_grid = [{"C": 0.1}, {"C": 0.3}, {"C": 1.0}, {"C": 3.0}]

    def factory(params):
        return LogisticRegression(
            C=params["C"], max_iter=2000,
            random_state=RANDOM_STATE, solver="lbfgs",
            class_weight="balanced",
        )
    return tune_model(factory, param_grid, X_train, y_train)


def train_random_forest(X_train, y_train):
    param_grid = [
        {"n_estimators": 250, "max_depth": 8,    "min_samples_leaf": 1},
        {"n_estimators": 250, "max_depth": 12,   "min_samples_leaf": 1},
        {"n_estimators": 400, "max_depth": 12,   "min_samples_leaf": 2},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 3},
    ]

    def factory(params):
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            n_jobs=-1, random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    return tune_model(factory, param_grid, X_train, y_train)


def train_decision_tree(X_train, y_train):
    param_grid = [
        {"max_depth": 4,    "min_samples_leaf": 10},
        {"max_depth": 6,    "min_samples_leaf": 10},
        {"max_depth": 8,    "min_samples_leaf": 20},
        {"max_depth": 10,   "min_samples_leaf": 30},
        {"max_depth": None, "min_samples_leaf": 50},
    ]

    def factory(params):
        return DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    return tune_model(factory, param_grid, X_train, y_train)


def train_boosting_model(X_train, y_train):
    param_grid = [
        {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 180, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 180, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 220, "learning_rate": 0.03, "max_depth": 3},
        {"n_estimators": 160, "learning_rate": 0.10, "max_depth": 2},
    ]

    def factory(params):
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=RANDOM_STATE,
        )

    sample_weight = compute_sample_weight("balanced", y_train)
    return tune_model(factory, param_grid, X_train, y_train, sample_weight=sample_weight)


# ---------------------------------------------------------------------------
# Walk-forward cross-validation (fix 4)
# ---------------------------------------------------------------------------

def walk_forward_cv_scores(model_factory, X_raw, y_all, dates, cutpoints, sample_weight_fn=None):
    """
    Evaluates a model factory using expanding-window walk-forward CV.

    Each fold: train on data before cutpoints[i], test on [cutpoints[i], cutpoints[i+1]).
    The scaler is refit within each fold to avoid future-data leakage in scaling.
    Uses pre-tuned params (no inner grid search per fold) for tractability.

    Returns (mean_macro_f1, std_macro_f1) across folds.
    """
    fold_scores = []

    for i in range(len(cutpoints) - 1):
        train_end = pd.Timestamp(cutpoints[i])
        test_end  = pd.Timestamp(cutpoints[i + 1])

        train_mask = dates < train_end
        test_mask  = (dates >= train_end) & (dates < test_end)

        if train_mask.sum() < 200 or test_mask.sum() < 30:
            continue

        X_tr_raw = X_raw[train_mask]
        X_te_raw = X_raw[test_mask]
        y_tr     = y_all[train_mask]
        y_te     = y_all[test_mask]

        # Rescale per fold: no leakage from future data into scaler statistics
        fold_scaler = StandardScaler()
        X_tr = fold_scaler.fit_transform(X_tr_raw)
        X_te = fold_scaler.transform(X_te_raw)

        sw = sample_weight_fn(y_tr) if sample_weight_fn is not None else None
        model = model_factory()
        fit_kwargs = {"sample_weight": sw} if sw is not None else {}
        model.fit(X_tr, y_tr, **fit_kwargs)

        preds = model.predict(X_te)
        fold_scores.append(f1_score(y_te, preds, average="macro", zero_division=0))

    if not fold_scores:
        return float("nan"), float("nan")
    return float(np.mean(fold_scores)), float(np.std(fold_scores))


# ---------------------------------------------------------------------------
# Probability calibration (fix 5)
# ---------------------------------------------------------------------------

def calibrate_model(model_train_fn, X_train, y_train):
    """
    Platt scaling: train base model on 80% of training data, then fit a logistic
    regression on its raw probability outputs from the held-out 20%.
    The returned scaler is applied to the full model's predict_proba at test time.
    This avoids sklearn's cv="prefit" which was removed in 1.6+.
    """
    cal_split = int(len(X_train) * 0.8)
    X_fit, X_calib = X_train[:cal_split], X_train[cal_split:]
    y_fit, y_calib = y_train[:cal_split], y_train[cal_split:]

    base_model, _, _ = model_train_fn(X_fit, y_fit)
    platt = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    platt.fit(base_model.predict_proba(X_calib), y_calib)
    return platt


# ---------------------------------------------------------------------------
# Statistical significance — McNemar's test (fix 6)
# ---------------------------------------------------------------------------

def mcnemar_pvalue(y_true, preds_a, preds_b):
    """
    Exact McNemar's test: does model A differ significantly from model B?
    Tests H0: P(A correct, B wrong) = P(A wrong, B correct).
    Returns two-sided p-value.
    """
    correct_a = np.array(preds_a) == np.array(y_true)
    correct_b = np.array(preds_b) == np.array(y_true)
    b = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    if b + c == 0:
        return 1.0
    return binomtest(min(b, c), b + c, p=0.5, alternative="two-sided").pvalue


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_confusion_matrices(metric_by_key):
    n_models = len(metric_by_key)
    n_cols   = 2
    n_rows   = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.5 * n_rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for axis in axes[n_models:]:
        axis.axis("off")

    for axis, (title, metrics) in zip(axes, metric_by_key.items()):
        matrix    = np.array(metrics["confusion_matrix"])
        image     = axis.imshow(matrix, cmap="Blues")
        threshold = matrix.max() / 2

        axis.set_title(title)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=25, ha="right")
        axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                color = "white" if matrix[row, col] > threshold else "black"
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrices on Test Set", fontsize=14)
    fig.savefig("outputs/figures/confusion_matrices.png", dpi=180)
    plt.close(fig)


def plot_accuracy_comparison(metric_by_key):
    names      = list(metric_by_key.keys())
    accuracies = [metric_by_key[n]["test_accuracy"] for n in names]
    f1_macros  = [metric_by_key[n]["test_f1_macro"]  for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for axis, values, ylabel, title in [
        (axes[0], accuracies, "Test accuracy",  "Model Accuracy Comparison"),
        (axes[1], f1_macros,  "Test macro F1",  "Model Macro F1 Comparison"),
    ]:
        bars = axis.bar(names, values)
        axis.set_ylim(0, max(values) * 1.2)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=25)
        for bar, val in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    fig.tight_layout()
    fig.savefig("outputs/figures/accuracy_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model, feature_names):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
    )

    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(importance_df["feature"], importance_df["importance"], color="#4c78a8")
    axis.set_xlabel("Importance")
    axis.set_title("Boosting Model Feature Importance")
    axis.grid(axis="x", alpha=0.25)
    fig.savefig("outputs/figures/feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_model_comparison(metric_by_key, cv_scores, baseline_f1):
    """
    Writes model comparison CSV.
    cv_scores: dict of model_key -> (mean_cv_f1, std_cv_f1)
    baseline_f1: test macro F1 of the dummy baseline (for delta column)

    NOTE: accuracy is intentionally lower than the naive "always home-win" baseline
    because class_weight="balanced" trades accuracy for balanced recall across all
    three outcome classes.  Macro F1 is the primary comparison metric.
    """
    rows = []
    for name, metrics in metric_by_key.items():
        cv_mean, cv_std = cv_scores.get(name, (None, None))
        rows.append({
            "model":             name,
            "accuracy":          metrics["test_accuracy"],
            "f1_macro":          metrics["test_f1_macro"],
            "f1_macro_delta_vs_baseline": metrics["test_f1_macro"] - baseline_f1,
            "precision_macro":   metrics["test_precision_macro"],
            "recall_macro":      metrics["test_recall_macro"],
            "f1_weighted":       metrics["test_f1_weighted"],
            "cv_macro_f1_mean":  cv_mean,
            "cv_macro_f1_std":   cv_std,
            "validation_macro_f1": metrics["validation_macro_f1"],
        })

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    df.to_csv("outputs/reports/model_comparison_table.csv", index=False)


def write_classification_reports(metric_by_key):
    rows = []
    for model_name, metrics in metric_by_key.items():
        report = metrics["classification_report"]
        for class_name in CLASS_NAMES:
            cm = report[class_name]
            rows.append({
                "model":     model_name,
                "class":     class_name,
                "precision": cm["precision"],
                "recall":    cm["recall"],
                "f1_score":  cm["f1-score"],
                "support":   cm["support"],
            })
    pd.DataFrame(rows).to_csv("outputs/reports/classification_report_by_class.csv", index=False)


def write_logistic_regression_coefficients(model, feature_names):
    rows = []
    for row_index, class_label in enumerate(model.classes_):
        class_name   = CLASS_NAMES[CLASS_LABELS.index(class_label)]
        coefficients = model.coef_[row_index]
        for feature_name, coef in zip(feature_names, coefficients):
            rows.append({
                "class": class_name, "feature": feature_name,
                "coefficient": coef, "abs_coefficient": abs(coef),
            })

    df = pd.DataFrame(rows).sort_values(["class", "abs_coefficient"], ascending=[True, False])
    df.to_csv("outputs/reports/logistic_regression_coefficients.csv", index=False)


def write_significance_table(predictions_by_key, y_test):
    """
    Pairwise McNemar's test between all non-baseline model pairs.
    Saves p-values and significance flags to CSV.
    """
    model_keys = [k for k in predictions_by_key if k != "Baseline"]
    rows = []

    for i, name_a in enumerate(model_keys):
        for name_b in model_keys[i + 1:]:
            pval = mcnemar_pvalue(y_test, predictions_by_key[name_a], predictions_by_key[name_b])
            rows.append({
                "model_a":   name_a,
                "model_b":   name_b,
                "p_value":   round(pval, 4),
                "significant_at_0.05": pval < 0.05,
            })

    pd.DataFrame(rows).to_csv("outputs/reports/significance_tests.csv", index=False)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for d in ["outputs/metrics", "outputs/reports", "outputs/figures"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names = load_inputs()
    X_raw, y_all, dates = load_clean_df(feature_names)

    # ------------------------------------------------------------------
    # Train all models
    # ------------------------------------------------------------------
    baseline_model, baseline_params, _ = train_most_frequent_baseline(X_train, y_train)
    baseline_preds   = baseline_model.predict(X_test)
    baseline_metrics = metric_payload("Baseline: Most Frequent Class", baseline_params, y_test, baseline_preds)
    np.save("outputs/metrics/baseline_predictions.npy", baseline_preds)
    write_json("outputs/metrics/baseline_metrics.json", baseline_metrics)

    lr_model, lr_params, lr_val = train_logistic_regression(X_train, y_train)
    lr_preds   = lr_model.predict(X_test)
    lr_metrics = metric_payload("Logistic Regression", lr_params, y_test, lr_preds, lr_val)
    np.save("outputs/metrics/lr_predictions.npy", lr_preds)
    write_json("outputs/metrics/lr_metrics.json", lr_metrics)
    joblib.dump(lr_model, "outputs/metrics/lr_model.joblib")

    rf_model, rf_params, rf_val = train_random_forest(X_train, y_train)
    rf_preds   = rf_model.predict(X_test)
    rf_metrics = metric_payload("Random Forest", rf_params, y_test, rf_preds, rf_val)
    np.save("outputs/metrics/rf_predictions.npy", rf_preds)
    write_json("outputs/metrics/rf_metrics.json", rf_metrics)

    dt_model, dt_params, dt_val = train_decision_tree(X_train, y_train)
    dt_preds   = dt_model.predict(X_test)
    dt_metrics = metric_payload("Decision Tree", dt_params, y_test, dt_preds, dt_val)
    np.save("outputs/metrics/dt_predictions.npy", dt_preds)
    write_json("outputs/metrics/dt_metrics.json", dt_metrics)

    boosting_model, boosting_params, boosting_val = train_boosting_model(X_train, y_train)
    boosting_preds   = boosting_model.predict(X_test)
    boosting_metrics = metric_payload(
        "Gradient Boosting", boosting_params, y_test, boosting_preds, boosting_val,
    )
    boosting_metrics["note"] = (
        "sklearn GradientBoostingClassifier used as fallback (xgboost unavailable)."
    )
    np.save("outputs/metrics/xgb_predictions.npy", boosting_preds)
    write_json("outputs/metrics/xgb_metrics.json", boosting_metrics)

    metric_by_key = {
        "Baseline":           baseline_metrics,
        "Logistic Regression": lr_metrics,
        "Decision Tree":       dt_metrics,
        "Random Forest":       rf_metrics,
        "Gradient Boosting":   boosting_metrics,
    }

    # ------------------------------------------------------------------
    # Walk-forward cross-validation (fix 4)
    # Each fold rescales independently to avoid leakage.
    # Best params from main training are reused (no inner grid search per fold).
    # ------------------------------------------------------------------
    print("\nRunning walk-forward CV (3 folds: 2019 / 2020 / 2021 as test years)...")

    def lr_factory():
        return LogisticRegression(
            C=lr_params["C"], max_iter=2000,
            random_state=RANDOM_STATE, solver="lbfgs", class_weight="balanced",
        )

    def rf_factory():
        return RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced",
        )

    def dt_factory():
        return DecisionTreeClassifier(
            max_depth=dt_params["max_depth"],
            min_samples_leaf=dt_params["min_samples_leaf"],
            random_state=RANDOM_STATE, class_weight="balanced",
        )

    def gb_factory():
        return GradientBoostingClassifier(
            n_estimators=boosting_params["n_estimators"],
            learning_rate=boosting_params["learning_rate"],
            max_depth=boosting_params["max_depth"],
            random_state=RANDOM_STATE,
        )

    cv_scores = {
        "Baseline":           (None, None),
        "Logistic Regression": walk_forward_cv_scores(lr_factory, X_raw, y_all, dates, WF_CUTPOINTS),
        "Decision Tree":       walk_forward_cv_scores(dt_factory, X_raw, y_all, dates, WF_CUTPOINTS),
        "Random Forest":       walk_forward_cv_scores(rf_factory, X_raw, y_all, dates, WF_CUTPOINTS),
        "Gradient Boosting":   walk_forward_cv_scores(
            gb_factory, X_raw, y_all, dates, WF_CUTPOINTS,
            sample_weight_fn=lambda y: compute_sample_weight("balanced", y),
        ),
    }

    # ------------------------------------------------------------------
    # Probability calibration for RF (fix 5)
    # Platt scaling: base model trained on 80%, LR scaler fitted on 20% hold-out.
    # Applied to the full-trained rf_model's raw probs at test time.
    # ------------------------------------------------------------------
    print("Calibrating RF probabilities (Platt scaling)...")
    rf_platt = calibrate_model(train_random_forest, X_train, y_train)
    rf_calib_probs = rf_platt.predict_proba(rf_model.predict_proba(X_test))
    np.save("outputs/metrics/rf_calibrated_probs.npy", rf_calib_probs)
    joblib.dump(rf_platt, "outputs/metrics/rf_platt_scaler.joblib")

    # ------------------------------------------------------------------
    # McNemar's significance tests between all model pairs (fix 6)
    # ------------------------------------------------------------------
    predictions_by_key = {
        "Baseline":            baseline_preds,
        "Logistic Regression": lr_preds,
        "Decision Tree":       dt_preds,
        "Random Forest":       rf_preds,
        "Gradient Boosting":   boosting_preds,
    }
    significance_rows = write_significance_table(predictions_by_key, y_test)

    # ------------------------------------------------------------------
    # Write reports and visualisations
    # ------------------------------------------------------------------
    baseline_f1 = baseline_metrics["test_f1_macro"]
    write_model_comparison(metric_by_key, cv_scores, baseline_f1)
    write_classification_reports(metric_by_key)
    write_logistic_regression_coefficients(lr_model, feature_names)

    plot_confusion_matrices(metric_by_key)
    plot_accuracy_comparison(metric_by_key)
    plot_feature_importance(boosting_model, feature_names)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TEST SET RESULTS")
    print("=" * 65)
    print(f"{'Model':<28} {'Accuracy':>9} {'Macro F1':>9} {'F1 delta':>12}")
    print("-" * 65)
    for name, metrics in metric_by_key.items():
        delta = metrics["test_f1_macro"] - baseline_f1
        print(
            f"{name:<28} {metrics['test_accuracy']:>9.4f}"
            f" {metrics['test_f1_macro']:>9.4f} {delta:>+12.4f}"
        )

    print("\n" + "=" * 65)
    print("WALK-FORWARD CV  (macro F1, 3 folds: 2019 / 2020 / 2021)")
    print("=" * 65)
    for name, (mean, std) in cv_scores.items():
        if mean is None:
            print(f"  {name:<26}  n/a")
        else:
            print(f"  {name:<26}  {mean:.4f} ± {std:.4f}")

    print("\n" + "=" * 65)
    print("McNEMAR'S SIGNIFICANCE TESTS  (alpha = 0.05)")
    print("=" * 65)
    for row in significance_rows:
        sig = "SIGNIFICANT" if row["significant_at_0.05"] else "not significant"
        print(f"  {row['model_a']} vs {row['model_b']}: p={row['p_value']:.4f}  [{sig}]")

    print("\n" + "=" * 65)
    print("NOTE ON ACCURACY")
    print("=" * 65)
    print(
        "  Accuracy may fall below the naive baseline (always predict home win).\n"
        "  This is EXPECTED: class_weight='balanced' and macro-F1 tuning trade\n"
        "  raw accuracy for balanced recall across all three outcome classes.\n"
        "  Macro F1 is the primary comparison metric for this reason."
    )
    print("=" * 65)


if __name__ == "__main__":
    main()
