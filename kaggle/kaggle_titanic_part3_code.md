# Kaggle Titanic — Part III Code Guide

This notebook guide focuses on classification metrics, validation stability, probability quality, threshold selection, subgroup analysis, and evidence-based model comparison.

Run the sections in order. The setup is self-contained, so this guide can be used without reopening the Part II notebook.

## 1. Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
```

## 2. Load the Titanic files

```python
TRAIN_PATH = "/kaggle/input/competitions/titanic/train.csv"
TEST_PATH = "/kaggle/input/competitions/titanic/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
```

Expected output:

- Training data: `(891, 12)`
- Test data: `(418, 11)`

## 3. Recreate the Part II features

```python
def add_features(df):
    out = df.copy()

    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    out["Title"] = (
        out["Name"]
        .str.extract(r",\s*([^.]*)\.", expand=False)
        .str.strip()
    )

    common_titles = ["Mr", "Mrs", "Miss", "Master"]
    out["Title"] = out["Title"].where(
        out["Title"].isin(common_titles),
        "Rare",
    )

    out["Deck"] = out["Cabin"].str[0].fillna("U")

    return out
```

```python
train_fe = add_features(train)
test_fe = add_features(test)

X = train_fe.drop(columns="Survived")
y = train_fe["Survived"]
```

## 4. Build a reusable preprocessing pipeline

```python
numeric_features = [
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
]

categorical_features = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "Deck",
]
```

```python
def make_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def make_model(estimator):
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("model", estimator),
        ]
    )
```

## 5. Create an honest validation split

```python
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_valid))
print("Training positive rate:", y_train.mean().round(3))
print("Validation positive rate:", y_valid.mean().round(3))
```

## 6. Train the baseline and generate both outputs

`predict()` returns hard labels. `predict_proba()` returns class probabilities.

```python
baseline = make_model(
    LogisticRegression(max_iter=1000)
)

baseline.fit(X_train, y_train)

valid_predictions = baseline.predict(X_valid)
valid_probabilities = baseline.predict_proba(X_valid)[:, 1]

print("First labels:", valid_predictions[:5])
print("First probabilities:", valid_probabilities[:5].round(3))
```

## 7. Establish the majority-class baseline

Always compare a model with a simple reference.

```python
majority_class = y_train.mode()[0]
majority_predictions = np.full(
    shape=len(y_valid),
    fill_value=majority_class,
)

majority_accuracy = accuracy_score(y_valid, majority_predictions)
model_accuracy = accuracy_score(y_valid, valid_predictions)

print(f"Majority baseline accuracy: {majority_accuracy:.3f}")
print(f"Model accuracy:             {model_accuracy:.3f}")
```

## 8. Confusion matrix

```python
ConfusionMatrixDisplay.from_predictions(
    y_valid,
    valid_predictions,
    display_labels=["Did not survive", "Survived"],
    cmap="Blues",
    values_format="d",
)

plt.title("Logistic Regression — Validation Confusion Matrix")
plt.show()
```

Extract the four counts:

```python
tn, fp, fn, tp = confusion_matrix(
    y_valid,
    valid_predictions,
).ravel()

print("True negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives:", tp)
```

## 9. Hard-label metrics

```python
accuracy = accuracy_score(y_valid, valid_predictions)
balanced_accuracy = balanced_accuracy_score(y_valid, valid_predictions)
precision = precision_score(y_valid, valid_predictions, zero_division=0)
recall = recall_score(y_valid, valid_predictions, zero_division=0)
specificity = tn / (tn + fp)
f1 = f1_score(y_valid, valid_predictions, zero_division=0)
mcc = matthews_corrcoef(y_valid, valid_predictions)

hard_label_metrics = pd.Series(
    {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "MCC": mcc,
    },
    name="score",
)

hard_label_metrics.round(3)
```

Interpretation:

- **Accuracy:** fraction of all correct predictions.
- **Balanced Accuracy:** average of Recall and Specificity.
- **Precision:** fraction of predicted survivors who actually survived.
- **Recall:** fraction of actual survivors found by the model.
- **Specificity:** fraction of actual non-survivors correctly rejected.
- **F1:** harmonic balance between Precision and Recall.
- **MCC:** single correlation-like summary using TP, TN, FP, and FN.

## 10. Full classification report

```python
print(
    classification_report(
        y_valid,
        valid_predictions,
        target_names=["Did not survive", "Survived"],
        digits=3,
        zero_division=0,
    )
)
```

Pay attention to:

- Metrics for each class separately.
- `macro avg`, which weights both classes equally.
- `weighted avg`, which weights classes by their frequency.
- `support`, the number of actual examples in each class.

## 11. Probability metrics

```python
roc_auc = roc_auc_score(y_valid, valid_probabilities)
average_precision = average_precision_score(y_valid, valid_probabilities)
logloss = log_loss(y_valid, valid_probabilities)
brier = brier_score_loss(y_valid, valid_probabilities)

probability_metrics = pd.Series(
    {
        "ROC-AUC": roc_auc,
        "Average Precision": average_precision,
        "Log Loss": logloss,
        "Brier Score": brier,
    },
    name="score",
)

probability_metrics.round(3)
```

Metric direction:

- Higher is better: ROC-AUC and Average Precision.
- Lower is better: Log Loss and Brier Score.

## 12. ROC curve

```python
false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
    y_valid,
    valid_probabilities,
)

plt.figure(figsize=(7, 5))
plt.plot(
    false_positive_rate,
    true_positive_rate,
    label=f"Model — AUC = {roc_auc:.3f}",
)
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random ranking")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate — Recall")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.2)
plt.show()
```

ROC-AUC measures ranking quality across thresholds. It does not tell you whether the probabilities are calibrated.

## 13. Precision–Recall curve

```python
curve_precision, curve_recall, pr_thresholds = precision_recall_curve(
    y_valid,
    valid_probabilities,
)

positive_rate = y_valid.mean()

plt.figure(figsize=(7, 5))
plt.plot(
    curve_recall,
    curve_precision,
    label=f"Model — AP = {average_precision:.3f}",
)
plt.axhline(
    positive_rate,
    linestyle="--",
    color="gray",
    label=f"Positive-rate baseline = {positive_rate:.3f}",
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.grid(alpha=0.2)
plt.show()
```

The Precision–Recall curve is especially useful when the positive class is rare or positive-class errors are the main concern.

## 14. Probability calibration

```python
CalibrationDisplay.from_predictions(
    y_valid,
    valid_probabilities,
    n_bins=8,
    strategy="quantile",
)

plt.title("Probability Calibration")
plt.grid(alpha=0.2)
plt.show()
```

A calibrated model behaves approximately like this:

- Among predictions near 0.20, about 20% are positive.
- Among predictions near 0.70, about 70% are positive.

Calibration is important when decisions depend on the probability itself, not only the final class.

## 15. Compare decision thresholds

```python
def evaluate_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "specificity": tn / (tn + fp),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "false_positives": fp,
        "false_negatives": fn,
    }
```

```python
threshold_results = pd.DataFrame(
    [
        evaluate_threshold(y_valid, valid_probabilities, threshold)
        for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]
    ]
)

threshold_results.round(3)
```

Visualize the tradeoff:

```python
threshold_results.plot(
    x="threshold",
    y=["precision", "recall", "f1", "specificity"],
    marker="o",
    figsize=(8, 5),
)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Metric Tradeoff by Decision Threshold")
plt.grid(alpha=0.2)
plt.show()
```

Choose a threshold from a defined objective, not simply because it gives the highest value in one small validation split.

## 16. Compare candidate models on the same split

```python
candidate_estimators = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    ),
}
```

```python
trained_models = {}
comparison_rows = []

for name, estimator in candidate_estimators.items():
    candidate = make_model(estimator)
    candidate.fit(X_train, y_train)

    predictions = candidate.predict(X_valid)
    probabilities = candidate.predict_proba(X_valid)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_valid, predictions).ravel()

    comparison_rows.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_valid, predictions),
            "Balanced Accuracy": balanced_accuracy_score(
                y_valid,
                predictions,
            ),
            "Precision": precision_score(
                y_valid,
                predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_valid,
                predictions,
                zero_division=0,
            ),
            "Specificity": tn / (tn + fp),
            "F1": f1_score(
                y_valid,
                predictions,
                zero_division=0,
            ),
            "MCC": matthews_corrcoef(y_valid, predictions),
            "ROC-AUC": roc_auc_score(y_valid, probabilities),
            "Average Precision": average_precision_score(
                y_valid,
                probabilities,
            ),
            "Log Loss": log_loss(y_valid, probabilities),
            "Brier Score": brier_score_loss(
                y_valid,
                probabilities,
            ),
        }
    )

    trained_models[name] = candidate

comparison = (
    pd.DataFrame(comparison_rows)
    .set_index("Model")
)

comparison.round(3)
```

Do not sort blindly by every metric. Decide which metric is primary and which metrics are constraints.

## 17. Stratified cross-validation

Use cross-validation to check whether a result is stable across several splits.

```python
cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "neg_log_loss": "neg_log_loss",
}
```

```python
cv_rows = []

for name, estimator in candidate_estimators.items():
    candidate = make_model(estimator)

    scores = cross_validate(
        candidate,
        X,
        y,
        cv=cross_validation,
        scoring=scoring,
        n_jobs=-1,
    )

    row = {"Model": name}

    for metric_name in scoring:
        values = scores[f"test_{metric_name}"]

        # Convert negative Log Loss back to positive loss.
        if metric_name == "neg_log_loss":
            values = -values
            display_name = "log_loss"
        else:
            display_name = metric_name

        row[f"{display_name}_mean"] = values.mean()
        row[f"{display_name}_std"] = values.std()

    cv_rows.append(row)

cv_results = pd.DataFrame(cv_rows).set_index("Model")
cv_results.round(3)
```

Report results as mean ± standard deviation. For example:

```python
for model_name, row in cv_results.iterrows():
    print(
        f"{model_name}: "
        f"ROC-AUC = {row['roc_auc_mean']:.3f} "
        f"± {row['roc_auc_std']:.3f}"
    )
```

## 18. Learning curve

A learning curve compares training and cross-validation performance as the amount of training data increases.

```python
learning_model = make_model(
    LogisticRegression(max_iter=1000)
)

train_sizes, train_scores, valid_scores = learning_curve(
    learning_model,
    X,
    y,
    cv=cross_validation,
    scoring="accuracy",
    train_sizes=np.linspace(0.20, 1.00, 5),
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
valid_mean = valid_scores.mean(axis=1)
valid_std = valid_scores.std(axis=1)
```

```python
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, marker="o", label="Training")
plt.plot(train_sizes, valid_mean, marker="o", label="Cross-validation")

plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15,
)
plt.fill_between(
    train_sizes,
    valid_mean - valid_std,
    valid_mean + valid_std,
    alpha=0.15,
)

plt.xlabel("Number of training examples")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.grid(alpha=0.2)
plt.show()
```

Interpretation:

- Both curves low and close: likely underfitting.
- Training high and validation much lower: likely overfitting.
- Curves converging at a useful score: healthier generalization.

## 19. Evaluate meaningful subgroups

Overall metrics can hide large differences between groups. The function below calculates the same hard-label metrics for each subgroup and includes its sample size.

```python
evaluation_frame = X_valid[["Sex", "Pclass", "Age"]].copy()
evaluation_frame["actual"] = y_valid
evaluation_frame["predicted"] = valid_predictions
evaluation_frame["probability"] = valid_probabilities

evaluation_frame["AgeGroup"] = pd.cut(
    evaluation_frame["Age"],
    bins=[0, 12, 18, 40, 65, np.inf],
    labels=["Child", "Teen", "Adult", "Older adult", "Senior"],
    include_lowest=True,
)
```

```python
def subgroup_metrics(group):
    y_true = group["actual"]
    y_pred = group["predicted"]

    return pd.Series(
        {
            "n": len(group),
            "positive_rate": y_true.mean(),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "f1": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
        }
    )
```

```python
metrics_by_sex = (
    evaluation_frame
    .groupby("Sex", observed=True)
    .apply(subgroup_metrics, include_groups=False)
)

metrics_by_class = (
    evaluation_frame
    .groupby("Pclass", observed=True)
    .apply(subgroup_metrics, include_groups=False)
)

metrics_by_age_group = (
    evaluation_frame
    .dropna(subset=["AgeGroup"])
    .groupby("AgeGroup", observed=True)
    .apply(subgroup_metrics, include_groups=False)
)

display(metrics_by_sex.round(3))
display(metrics_by_class.round(3))
display(metrics_by_age_group.round(3))
```

If your Pandas version does not support `include_groups=False`, remove that argument:

```python
metrics_by_sex = evaluation_frame.groupby("Sex").apply(subgroup_metrics)
```

Always interpret subgroup metrics together with `n`. A score from a very small group has high uncertainty.

## 20. Create a final evidence-based scorecard

Start with the cross-validation summary and add decision criteria.

```python
scorecard_columns = [
    "accuracy_mean",
    "accuracy_std",
    "balanced_accuracy_mean",
    "f1_mean",
    "roc_auc_mean",
    "roc_auc_std",
    "average_precision_mean",
    "log_loss_mean",
]

model_scorecard = cv_results[scorecard_columns].copy()
model_scorecard.round(3)
```

Answer these questions before choosing a model:

1. Which metric represents the objective?
2. Which error is more costly: FP or FN?
3. Is the cross-validation standard deviation acceptable?
4. Does performance collapse for any meaningful subgroup?
5. Is the added model complexity justified?
6. Are the probabilities sufficiently calibrated?

## 21. Refit the selected model

The example below chooses Logistic Regression. Replace it only if your evaluation supports another choice.

```python
selected_model = make_model(
    LogisticRegression(max_iter=1000)
)

selected_model.fit(X, y)
```

## 22. Create the Kaggle submission

The Titanic competition evaluates hard labels with Accuracy, so the default 0.50 threshold is a reasonable baseline. If you select another threshold, document why it was chosen using validation data.

Default threshold:

```python
test_predictions = selected_model.predict(test_fe)
```

Optional custom threshold:

```python
SELECTED_THRESHOLD = 0.50

test_probabilities = selected_model.predict_proba(test_fe)[:, 1]
test_predictions = (
    test_probabilities >= SELECTED_THRESHOLD
).astype(int)
```

Create and validate the file:

```python
submission = pd.DataFrame(
    {
        "PassengerId": test["PassengerId"],
        "Survived": test_predictions,
    }
)

assert submission.shape == (418, 2)
assert submission.columns.tolist() == ["PassengerId", "Survived"]
assert submission.isna().sum().sum() == 0
assert set(submission["Survived"].unique()).issubset({0, 1})

OUTPUT_PATH = "/kaggle/working/submission.csv"
submission.to_csv(OUTPUT_PATH, index=False)

print("Saved to:", OUTPUT_PATH)
submission.head()
```

## 23. Final recommendation template

Complete this text with your actual evidence:

```text
Primary metric:
Reason for choosing it:

Selected model:
Cross-validation result — mean ± standard deviation:

Most common validation error:
Threshold used:

Weakest subgroup and its sample size:
Probability calibration observation:

Why this model was selected:
Most important limitation:
Next experiment:
```

The final recommendation should connect the problem objective, error cost, validation evidence, stability, subgroup behavior, and model complexity.
