# Supervised Learning Models

## Table of Contents
1. [Introduction to Supervised Learning](#introduction-to-supervised-learning)
2. [Confusion Matrix: Understanding Classification Performance](#confusion-matrix)
3. [Logistic Regression](#logistic-regression)
4. [Random Forest](#random-forest)
5. [Recommendation Systems](#recommendation-systems)
6. [Model Evaluation Metrics](#model-evaluation-metrics)
7. [Hands-On Examples](#hands-on-examples)

---

## Introduction to Supervised Learning

Supervised learning is a machine learning approach where we train models on labeled data—data where we already know the correct answers. The model learns the relationship between inputs (features) and outputs (targets) from these examples.

### Key Concepts

**Supervised vs. Unsupervised Learning:**

| Aspect | Supervised | Unsupervised |
|--------|-----------|------------|
| Training Data | Labeled (has target values) | Unlabeled (no target values) |
| Goal | Predict target for new data | Find patterns/structure in data |
| Examples | Classification, Regression | Clustering, Dimensionality reduction |
| Use Cases | Email spam detection | Customer segmentation |

### Types of Supervised Learning

#### 1. **Classification**
Predicting which category something belongs to.

**Examples:**
- Is an email spam or not? (Binary classification)
- Which digit is this handwritten number? (Multi-class classification)
- Is this loan applicant high, medium, or low risk? (Multi-class classification)

**Algorithms:** Logistic Regression, Random Forest, Neural Networks, SVM

#### 2. **Regression**
Predicting continuous numerical values.

**Examples:**
- What will the house price be?
- What will the stock price be tomorrow?
- How many visitors will the website have?

**Algorithms:** Linear Regression, Polynomial Regression, SVR (Support Vector Regression), Random Forest Regressor

### Supervised Learning Workflow

```
1. Prepare Data
   ├─ Collect data
   ├─ Handle missing values
   └─ Scale/normalize
        ↓
2. Split Data
   ├─ Training set (70%)
   ├─ Validation set (15%)
   └─ Test set (15%)
        ↓
3. Choose Model
   └─ Select algorithm
        ↓
4. Train Model
   └─ Fit on training data
        ↓
5. Validate & Tune
   ├─ Evaluate on validation data
   └─ Adjust hyperparameters
        ↓
6. Test on Final Data
   └─ Evaluate on test set
        ↓
7. Make Predictions
   └─ Use on new, unseen data
```

---

## Confusion Matrix: Understanding Classification Performance

### What is a Confusion Matrix?

A confusion matrix is a table that shows how well a classification model performed. It compares predicted values vs. actual values, revealing what the model got right and what it got wrong.

### Understanding the Matrix

For **binary classification** (2 classes):

```
                    Predicted
                    Negative    Positive
Actual  
Negative    TN (True Negative)   FP (False Positive)
Positive    FN (False Negative)   TP (True Positive)
```

**Terminology:**
- **True Positive (TP)**: Model predicted positive, actually positive ✓ Correct!
- **True Negative (TN)**: Model predicted negative, actually negative ✓ Correct!
- **False Positive (FP)**: Model predicted positive, actually negative ✗ Type I Error
- **False Negative (FN)**: Model predicted negative, actually positive ✗ Type II Error

### Real-World Example: Email Spam Detection

```
                    Predicted
                    Not Spam   Spam
Actual  
Not Spam     95 (TN)      5 (FP)  ← Good emails marked spam (Bad!)
Spam          3 (FN)     97 (TP)  ← Spam not caught (Bad!)
```

**Interpretation:**
- 95 + 97 = 192 correct predictions
- 5 emails incorrectly marked as spam
- 3 spam emails not detected

### Confusion Matrix for Multi-Class Classification

When you have more than 2 classes, the matrix becomes larger. For example, with 3 classes (A, B, C):

```
        Predicted A  Predicted B  Predicted C
Actual A     90          8            2
Actual B      5         92            3
Actual C      1          4           95
```

The diagonal (90, 92, 95) shows correct predictions.

### Creating a Confusion Matrix in Python

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Assuming you have y_true (actual values) and y_pred (predictions)
cm = confusion_matrix(y_true, y_pred)

# Visualize
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
```

### What the Confusion Matrix Tells You

- **High diagonal values**: Model is doing well
- **High off-diagonal values**: Model is making mistakes
- **Asymmetric matrix**: Different types of errors more common

**Example - Good Model:**
```
95%  5%
 2% 98%  ← High diagonal, low off-diagonal
```

**Example - Poor Model:**
```
60% 40%
35% 65%  ← Low diagonal, high off-diagonal
```

---

## Logistic Regression

### What is Logistic Regression?

Despite its name, logistic regression is a **classification** algorithm, not a regression algorithm. It predicts the probability that something belongs to a particular class.

### Why "Logistic"?

Logistic regression uses the **logistic function** (sigmoid function) to convert linear relationships into probabilities:

```
Probability Output
    1.0 │         ╱╱╱╱╱
        │        ╱╱╱╱
    0.8 │       ╱╱╱╱
        │      ╱╱╱╱
    0.6 │     ╱╱╱╱
        │    ╱╱╱╱
    0.4 │   ╱╱╱╱
        │  ╱╱╱╱
    0.2 │ ╱╱╱╱
        │╱╱╱╱
    0.0 └─────────────────→ Input
```

The curve shows how the model's confidence grows smoothly from 0 to 1.

### When to Use Logistic Regression

**Best for:**
- Binary classification (yes/no, positive/negative)
- When you want probability outputs
- When your data should follow a linear boundary
- When you need an interpretable model
- For baseline models to compare against

**Not ideal for:**
- Complex non-linear relationships
- Very large feature sets
- Problems where classes aren't linearly separable

### How Logistic Regression Works

```
Step 1: Calculate linear combination
    z = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ

Step 2: Apply sigmoid function
    y = 1 / (1 + e^(-z))
    This gives probability between 0 and 1

Step 3: Make prediction
    If probability > 0.5: Predict class 1
    If probability ≤ 0.5: Predict class 0
```

### Implementation Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Get probabilities
probabilities = model.predict_proba(X_test)
print(f"Confidence for first prediction: {max(probabilities[0]):.2f}")
```

### Advantages and Disadvantages

**Advantages:**
- ✓ Simple and fast
- ✓ Interpretable (can see which features matter)
- ✓ Probabilistic output
- ✓ Works well on linearly separable data
- ✓ Good baseline model

**Disadvantages:**
- ✗ Assumes linear relationship
- ✗ Doesn't handle non-linear patterns well
- ✗ Sensitive to outliers
- ✗ Can struggle with high-dimensional data

### Recommended Video
Learn more about Logistic Regression: https://www.youtube.com/watch?v=7TqhmX92P6U

---

## Random Forest

### What is Random Forest?

Random Forest is an **ensemble learning** method that creates multiple decision trees and combines their predictions. It's powerful, flexible, and often produces excellent results.

### Why "Forest"?

- **Random**: Uses random subsets of data and features
- **Forest**: Creates multiple trees
- **Ensemble**: Combines predictions from all trees

### How It Works

```
1. Create Bootstrap Samples
   Original Data → Random Subset 1
                → Random Subset 2
                → Random Subset 3
                → ... N times

2. Build Decision Trees
   Subset 1 → Tree 1
   Subset 2 → Tree 2
   Subset 3 → Tree 3
   Subset N → Tree N

3. Make Predictions
   Input → Tree 1 → vote 1
        → Tree 2 → vote 2
        → Tree 3 → vote 3
        → Tree N → vote N
                   ↓
            Final Vote (majority wins)

4. Output Result
   Most common prediction = Final Answer
```

### Visual Example

Imagine predicting if someone will like a movie:

```
Input person's preferences
           ↓
    Tree 1: Age → Action? → Like? → YES
    Tree 2: Genre → Ratings → Like? → YES
    Tree 3: Reviews → Cast → Like? → NO
           ↓
    Final: 2 out of 3 trees vote YES → Prediction: LIKE
```

### When to Use Random Forest

**Best for:**
- Both classification and regression
- Non-linear relationships
- When you have many features
- When you want robust predictions
- When you want feature importance scores
- When interpretability matters less than accuracy

**Not ideal for:**
- Very large datasets (slow)
- When you need real-time predictions
- Ultra-interpretability required (black box model)

### Implementation Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train random forest
# n_estimators: number of trees (more = better but slower)
# max_depth: maximum depth of each tree (prevents overfitting)
# random_state: for reproducibility
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Feature importance
importances = model.feature_importances_
feature_names = iris.feature_names
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.3f}")
```

### Important Hyperparameters

**n_estimators**: Number of trees
- More trees = slower but potentially better
- Typical: 100-1000

```python
model = RandomForestClassifier(n_estimators=200)  # 200 trees
```

**max_depth**: Maximum depth of each tree
- Limits tree complexity
- Prevents overfitting
- None means unlimited (can overfit!)

```python
model = RandomForestClassifier(max_depth=10)  # Trees max 10 levels deep
```

**min_samples_split**: Minimum samples to split a node
- Higher value = simpler trees (prevents overfitting)
- Default: 2

```python
model = RandomForestClassifier(min_samples_split=5)  # Need at least 5 samples to split
```

**max_features**: Number of features to consider
- "sqrt": sqrt(n_features)
- "log2": log2(n_features)
- Lower value = more randomness = less overfitting

```python
model = RandomForestClassifier(max_features="sqrt")
```

### Feature Importance

Random Forest can tell you which features matter most:

```python
# Get feature importances
importances = model.feature_importances_

# Visualize
import matplotlib.pyplot as plt
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.show()
```

Features with higher importance scores contribute more to predictions.

### Advantages and Disadvantages

**Advantages:**
- ✓ Handles non-linear relationships
- ✓ Less prone to overfitting than single trees
- ✓ Provides feature importance
- ✓ Works with both classification and regression
- ✓ Robust to outliers
- ✓ Handles mixed data types

**Disadvantages:**
- ✗ Less interpretable than simple models
- ✗ Slower predictions (many trees to evaluate)
- ✗ Memory intensive
- ✗ Can still overfit if not tuned properly
- ✗ Biased toward high-cardinality features

### Recommended Video
Master Random Forest: https://www.youtube.com/watch?v=uuu44kIE1AI

---

## Recommendation Systems

### What is a Recommendation System?

A recommendation system predicts what items a user will like based on:
- Their past behavior (what they bought/rated/watched)
- Similar users' behavior
- Item characteristics
- Combinations of the above

### Real-World Examples

**Netflix**: "Because you watched X, you might like Y"
**Amazon**: "Customers who bought X also bought Y"
**Spotify**: "Recommended songs based on your listening history"
**YouTube**: "Videos you might like"

### Types of Recommendation Systems

#### 1. **Collaborative Filtering**
Idea: People who agreed in the past tend to agree in the future.

**User-based Collaborative Filtering:**
```
User A and User B rated similar movies similarly
→ Find movies User A watched that User B didn't
→ Recommend those to User B
```

**Item-based Collaborative Filtering:**
```
Users who liked Movie X also liked Movie Y
→ If User A liked Movie X
→ Recommend Movie Y to User A
```

**Advantages:**
- Simple and intuitive
- No need to understand item features
- Often works very well

**Disadvantages:**
- Cold start problem (new users/items)
- Sparsity (users rate few items)
- Scalability issues with large datasets

#### 2. **Content-Based Filtering**
Idea: Recommend items similar to what the user liked before.

```
User liked Movie A (Action, Thriller, 2020)
→ Find similar movies (Action, Thriller, 2019-2021)
→ Recommend those movies
```

**Advantages:**
- No cold start problem
- Transparent (can explain recommendations)
- Works with new items

**Disadvantages:**
- Requires good feature descriptions
- May recommend too similar items
- Limited novelty

#### 3. **Hybrid Approaches**
Combines collaborative filtering with content-based methods.

### How Did You Get Here?

Recommendation systems are often built on **matrix factorization**:

```
        Movie 1  Movie 2  Movie 3  Movie 4
User 1    5        4        ?        2
User 2    3        ?        2        ?
User 3    2        3        5        4
User 4    ?        2        3        5

Question marks = unknown ratings to predict
```

The algorithm tries to complete missing values based on patterns.

### Simple Recommendation Example

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie ratings
ratings = pd.DataFrame({
    'User 1': [5, 4, 0, 2, 0],
    'User 2': [3, 0, 2, 0, 5],
    'User 3': [2, 3, 5, 4, 0],
    'User 4': [0, 2, 3, 5, 4]
})

# Calculate similarity between users (ignore 0s)
# In real world, would use more sophisticated methods
user_similarity = cosine_similarity(ratings.T)

# Find similar users to User 1
similar_users = user_similarity[0].argsort()[::-1][1:3]
print(f"Users most similar to User 1: {similar_users}")
```

### Recommendation System Challenges

**Cold Start Problem:**
- New users have no history
- New items have no ratings
- Solution: Use content features, popularity, or hybrid approach

**Sparsity:**
- Users rate very few items
- Most of the rating matrix is empty
- Solution: Matrix factorization, deep learning

**Scalability:**
- Millions of users and items
- Computing similarities is expensive
- Solution: Approximate nearest neighbors, distributed computing

### Evaluation Metrics

**How do we know if recommendations are good?**

```python
from sklearn.metrics import mean_squared_error

# Predict ratings and compare to actual
predicted_ratings = model.predict(test_data)
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
print(f"RMSE: {rmse:.2f}")  # Lower is better

# Other metrics: Precision@K, Recall@K, NDCG, MAP
```

### Recommended Video
Understand Recommendation Systems: https://www.youtube.com/watch?v=0RT2Q0qwXSA

---

## Model Evaluation Metrics

After training a supervised learning model, you need to know: **How well does it actually work?**

### For Classification Models

#### 1. **Accuracy**
What percentage of predictions were correct?

```
Accuracy = (TP + TN) / Total Predictions
```

**When to use:** When classes are balanced
**When NOT to use:** Imbalanced datasets

Example:
- 95% accuracy seems great!
- But if 94% of data is negative class, a model that always predicts "negative" gets 94% accuracy

#### 2. **Precision**
Of the items the model predicted as positive, how many were actually positive?

```
Precision = TP / (TP + FP)
Focus: Minimize False Positives
```

**Use when:** False positives are expensive
- Example: Medical diagnosis (false positive = unnecessary treatment)
- Spam detection should have high precision (don't mark good emails as spam)

#### 3. **Recall (Sensitivity)**
Of all the actual positive cases, how many did the model find?

```
Recall = TP / (TP + FN)
Focus: Minimize False Negatives
```

**Use when:** False negatives are expensive
- Example: Disease detection (false negative = missing diagnosis)
- Fraud detection should have high recall (catch all fraud)

#### 4. **F1 Score**
Harmonic mean of precision and recall. Balances both.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Use when:** You want balanced precision and recall

#### 5. **ROC-AUC**
Shows model's performance across all classification thresholds.

**ROC Curve:** True Positive Rate vs False Positive Rate

Higher AUC (closer to 1.0) = better model

### Comparison Table

| Metric | Focus | Best For |
|--------|-------|----------|
| Accuracy | Overall correctness | Balanced classes |
| Precision | Minimize false alarms | Important not to wrongly predict positive |
| Recall | Minimize missed cases | Important not to miss positives |
| F1 Score | Balance both | Balanced focus |
| ROC-AUC | Threshold-independent | Comparing models |

### Implementation Example

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report,
                            confusion_matrix)

# Assuming y_true and y_pred
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"ROC-AUC:   {roc_auc:.3f}")

# Detailed report
print(classification_report(y_true, y_pred))
```

### For Regression Models

#### 1. **Mean Squared Error (MSE)**
Average squared difference between predicted and actual values

```
MSE = Σ(actual - predicted)² / n
```

**Lower is better. Penalizes large errors more.**

#### 2. **Root Mean Squared Error (RMSE)**
Square root of MSE. In same units as target.

```
RMSE = √MSE
```

**Interpretation:** Average prediction error

#### 3. **Mean Absolute Error (MAE)**
Average absolute difference between predicted and actual values

```
MAE = Σ|actual - predicted| / n
```

**Interpretation:** Average error amount (more interpretable than RMSE)

#### 4. **R² Score**
What proportion of variance in target is explained by the model?

```
R² ranges from 0 to 1 (can be negative)
1.0 = Perfect predictions
0.0 = No better than mean
```

**Interpretation:** What percentage of variance is explained

### Cross-Validation

Instead of one train-test split, use **K-Fold Cross-Validation**:

```
Original Data
    ├─ Fold 1: train on 4, test on 1
    ├─ Fold 2: train on 4, test on 1
    ├─ Fold 3: train on 4, test on 1
    ├─ Fold 4: train on 4, test on 1
    ├─ Fold 5: train on 4, test on 1
    └─ Average all results for final score
```

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## Hands-On Examples

### Example 1: Binary Classification with Logistic Regression

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
```

### Example 2: Multi-Class Classification with Random Forest

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train random forest
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")

# Feature importance
for name, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"{name}: {importance:.3f}")
```

### Example 3: Hyperparameter Tuning

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# Load your data
# X_train, X_test, y_train, y_test = ...

# Define hyperparameters to try
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Grid search
model = RandomForestClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-val score: {grid_search.best_score_:.3f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test set accuracy: {test_score:.3f}")
```

---

## Summary

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| Logistic Regression | Simple classification | Fast, interpretable | Assumes linearity |
| Random Forest | Complex classification | Powerful, handles non-linearity | Less interpretable |
| Recommendation System | Personalized suggestions | High engagement | Cold start problem |

**Remember:**
- Start with simple models (Logistic Regression)
- Gradually increase complexity (Random Forest)
- Always validate on unseen data
- Choose metrics that match your problem
- Watch for overfitting!

---

## Recommended Videos

- **Logistic Regression**: https://www.youtube.com/watch?v=yIYKR4sgzI8
- **Random Forest**: https://www.youtube.com/watch?v=uuu44kIE1AI
- **Recommendation Systems**: https://www.youtube.com/watch?v=U-yq3I9QugQ

---

**Next Step:** Ready to learn about finding patterns in data? Head to [Unsupervised Learning Models](./03_unsupervised_models.md)!
