# Introduction to Data Science & Fundamentals

## Table of Contents
1. [What is Predictive Modeling?](#what-is-predictive-modeling)
2. [Jupyter Notebook](#jupyter-notebook)
3. [Pandas: Data Manipulation](#pandas)
4. [NumPy: Numerical Computing](#numpy)
5. [Understanding Overfitting](#overfitting)
6. [Best Practices](#best-practices)

---

## What is Predictive Modeling?

Predictive modeling is the process of creating a mathematical model to predict future outcomes based on historical data. It's the foundation of machine learning and data science applications.

### Key Concepts

**Machine Learning Workflow:**
1. **Data Collection**: Gather relevant data for your problem
2. **Data Preprocessing**: Clean and prepare data for analysis
3. **Exploratory Data Analysis (EDA)**: Understand your data's structure and patterns
4. **Feature Engineering**: Create meaningful features for your model
5. **Model Selection**: Choose appropriate algorithms
6. **Training**: Learn patterns from data
7. **Evaluation**: Measure model performance
8. **Deployment**: Put your model into production

### Types of Predictive Tasks

- **Classification**: Predicting categories (e.g., spam vs. not spam)
- **Regression**: Predicting continuous values (e.g., house prices)
- **Clustering**: Finding groups in data (e.g., customer segments)
- **Recommendation**: Suggesting items users might like

---

## Jupyter Notebook

Jupyter Notebook is an open-source web application that allows you to create and share documents containing live code, equations, visualizations, and narrative text.

### Why Use Jupyter Notebook?

- **Interactive Development**: Write and run code in cells, seeing immediate results
- **Documentation**: Mix code with markdown explanations
- **Visualization**: Display plots and charts inline
- **Experimentation**: Easy to test hypotheses and explore data
- **Reproducibility**: Keep all analysis steps in one document

### Basic Jupyter Concepts

**Cells**: The building blocks of notebooks
- **Code cells**: Execute Python code
- **Markdown cells**: Add formatted text, headers, and explanations

**Shortcuts** (Command mode - press Esc first):
- `A`: Insert cell above
- `B`: Insert cell below
- `D, D`: Delete cell
- `Y`: Convert to code cell
- `M`: Convert to markdown cell
- `Shift + Enter`: Run cell and move to next
- `Ctrl + Enter`: Run cell and stay

### First Steps in Jupyter

```python
# Create a simple calculation
x = 10
y = 20
result = x + y
print(f"The sum of {x} and {y} is {result}")
```

### Recommended Video
Watch this introduction to get started: https://www.youtube.com/watch?v=W01tIRP_Rqs

---

## Pandas: Data Manipulation

Pandas is a powerful Python library for data manipulation and analysis. It provides data structures and functions needed to work with structured data.

### What is Pandas?

Pandas is built on top of NumPy and provides:
- **DataFrame**: 2D table with rows and columns (like Excel spreadsheet)
- **Series**: 1D array with labeled indices
- **Tools**: Reading, writing, cleaning, and transforming data

### Core Components

#### Series
A Series is a 1D array with an associated index:

```python
import pandas as pd

# Create a Series from a list
ages = pd.Series([25, 30, 35, 40], index=['Alice', 'Bob', 'Charlie', 'David'])
print(ages)
# Alice      25
# Bob        30
# Charlie    35
# David      40
```

#### DataFrame
A DataFrame is a 2D table, the most common Pandas data structure:

```python
# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Paris']
}
df = pd.DataFrame(data)
print(df)
#       Name  Age     City
# 0    Alice   25  New York
# 1      Bob   30    London
# 2  Charlie   35    Paris
```

### Essential Pandas Operations

**Loading Data:**
```python
# Read CSV file
df = pd.read_csv('data.csv')

# Read Excel file
df = pd.read_excel('data.xlsx')
```

**Exploring Data:**
```python
# First 5 rows
df.head()

# Data shape
df.shape  # (rows, columns)

# Data info
df.info()

# Statistical summary
df.describe()

# Check for missing values
df.isnull().sum()
```

**Data Cleaning:**
```python
# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()  # Remove rows with missing values
df = df.fillna(0)  # Fill missing values with 0

# Rename columns
df = df.rename(columns={'old_name': 'new_name'})
```

**Selecting Data:**
```python
# Select a single column
df['Age']

# Select multiple columns
df[['Name', 'Age']]

# Filter by condition
df[df['Age'] > 30]

# Filter multiple conditions
df[(df['Age'] > 25) & (df['City'] == 'London')]
```

**Grouping and Aggregation:**
```python
# Group by city and get mean age
avg_age_by_city = df.groupby('City')['Age'].mean()

# Multiple aggregations
df.groupby('City').agg({'Age': ['mean', 'max'], 'Name': 'count'})
```

**Creating New Columns:**
```python
# Add a new column
df['Age_Group'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Mature')

# Mathematical operations
df['Double_Age'] = df['Age'] * 2
```

### Why Pandas Matters
- **Data Cleaning**: Most of your time will be spent preparing data
- **Exploration**: Understand your data before building models
- **Transformation**: Convert raw data into useful features
- **Integration**: Works seamlessly with NumPy and scikit-learn

---

## NumPy: Numerical Computing

NumPy (Numerical Python) is a fundamental library for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices.

### Why NumPy?

- **Performance**: NumPy operations are much faster than pure Python
- **Convenience**: Simplifies mathematical and statistical operations
- **Integration**: Foundation for most Python data science libraries
- **Flexibility**: Works with arrays of any dimension

### Core NumPy Concepts

#### Arrays

NumPy's main object is the `ndarray` (n-dimensional array):

```python
import numpy as np

# Create arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array

# Array properties
print(arr2.shape)    # (2, 3) - 2 rows, 3 columns
print(arr2.dtype)    # int64
print(arr2.ndim)     # 2 - number of dimensions
print(arr2.size)     # 6 - total elements
```

#### Creating Arrays

```python
# Zeros and ones
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))

# Range of values
range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# Evenly spaced values
linspace_arr = np.linspace(0, 1, 5)  # [0., 0.25, 0.5, 0.75, 1.]

# Random numbers
random_arr = np.random.rand(3, 3)  # Random values between 0 and 1
```

### Basic Operations

**Arithmetic Operations:**
```python
arr = np.array([1, 2, 3, 4])

# Element-wise operations
arr + 2  # [3, 4, 5, 6]
arr * 3  # [3, 6, 9, 12]
arr ** 2  # [1, 4, 9, 16]

# Array to array operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr1 + arr2  # [5, 7, 9]
```

**Statistical Functions:**
```python
data = np.array([1, 2, 3, 4, 5])

np.mean(data)    # Average
np.median(data)  # Middle value
np.std(data)     # Standard deviation
np.sum(data)     # Sum
np.max(data)     # Maximum
np.min(data)     # Minimum
```

**Indexing and Slicing:**
```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

arr[0]        # First row: [1, 2, 3]
arr[0, 1]     # First row, second column: 2
arr[:, 1]     # All rows, second column: [2, 5, 8]
arr[1:3, 1:3] # Subarray: [[5, 6], [8, 9]]
```

**Broadcasting:**
One of NumPy's most powerful features. Allows operations between arrays of different shapes:

```python
# Broadcasting example
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr + np.array([1, 2, 3])  # Same as adding to each row
# [[2, 4, 6], [5, 7, 9]]
```

### NumPy in Machine Learning

```python
# Normalize features (common in ML)
data = np.array([1, 2, 3, 4, 5])
normalized = (data - data.mean()) / data.std()

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)  # Matrix multiplication
```

---

## Understanding Overfitting

Overfitting is one of the most critical concepts in machine learning. It occurs when a model learns the training data too well, including its noise and peculiarities, resulting in poor performance on new, unseen data.

### What is Overfitting?

Imagine you're preparing for an exam by memorizing practice questions word-for-word, but the actual exam has different questions. You'll likely fail! That's essentially what overfitting is.

**Key Terminology:**
- **Training Error**: How well the model performs on training data
- **Validation/Test Error**: How well the model performs on new data
- **Generalization**: Model's ability to perform well on unseen data

### Visualizing Overfitting

```
Performance
    ▲
    │     Underfitting    Optimal    Overfitting
    │         /            /\            \
    │        /            /  \            \
    │                    /    \            \
    ├─────────────────────────────────────────→ Model Complexity
    │
    │  (Too simple)    (Good fit)  (Too complex)
    │
```

### Visual Example

**Underfitting**: Model is too simple
```
Points: * * * * * * * * *
Model: _______________  (straight line doesn't capture pattern)
```

**Good Fit**: Model captures the pattern well
```
Points: * * * * * * * * *
Model: ~~~  (curve follows points)
```

**Overfitting**: Model memorizes noise
```
Points: * * * * * * * * *
Model: ~\~~/~~  (follows every single point, very wiggly)
```

### Causes of Overfitting

1. **Model Complexity**: Using too many features or parameters
2. **Insufficient Data**: Too few examples relative to model complexity
3. **Training Duration**: Training too long without validation monitoring
4. **Noise in Data**: Model learns random patterns instead of real relationships
5. **Lack of Regularization**: No penalty for complex models

### Signs of Overfitting

**In Your Training Process:**
- Training error keeps decreasing
- Validation error starts increasing
- Large gap between training and validation error
- Model performs great on training data but poorly on test data

```python
# Example: Training vs Validation Error
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_loss = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
val_loss =   [0.5, 0.4, 0.3, 0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
#                                           ↑ Starts increasing (overfitting!)
```

### How to Prevent Overfitting

#### 1. **Use More Data**
- Larger datasets reduce the impact of individual noisy examples
- More examples help model generalize better

```python
# Collect more training data
training_set_size = 1000  # Better than 100
```

#### 2. **Use Cross-Validation**
- Split data into multiple folds
- Train and validate on different splits
- Get more reliable performance estimates

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Average Score: {scores.mean()}")
```

#### 3. **Regularization**
- Add penalty for model complexity
- Encourages simpler models

```python
# L2 Regularization (Ridge)
from sklearn.linear_model import Ridge
model = Ridge(alpha=0.01)  # Higher alpha = more regularization

# L1 Regularization (Lasso)
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01)
```

#### 4. **Feature Selection**
- Use fewer, more relevant features
- Reduces model complexity

```python
# Keep only important features
selected_features = ['age', 'income', 'credit_score']
X_selected = X[selected_features]
```

#### 5. **Early Stopping**
- Monitor validation performance during training
- Stop when validation error stops improving

```python
# Stop training when validation performance plateaus
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=100)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
          early_stopping_rounds=10)
```

#### 6. **Ensemble Methods**
- Combine multiple models
- Reduces variance and overfitting

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest combines multiple decision trees
model = RandomForestClassifier(n_estimators=100)
```

#### 7. **Simplify the Model**
- Use fewer layers in neural networks
- Use simpler algorithms
- Reduce polynomial degree in regression

### The Bias-Variance Tradeoff

Overfitting is related to the **bias-variance tradeoff**:

- **High Bias**: Model is too simple (underfitting)
  - Makes strong assumptions about data
  - Misses underlying patterns
  - High training AND validation error

- **High Variance**: Model is too complex (overfitting)
  - Very sensitive to training data variations
  - Low training error, high validation error
  - Poor generalization

**Goal**: Find the sweet spot between bias and variance

```
Error
  ▲
  │         Total Error
  │            /\
  │    Bias   /  \   Variance
  │         /      \
  ├────────────────────────────→ Model Complexity
  │
  │ Simple          Complex
  │ Underfitting    Overfitting
```

### Monitoring for Overfitting

```python
import matplotlib.pyplot as plt

# Plot training vs validation error
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Healthy curve: both decrease and follow similar path
# Overfitting curve: training decreases while validation increases
```

### Best Practice Checklist

- [ ] Always split data into train/validation/test sets
- [ ] Monitor both training and validation error
- [ ] Use cross-validation for robust estimates
- [ ] Start with a simple model, then increase complexity
- [ ] Apply regularization when needed
- [ ] Use feature selection to remove irrelevant features
- [ ] Consider ensemble methods for better generalization
- [ ] Validate on completely separate test set before deployment

---

## Best Practices

### Data Science Workflow Checklist

1. **Define the Problem**
   - What are you trying to predict?
   - What data is available?
   - What's success for your project?

2. **Explore Your Data**
   - Check for missing values
   - Identify outliers
   - Understand feature distributions
   - Look for patterns and relationships

3. **Prepare Your Data**
   - Handle missing values
   - Remove or fix outliers
   - Normalize/standardize features
   - Create meaningful features

4. **Split Your Data**
   - Training set (70%): Train your model
   - Validation set (15%): Tune hyperparameters
   - Test set (15%): Final evaluation

5. **Start Simple**
   - Build a baseline model first
   - Use simple algorithms initially
   - Gradually increase complexity

6. **Evaluate Properly**
   - Don't just look at accuracy
   - Use appropriate metrics for your problem
   - Validate on unseen data
   - Compare to baseline

7. **Iterate and Improve**
   - Analyze errors
   - Try different features
   - Tune hyperparameters
   - But watch for overfitting!

### Code Organization Tips

```python
# Good template structure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
df = pd.read_csv('data.csv')

# 2. EXPLORE
print(df.head())
print(df.describe())

# 3. PREPARE
X = df.drop('target', axis=1)
y = df['target']

# 4. SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. SCALE
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. TRAIN
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 7. EVALUATE
score = model.score(X_test, y_test)
print(f"Accuracy: {score}")
```

---

## Recommended Resources

- **Video Tutorials**: 
  - Jupyter Notebook Intro: https://www.youtube.com/watch?v=W01tIRP_Rqs
  - Pandas Basics: https://www.youtube.com/watch?v=7TqhmX92P6U
  
- **Documentation**:
  - [Pandas Official Documentation](https://pandas.pydata.org/docs/)
  - [NumPy Official Documentation](https://numpy.org/doc/)
  - [Jupyter Documentation](https://jupyter.org/documentation)

- **Interactive Learning**:
  - Practice Jupyter notebooks and pandas with datasets
  - Build small projects to apply your knowledge
  - Monitor training vs validation errors in your projects

---

**Ready to move forward?** Once you're comfortable with these fundamentals, head to [Supervised Learning Models](./02_supervised_models.md) to learn about building predictive models!
