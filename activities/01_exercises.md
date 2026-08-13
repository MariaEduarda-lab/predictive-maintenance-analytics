# Exercise List: Python, Pandas, Visualization, Statistics, and Introductory Machine Learning

This exercise list is designed as supplementary practice material. It does not replace classes, professor guidance, official assignments, rubrics, or course requirements.

## Instructions

- Answer all questions in English.
- For coding questions, write code and briefly explain your reasoning, assumptions, and expected output.
- Use clear variable names and organize your code so another student could read and understand it.
- When a question asks for analysis, include both the code and a short interpretation of the result.

* Monitor Tip: there are a lot of questions here. You can do all the objective ones and than you should focous in the code ones. Choose the ones which make sense for you and your learning path. 

---

## Part 1: Visualization, Statistics, Modeling Concepts, and CRISP-DM

Answer the following 30 objective questions. Some questions use single-choice alternatives, some use assertion-reason format, some ask you to identify the correct set of statements, and some ask you to recognize Python errors.

### Single-Choice Questions

1. **Matplotlib API Choice**  
   A student is creating a figure with four subplots, shared axes, customized titles, and different annotations in each panel. Which approach is generally most appropriate?
   - A. Use only `plt.plot()` repeatedly, relying on the current active axes.
   - B. Use the object-oriented interface with `fig, axes = plt.subplots(...)` and call methods on each `Axes`.
   - C. Use only `pandas.DataFrame.describe()` because it automatically creates subplots.
   - D. Use `sns.heatmap()` for all plots because it is based on Matplotlib.

2. **Line Plot Interpretation**  
   You have monthly revenue data ordered from January to December. Which visualization is usually most appropriate for emphasizing change over time?
   - A. A line plot with months on the x-axis and revenue on the y-axis.
   - B. A pie chart with one slice for each month.
   - C. A correlation heatmap using only the month names.
   - D. A confusion matrix.

3. **Histogram vs Boxplot**  
   A teacher wants students to inspect both the overall shape of exam score distribution and the presence of possible outliers. Which pair of plots best matches these goals?
   - A. Histogram for distribution shape and boxplot for outliers.
   - B. Heatmap for distribution shape and line plot for outliers.
   - C. Bar chart for distribution shape and scatter plot for outliers.
   - D. Pie chart for distribution shape and count plot for outliers.

4. **Seaborn Semantic Mappings**  
   In a Seaborn scatter plot, what is the main purpose of using `hue`, `style`, or `size`?
   - A. To remove missing values before plotting.
   - B. To map additional variables to visual properties of the points.
   - C. To convert categorical variables into numeric targets automatically.
   - D. To guarantee that correlation implies causation.

5. **Correlation Interpretation**  
   A correlation matrix shows that `hours_studied` and `exam_score` have a correlation of `0.82`. Which interpretation is most appropriate?
   - A. More study hours are strongly associated with higher exam scores, but this alone does not prove causation.
   - B. Study hours cause exactly 82% of the final grade.
   - C. The two variables are unrelated because correlation only applies to categorical data.
   - D. The model must be unsupervised because correlation is positive.

6. **Mean and Median Under Outliers**  
   A dataset of salaries contains one extremely high executive salary. Which statement is most likely true?
   - A. The mean may increase substantially, while the median is usually more resistant.
   - B. The median must always be larger than the mean.
   - C. Both mean and median are unaffected by outliers.
   - D. Standard deviation becomes zero.

7. **Supervised vs Unsupervised Learning**  
   Which task is the clearest example of unsupervised learning?
   - A. Predicting whether an email is spam using emails labeled as spam or not spam.
   - B. Predicting house prices using historical prices.
   - C. Grouping customers into segments using purchase behavior without predefined segment labels.
   - D. Predicting student approval using final course outcomes.

8. **CRISP-DM Phase Identification**  
   A team interviews stakeholders, defines project success criteria, and translates a business problem into a data mining goal. Which CRISP-DM phase is this?
   - A. Modeling
   - B. Business Understanding
   - C. Deployment
   - D. Data Preparation

### Assertion-Reason Questions

For questions 9 to 14, choose one option:

- A. Both the assertion and the reason are true, and the reason correctly explains the assertion.
- B. Both the assertion and the reason are true, but the reason does not correctly explain the assertion.
- C. The assertion is true, but the reason is false.
- D. The assertion is false, but the reason is true.
- E. Both the assertion and the reason are false.

9. **Train/Test Split**  
   Assertion: Evaluating a supervised model only on the training data can produce an overly optimistic estimate of performance.  
   Reason: A model may learn patterns specific to the training data that do not generalize to unseen data.

10. **Silhouette Score**  
   Assertion: A silhouette score close to `+1` generally indicates better-defined clusters.  
   Reason: The silhouette coefficient compares average distance to points in the same cluster with average distance to the nearest different cluster.

11. **Elbow Method**  
   Assertion: In the elbow method, inertia usually increases as K increases.  
   Reason: Adding more clusters gives the algorithm more centroids, which usually allows points to be closer to their assigned centroid.

12. **Seaborn and Matplotlib**  
   Assertion: Seaborn can be described as a high-level statistical visualization library built on top of Matplotlib.  
   Reason: Seaborn is designed to make many statistical plots easier to create and customize.

13. **Classification Thresholds**  
   Assertion: Changing the probability threshold in a binary classifier can change the number of false positives and false negatives.  
   Reason: The threshold controls how predicted probabilities are converted into class labels.

14. **CRISP-DM Evaluation**  
   Assertion: In CRISP-DM, the Evaluation phase should consider whether model results satisfy the original business objectives, not only whether a metric is high.  
   Reason: A technically accurate model may still be unsuitable if it fails to address the business problem or practical constraints.

### Multiple-Statement Questions

For questions 15 to 20, analyze the statements and choose the alternative containing only the correct statements.

15. **Visualization Selection**  
   I. A scatter plot is useful for inspecting the relationship between two numeric variables.  
   II. A histogram is useful for inspecting the distribution of one numeric variable.  
   III. A confusion matrix is primarily used to visualize clustering inertia.  
   IV. A bar chart can compare aggregated values across categories.  
   - A. I, II, and IV only
   - B. I and III only
   - C. II, III, and IV only
   - D. I, II, III, and IV

16. **Statistics and Data Quality**  
   I. Missing values can bias analysis if they are not handled carefully.  
   II. Standard deviation measures spread around the mean.  
   III. Correlation always proves causation.  
   IV. Outliers can affect the mean more strongly than the median.  
   - A. I, II, and IV only
   - B. II and III only
   - C. I and III only
   - D. III and IV only

17. **Model Types**  
   I. Classification predicts discrete categories.  
   II. Regression predicts continuous numeric values.  
   III. Clustering usually requires target labels during training.  
   IV. Dimensionality reduction can be used before visualization or modeling.  
   - A. I, II, and IV only
   - B. I and III only
   - C. II and III only
   - D. I, II, III, and IV

18. **K-Means, Elbow, and Silhouette**  
   I. K-Means requires choosing the number of clusters K.  
   II. Inertia alone always identifies the true number of natural groups without ambiguity.  
   III. Silhouette score considers both cohesion and separation.  
   IV. A silhouette score near zero may indicate overlapping clusters or boundary points.  
   - A. I, III, and IV only
   - B. I and II only
   - C. II and IV only
   - D. I, II, III, and IV

19. **CRISP-DM Process**  
   I. Business Understanding comes before Modeling in the CRISP-DM process.  
   II. Data Understanding includes exploring data and checking data quality.  
   III. Deployment only means writing Python code to train a model.  
   IV. Evaluation includes reviewing whether the project goals have been met.  
   - A. I, II, and IV only
   - B. I and III only
   - C. II and III only
   - D. I, II, III, and IV

20. **Introductory Machine Learning Workflow**  
   I. Feature scaling can be important for distance-based algorithms.  
   II. Overfitting happens when a model performs well on training data but poorly on new data.  
   III. Data visualization is useful only after the final model has been deployed.  
   IV. Train/test splitting helps estimate how a model may perform on unseen examples.  
   - A. I, II, and IV only
   - B. I and III only
   - C. II and III only
   - D. I, II, III, and IV

### Additional Single-Choice Questions

21. **Python Error Recognition: Missing Import**  
   A student runs the following code:

   ```python
   df = pd.DataFrame({"score": [7, 8, 9]})
   print(df.mean())
   ```

   The code raises `NameError: name 'pd' is not defined`. What is the most likely correction?
   - A. Add `import pandas as pd` before creating the DataFrame.
   - B. Replace `pd.DataFrame` with `plt.DataFrame`.
   - C. Convert the list into a tuple before creating the DataFrame.
   - D. Use `df.mean(axis="score")` instead of `df.mean()`.

22. **Python Error Recognition: Wrong Column Name**  
   A DataFrame has columns `["age", "income", "segment"]`. The student runs:

   ```python
   df["salary"].mean()
   ```

   Which error or issue is most likely?
   - A. `KeyError`, because the column `"salary"` does not exist.
   - B. `SyntaxError`, because brackets cannot be used with DataFrames.
   - C. `TypeError`, because `.mean()` cannot be used in Pandas.
   - D. No error, because Pandas automatically renames `"income"` to `"salary"`.

23. **Python Error Recognition: Seaborn Column Mapping**  
   Consider the code:

   ```python
   sns.scatterplot(data=df, x="height", y="weight", hue="class")
   ```

   If `df` does not contain a column called `"class"`, what is the most appropriate diagnosis?
   - A. The plot may fail because `hue` references a missing column.
   - B. The plot will automatically create the `"class"` column.
   - C. The plot will switch from scatter plot to histogram.
   - D. The plot will ignore both `x` and `y`.

24. **Python Error Recognition: Train/Test Split Assignment**  
   A student writes:

   ```python
   X_train, X_test = train_test_split(X, y, test_size=0.2)
   ```

   Why is this code likely incorrect?
   - A. `train_test_split(X, y, ...)` returns four objects, not two.
   - B. `test_size` cannot be a decimal value.
   - C. `train_test_split` can only be used for unsupervised learning.
   - D. `X` and `y` must always be strings.

25. **Python Error Recognition: Matplotlib Object-Oriented API**  
   A student writes:

   ```python
   fig, ax = plt.subplots()
   ax.xlabel("Month")
   ```

   What is the correct way to set the x-axis label using the object-oriented API?
   - A. `ax.set_xlabel("Month")`
   - B. `ax.labelx("Month")`
   - C. `fig.xlabel("Month")`
   - D. `plt.axes.xlabel("Month")`

26. **Choosing the Correct Plot**  
   You want to compare the distribution of `salary` across three departments and inspect differences in median and spread. Which plot is most appropriate?
   - A. Boxplot
   - B. Pie chart
   - C. Confusion matrix
   - D. Line plot with time on the x-axis

27. **Data Leakage Recognition**  
   A model predicts whether a student will pass a course. The feature table includes a column called `final_grade`, which is only known after the course ends. What is the main problem?
   - A. Data leakage, because the model uses information that would not be available at prediction time.
   - B. Class imbalance, because final grades are always categorical.
   - C. Underfitting, because the model has too few columns.
   - D. Dimensionality reduction, because the target has been removed.

28. **Classification Metric Choice**  
   In a medical screening problem, missing a positive case is much more serious than incorrectly flagging a healthy patient for further testing. Which metric should receive special attention?
   - A. Recall for the positive class
   - B. Mean squared error
   - C. Inertia
   - D. Silhouette score

29. **CRISP-DM and Data Preparation**  
   Which activity best belongs to the Data Preparation phase of CRISP-DM?
   - A. Cleaning missing values, transforming variables, and constructing the modeling table.
   - B. Interviewing stakeholders to define business objectives.
   - C. Presenting the final system to users after approval.
   - D. Choosing whether the project is financially worthwhile before seeing any data.

30. **Python Error Recognition: Scaling Before K-Means**  
   A student applies K-Means to a dataset with `annual_income` ranging from 20,000 to 200,000 and `age` ranging from 18 to 70, without scaling. What is the most likely issue?
   - A. The income variable may dominate the distance calculation because it has a much larger numeric scale.
   - B. K-Means will automatically convert both variables to the same scale.
   - C. The algorithm will become supervised because income is numeric.
   - D. Scaling is never relevant for distance-based algorithms.

---

## Part 4: Essay Questions

Answer the following 2 questions without code. Your answers should be written as short essays.

1. **Data Quality and Model Reliability**  
   Explain why data cleaning, exploratory data analysis, and understanding variable meaning are essential before building any machine learning model. Discuss how poor data quality can affect visualizations, statistics, and model conclusions.

2. **Choosing the Right Analytical Approach**  
   Explain how you would decide between descriptive statistics, visualization, supervised learning, and unsupervised learning when facing a new data problem. Include the role of CRISP-DM in organizing this decision-making process.

---

## Part 3: Basic Python Exercises

Answer the following questions using only basic Python. Do not use Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, or any data analysis library.

1. **Variables and Arithmetic**  
   Write a program that receives three numeric values representing the price of three products. Calculate the total price, apply a discount rule of your choice, and print the final amount. Explain the variables you created and the arithmetic operations used.

2. **Conditional Statements**  
   Write a program that receives a person's age and prints whether the person is a child, teenager, adult, or senior. Define your own age ranges and explain why the order of the conditions matters.

3. **Loops with Lists**  
   Create a list with at least 10 numbers. Use a loop to calculate the sum of the even numbers and the sum of the odd numbers separately. Print both results and explain how your condition separates even and odd values.

4. **String Manipulation**  
   Write a program that receives a sentence and counts how many words it contains. Then print the same sentence in uppercase, lowercase, and title case. Explain which string methods you used.

5. **Functions**  
   Create a function called `calculate_area(shape, value1, value2=0)` that calculates the area of a square, rectangle, or triangle depending on the value of `shape`. Include examples calling the function with different shapes.

6. **Dictionaries**  
   Create a dictionary representing a book, with keys for title, author, year, and number of pages. Write code to update one value, add a new key, remove one key, and print the final dictionary. Explain each operation.

7. **Nested Data Structures**  
   Create a list of dictionaries representing at least 5 products in a store. Each product should have name, price, and quantity. Write code to calculate the total value in stock for each product using only loops and basic Python.

8. **Error Handling**  
   Write a program that asks for two numbers and divides the first by the second. Use `try`, `except`, and `finally` to handle invalid input and division by zero. Explain why error handling is useful.

9. **List Comprehension**  
   Create a list of numbers from 1 to 50. Use list comprehension to create a new list containing only numbers divisible by 3 and another list containing the square of each number. Explain how list comprehension differs from a traditional loop.

10. **Basic Algorithmic Thinking**  
   Write a function that receives a list of words and returns the longest word. If there is a tie, return the first longest word found. Explain the logic of your algorithm step by step.

---

## Part 4: Python and Pandas

Answer the following 10 questions using Python and Pandas. Each answer must include code and a written explanation.

1. **Basic Python Data Structures**  
   Create a list of dictionaries representing at least 8 students. Each student must have a name, age, course, and final grade. Write Python code to calculate the average grade, identify the student with the highest grade, and list all students who passed with a grade greater than or equal to 7.0.

2. **Functions and Conditional Logic**  
   Write a function called `classify_grade(grade)` that receives a numeric grade and returns `"Excellent"`, `"Good"`, `"Pass"`, or `"Fail"` based on criteria you define. Apply the function to a list of grades and explain why your thresholds are reasonable.

3. **Creating a DataFrame**  
   Build a Pandas DataFrame with at least 10 rows representing product sales. Include columns for product name, category, unit price, quantity sold, and region. Create a new column called `total_revenue` and explain how it was calculated.

4. **Filtering and Sorting Data**  
   Using a DataFrame of your choice, write code to filter rows based on at least two conditions, such as category and revenue. Then sort the result by a numeric column in descending order. Explain what business question your filtering answers.

5. **Handling Missing Values**  
   Create or load a DataFrame with missing values in at least two columns. Write code to detect missing values, decide whether to remove or fill them, and justify your choice for each column.

6. **Grouping and Aggregation**  
   Given a sales DataFrame, group the data by category and calculate total revenue, average unit price, and total quantity sold. Explain what each aggregation tells you about the dataset.

7. **Working with Dates**  
   Create a DataFrame with a date column representing sales transactions over several months. Convert the column to datetime format, extract month and weekday information, and calculate total revenue by month. Explain any pattern you observe.

8. **Merging DataFrames**  
   Create two DataFrames: one with customer information and another with purchase information. Merge them using a common customer ID. Explain the difference between an inner join and a left join using your example.

9. **Data Cleaning Pipeline**  
   Build a small data cleaning pipeline for a messy DataFrame. Your code should rename columns, remove duplicates, handle missing values, convert data types, and create at least one new useful feature. Explain each step.

10. **Intermediate Pandas Analysis**  
   Use a DataFrame with at least three numeric columns and one categorical column. Write code to calculate summary statistics by group, detect possible outliers using a rule of your choice, and explain how those outliers could affect analysis.

---

## Answer Key

This answer key provides objective answers and reference solutions. For coding and essay questions, equivalent solutions are acceptable if they are correct, readable, and well explained.

### Part 1: Objective Questions

| Question | Answer |
|----------|--------|
| 1 | B |
| 2 | A |
| 3 | A |
| 4 | B |
| 5 | A |
| 6 | A |
| 7 | C |
| 8 | B |
| 9 | A |
| 10 | A |
| 11 | D |
| 12 | A |
| 13 | A |
| 14 | A |
| 15 | A |
| 16 | A |
| 17 | A |
| 18 | A |
| 19 | A |
| 20 | A |
| 21 | A |
| 22 | A |
| 23 | A |
| 24 | A |
| 25 | A |
| 26 | A |
| 27 | A |
| 28 | A |
| 29 | A |
| 30 | A |

### Essay Questions: Reference Answers

1. **Data Quality and Model Reliability**  
   A strong answer should explain that data cleaning and exploratory data analysis are necessary because models learn from the data they receive. Missing values, duplicated records, incorrect types, inconsistent categories, outliers, and poorly understood variables can distort statistics, visualizations, and model behavior. Poor data quality can lead to misleading correlations, biased predictions, unreliable clusters, and wrong business decisions. Before modeling, analysts should inspect distributions, check assumptions, understand variable meaning, and document transformations.

2. **Choosing the Right Analytical Approach**  
   A strong answer should explain that descriptive statistics summarize data, visualizations reveal patterns and anomalies, supervised learning is appropriate when there is a target variable to predict, and unsupervised learning is appropriate when the goal is to discover structure without labels. CRISP-DM helps organize this decision by starting with business understanding, then data understanding, preparation, modeling, evaluation, and deployment. The chosen approach should match the problem goal, data availability, evaluation criteria, and practical constraints.

### Basic Python Exercises: Reference Solutions

1. **Variables and Arithmetic**

```python
product_1 = 25.00
product_2 = 40.00
product_3 = 15.00

total = product_1 + product_2 + product_3
discount_rate = 0.10 if total >= 70 else 0.0
final_amount = total * (1 - discount_rate)

print(f"Total: {total:.2f}")
print(f"Final amount: {final_amount:.2f}")
```

2. **Conditional Statements**

```python
age = 21

if age < 13:
    category = "child"
elif age < 18:
    category = "teenager"
elif age < 60:
    category = "adult"
else:
    category = "senior"

print(category)
```

3. **Loops with Lists**

```python
numbers = [3, 8, 11, 14, 19, 22, 27, 30, 35, 40]
even_sum = 0
odd_sum = 0

for number in numbers:
    if number % 2 == 0:
        even_sum += number
    else:
        odd_sum += number

print(even_sum)
print(odd_sum)
```

4. **String Manipulation**

```python
sentence = "python is useful for problem solving"
words = sentence.split()

print(len(words))
print(sentence.upper())
print(sentence.lower())
print(sentence.title())
```

5. **Functions**

```python
def calculate_area(shape, value1, value2=0):
    shape = shape.lower()

    if shape == "square":
        return value1 ** 2
    if shape == "rectangle":
        return value1 * value2
    if shape == "triangle":
        return (value1 * value2) / 2

    return None

print(calculate_area("square", 4))
print(calculate_area("rectangle", 4, 6))
print(calculate_area("triangle", 4, 6))
```

6. **Dictionaries**

```python
book = {
    "title": "Clean Code",
    "author": "Robert C. Martin",
    "year": 2008,
    "pages": 464,
}

book["year"] = 2009
book["genre"] = "Programming"
book.pop("pages")

print(book)
```

7. **Nested Data Structures**

```python
products = [
    {"name": "Notebook", "price": 8.50, "quantity": 10},
    {"name": "Pen", "price": 2.00, "quantity": 30},
    {"name": "Backpack", "price": 120.00, "quantity": 4},
    {"name": "Pencil", "price": 1.50, "quantity": 50},
    {"name": "Eraser", "price": 3.00, "quantity": 20},
]

for product in products:
    stock_value = product["price"] * product["quantity"]
    print(product["name"], stock_value)
```

8. **Error Handling**

```python
try:
    number_1 = float(input("First number: "))
    number_2 = float(input("Second number: "))
    result = number_1 / number_2
    print(result)
except ValueError:
    print("Invalid numeric input.")
except ZeroDivisionError:
    print("Cannot divide by zero.")
finally:
    print("Execution finished.")
```

9. **List Comprehension**

```python
numbers = list(range(1, 51))
divisible_by_3 = [number for number in numbers if number % 3 == 0]
squares = [number ** 2 for number in numbers]

print(divisible_by_3)
print(squares)
```

10. **Basic Algorithmic Thinking**

```python
def longest_word(words):
    longest = words[0]

    for word in words[1:]:
        if len(word) > len(longest):
            longest = word

    return longest

print(longest_word(["data", "python", "visualization", "model"]))
```

### Python and Pandas Exercises: Reference Solutions

1. **Basic Python Data Structures**

```python
students = [
    {"name": "Ana", "age": 20, "course": "Data Science", "final_grade": 8.5},
    {"name": "Bruno", "age": 22, "course": "Data Science", "final_grade": 6.8},
    {"name": "Carla", "age": 21, "course": "AI", "final_grade": 9.2},
    {"name": "Diego", "age": 23, "course": "AI", "final_grade": 7.4},
    {"name": "Eva", "age": 20, "course": "Software", "final_grade": 5.9},
    {"name": "Felipe", "age": 24, "course": "Software", "final_grade": 8.1},
    {"name": "Giulia", "age": 19, "course": "Data Science", "final_grade": 7.0},
    {"name": "Hugo", "age": 22, "course": "AI", "final_grade": 6.5},
]

average_grade = sum(student["final_grade"] for student in students) / len(students)
best_student = max(students, key=lambda student: student["final_grade"])
passed_students = [student for student in students if student["final_grade"] >= 7.0]

print(average_grade)
print(best_student)
print(passed_students)
```

2. **Functions and Conditional Logic**

```python
def classify_grade(grade):
    if grade >= 9:
        return "Excellent"
    if grade >= 7:
        return "Good"
    if grade >= 5:
        return "Pass"
    return "Fail"

grades = [9.5, 8.0, 6.2, 4.8]
labels = [classify_grade(grade) for grade in grades]
print(labels)
```

3. **Creating a DataFrame**

```python
import pandas as pd

sales = pd.DataFrame({
    "product": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    "category": ["Tech", "Tech", "Food", "Food", "Home", "Home", "Tech", "Food", "Home", "Tech"],
    "unit_price": [100, 150, 20, 15, 80, 60, 200, 25, 90, 120],
    "quantity_sold": [3, 2, 10, 15, 4, 5, 1, 8, 3, 2],
    "region": ["North", "South", "North", "East", "West", "South", "East", "West", "North", "South"],
})

sales["total_revenue"] = sales["unit_price"] * sales["quantity_sold"]
print(sales)
```

4. **Filtering and Sorting Data**

```python
filtered = sales[(sales["category"] == "Tech") & (sales["total_revenue"] >= 240)]
filtered = filtered.sort_values("total_revenue", ascending=False)
print(filtered)
```

5. **Handling Missing Values**

```python
df = pd.DataFrame({
    "name": ["A", "B", "C", "D"],
    "age": [20, None, 22, 21],
    "score": [8.5, 7.0, None, 9.0],
})

print(df.isna().sum())
df["age"] = df["age"].fillna(df["age"].median())
df["score"] = df["score"].fillna(df["score"].mean())
print(df)
```

6. **Grouping and Aggregation**

```python
summary = sales.groupby("category").agg(
    total_revenue=("total_revenue", "sum"),
    average_unit_price=("unit_price", "mean"),
    total_quantity_sold=("quantity_sold", "sum"),
)

print(summary)
```

7. **Working with Dates**

```python
transactions = pd.DataFrame({
    "date": ["2026-01-05", "2026-01-20", "2026-02-10", "2026-02-18", "2026-03-03"],
    "revenue": [200, 150, 300, 250, 400],
})

transactions["date"] = pd.to_datetime(transactions["date"])
transactions["month"] = transactions["date"].dt.month
transactions["weekday"] = transactions["date"].dt.day_name()
monthly_revenue = transactions.groupby("month")["revenue"].sum()

print(transactions)
print(monthly_revenue)
```

8. **Merging DataFrames**

```python
customers = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "name": ["Ana", "Bruno", "Carla"],
})

purchases = pd.DataFrame({
    "customer_id": [1, 1, 2, 4],
    "purchase_value": [100, 50, 80, 120],
})

inner_result = customers.merge(purchases, on="customer_id", how="inner")
left_result = customers.merge(purchases, on="customer_id", how="left")

print(inner_result)
print(left_result)
```

9. **Data Cleaning Pipeline**

```python
messy = pd.DataFrame({
    "Customer Name ": ["Ana", "Ana", "Bruno", None],
    "Age": ["20", "20", "22", "21"],
    "Purchase": [100, 100, None, 80],
})

clean = messy.rename(columns=lambda column: column.strip().lower().replace(" ", "_"))
clean = clean.drop_duplicates()
clean["customer_name"] = clean["customer_name"].fillna("Unknown")
clean["age"] = clean["age"].astype(int)
clean["purchase"] = clean["purchase"].fillna(clean["purchase"].median())
clean["is_adult"] = clean["age"] >= 18

print(clean)
```

10. **Intermediate Pandas Analysis**

```python
df = pd.DataFrame({
    "group": ["A", "A", "A", "B", "B", "B"],
    "score": [70, 75, 300, 80, 85, 90],
    "hours": [5, 6, 7, 4, 5, 6],
    "attempts": [1, 2, 1, 2, 2, 3],
})

summary = df.groupby("group")[["score", "hours", "attempts"]].describe()

q1 = df["score"].quantile(0.25)
q3 = df["score"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = df[(df["score"] < lower_bound) | (df["score"] > upper_bound)]

print(summary)
print(outliers)
```
