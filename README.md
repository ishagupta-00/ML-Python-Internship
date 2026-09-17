<div align="center">

# 📊 Student Result Prediction

### Complete ML Project Notes — NumPy → Pandas → Data Cleaning → EDA → Scaling → Model Training → Evaluation

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Fundamentals-013243?logo=numpy\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas\&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-F7931E?logo=scikit-learn\&logoColor=white)
![Status](https://img.shields.io/badge/Notes-Exam%20Ready-2E7D32)

*Complete notes and practical workflow covering NumPy, Pandas, Data Cleaning, EDA, Feature Scaling, Model Training, and Evaluation.*

</div>

---

## 📑 Table of Contents

1. [Day 1 — ML Basics & Your First NumPy Arrays](#day-1--ml-basics--your-first-numpy-arrays)
2. [Day 2 — Reshaping, Axis & Boolean Logic in NumPy](#day-2--reshaping-axis--boolean-logic-in-numpy)
3. [Day 3 — Pandas Basics](#day-3--pandas-basics)
4. [Day 4 — Data Cleaning](#day-4--data-cleaning)
5. [Day 5 — Exploring the Data (EDA)](#day-5--exploring-the-data-eda)
6. [Day 6 — Scaling & Train/Test Split](#day-6--scaling--traintest-split)
7. [Day 7 — Training the Model](#day-7--training-the-model)
8. [Day 8 — Confusion Matrix & Metrics](#day-8--confusion-matrix--metrics)
9. [Quick Reference — All Commands](#-quick-reference--all-commands)

---

# Day 1 — ML Basics & Your First NumPy Arrays

## 1.1 What is Machine Learning?

Machine Learning teaches computers to learn patterns from data instead of requiring a human to write every rule explicitly.

| Approach                | Formula                                          |
| ----------------------- | ------------------------------------------------ |
| Traditional Programming | Rules + Data → Output                            |
| Machine Learning        | Data + Expected Output → Learned Pattern (Model) |

> 💡 **In simple words:** An ML model learns patterns from examples and then uses those patterns to make predictions on new data.

---

## 1.2 Three Types of Machine Learning

| Type              | How it learns                               | Example                                      |
| ----------------- | ------------------------------------------- | -------------------------------------------- |
| **Supervised**    | From labelled data (input + correct answer) | Predicting house price from area & location  |
| **Unsupervised**  | Finds hidden patterns in unlabeled data     | Grouping customers by shopping behavior      |
| **Reinforcement** | Trial, error, and rewards                   | A game agent that improves by scoring points |

---

## 1.3 Why NumPy?

NumPy ("Numerical Python") is a library for numerical computing in Python. It is widely used as a foundation for data science and machine learning.

```python
nums = [10, 20, 30]
doubled = [x * 2 for x in nums]

arr = np.array([10, 20, 30])
doubled = arr * 2

print(doubled)
# [20 40 60]
```

> 💡 **Vectorization** means applying an operation to a complete NumPy array without writing an explicit loop.

---

## 1.4 Your First Array & Its Attributes

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.ndim     # 2
arr.shape    # (2, 3)
arr.size     # 6
arr.dtype    # int64
```

| Attribute | Meaning                   |
| --------- | ------------------------- |
| `ndim`    | Number of dimensions      |
| `shape`   | Size along each dimension |
| `size`    | Total number of elements  |
| `dtype`   | Data type of the elements |

---

## 1.5 Indexing & Slicing

Python indexing starts from `0`.

```python
arr = np.array([10, 20, 30, 40, 50])

arr[0]      # 10
arr[-1]     # 50
arr[1:4]    # [20 30 40]
arr[::2]    # [10 30 50]
```

> 💡 In slicing, the start index is included and the end index is excluded.

---

## 1.6 Vectorized Arithmetic

```python
a = np.array([10, 20, 30])
b = np.array([1, 2, 3])

a + b
# [11 22 33]

a * b
# [10 40 90]
```

> 💡 Operations are performed element-by-element at the same positions.

---

# Day 2 — Reshaping, Axis & Boolean Logic in NumPy

## 2.1 Reshaping Arrays

`reshape()` changes the shape of an array without changing its data.

```python
arr = np.arange(1, 13)

arr.reshape(3, 4)      # OK
arr.reshape(5, 2)      # ERROR

arr.reshape(3, -1)     # NumPy calculates the second dimension
arr.reshape(-1)        # Converts to 1D
```

> ⚠️ **Golden Rule:** The new dimensions must contain the same total number of elements.

For example:

`3 × 4 = 12`

So an array with 12 elements can be reshaped into `(3, 4)`.

---

## 2.2 `flatten()` vs `ravel()`

|                                      | `flatten()` | `ravel()`            |
| ------------------------------------ | ----------- | -------------------- |
| Returns                              | A new copy  | A view when possible |
| Original affected by editing result? | No          | May be               |
| Memory use                           | More        | Usually less         |

---

## 2.3 Understanding Axis

For a 2D array:

| Axis     | Direction          | Result                |
| -------- | ------------------ | --------------------- |
| `axis=0` | Down the rows      | One result per column |
| `axis=1` | Across the columns | One result per row    |

```python
marks = np.array([
    [85, 78, 92],
    [67, 88, 74],
    [90, 95, 88]
])

marks.mean()
# 84.11...

marks.mean(axis=0)
# [80.67 87.   84.67]

marks.mean(axis=1)
# [85.   76.33 91.  ]
```

> 💡 `axis=0` works down the rows, while `axis=1` works across the columns.

---

## 2.4 Aggregation Functions

| Function         | Meaning                     |
| ---------------- | --------------------------- |
| `np.sum(arr)`    | Sum of all elements         |
| `np.mean(arr)`   | Average                     |
| `np.median(arr)` | Middle value                |
| `np.min(arr)`    | Smallest value              |
| `np.max(arr)`    | Largest value               |
| `np.std(arr)`    | Standard deviation / spread |

---

## 2.5 Comparison Operators & Boolean Masking

```python
arr = np.array([4, 9, 2, 10, 6])

arr > 5
# [False  True False  True  True]

arr[arr > 5]
# [ 9 10  6]

arr[(arr > 3) & (arr < 9)]
# [4 6]

arr[(arr < 5) | (arr > 9)]
# [ 4  2 10]
```

> ⚠️ Use parentheses around each condition and use `&` / `|` for element-wise conditions.

---

## 2.6 `np.where()` — Conditional Selection

```python
arr = np.array([4, 9, 2, 10, 6])

np.where(arr > 5, "High", "Low")
# ['Low' 'High' 'Low' 'High' 'High']

np.where(arr > 5, arr, 0)
# [ 0  9  0 10  6]
```

> 💡 `np.where(condition, value_if_true, value_if_false)` applies a condition element-by-element.

---

## 2.7 Random Module

```python
np.random.randint(1, 100, 5)
# 5 random integers

np.random.rand(4)
# 4 random decimals between 0 and 1
```

---

# Day 3 — Pandas Basics

## 3.1 Pandas Series

A **Series** is a one-dimensional labelled data structure.

```python
import pandas as pd

marks = pd.Series([85, 78, 92, 67], name="Math")

print(marks)
```

Output:

```text
0    85
1    78
2    92
3    67
Name: Math, dtype: int64
```

> 💡 A Series can be thought of as one column of a table.

---

## 3.2 Series Indexing & Attributes

```python
marks[0]
# 85

marks.values
# [85 78 92 67]

marks.index
# RangeIndex(start=0, stop=4, step=1)

marks.mean()
# 80.5
```

---

## 3.3 Introducing the DataFrame

A **DataFrame** is a two-dimensional table made up of multiple Series.

| Part    | Meaning                 |
| ------- | ----------------------- |
| Rows    | Individual records      |
| Columns | Attributes / features   |
| Index   | Labels identifying rows |

> 💡 A DataFrame is similar to an Excel sheet or SQL table.

---

## 3.4 Creating a DataFrame

```python
data = {
    "Name": ["Aman", "Riya", "Karan", "Divya", "Sahil"],
    "Math": [85, 67, 90, 55, 78],
    "Science": [78, 88, 95, 60, 82],
    "English": [92, 74, 88, 65, 79]
}

df = pd.DataFrame(data)
```

---

## 3.5 Reading CSV Files

```python
df = pd.read_csv("students.csv")

print(df.head(2))
```

> 💡 `read_csv()` reads a CSV file and converts it into a Pandas DataFrame.

---

## 3.6 Exploring the Dataset

```python
df.head()
df.tail(2)
df.shape
df.columns
df.info()
df.describe()
```

| Command         | Purpose                             |
| --------------- | ----------------------------------- |
| `df.head()`     | First 5 rows                        |
| `df.tail()`     | Last 5 rows                         |
| `df.shape`      | Number of rows and columns          |
| `df.columns`    | Column names                        |
| `df.info()`     | Data types and non-null information |
| `df.describe()` | Numeric summary statistics          |

---

## 3.7 Selecting Columns & Rows

```python
df["Math"]
# Single column -> Series

df[["Name", "Math"]]
# Multiple columns -> DataFrame

df.loc[0]
# Row by index label

df.loc[0:2]
# Labels 0, 1, 2

df.iloc[0]
# Row by position

df.iloc[0:2]
# Positions 0 and 1
```

> 💡 `loc` uses labels, while `iloc` uses integer positions.

---

## 3.8 Adding Columns & Sorting

```python
df["Total"] = df["Math"] + df["Science"] + df["English"]

df.sort_values("Total")

df.sort_values("Total", ascending=False)
```

---

# Day 4 — Data Cleaning

## 4.1 Meet the Dataset

### Student Result Prediction Dataset

**File:** `student_performance_large.csv`

**Rows:** 300
**Columns:** 8

| Column                  | Meaning                         |
| ----------------------- | ------------------------------- |
| `Student_ID`            | Unique ID of the student        |
| `Hours_Studied`         | Average daily study hours       |
| `Attendance`            | Attendance percentage           |
| `Previous_Score`        | Previous semester score         |
| `Assignments_Completed` | Number of completed assignments |
| `Sleep_Hours`           | Average daily sleep hours       |
| `Final_Marks`           | Final exam marks                |
| `Result`                | Pass / Fail target              |

---

## 4.2 Loading the Dataset

```python
from google.colab import files

uploaded = files.upload()

import pandas as pd
import numpy as np

df = pd.read_csv("student_performance_large.csv")

df.shape
# (300, 8)
```

> 💡 `shape` confirms the number of rows and columns after loading the dataset.

---

## 4.3 First Look at the Data

```python
df.head()
df.columns
```

---

## 4.4 Finding Missing Values

```python
df.isnull().sum()
```

### Missing Values Found

There were **20 missing values in total**.

| Column                  | Missing Values |
| ----------------------- | -------------: |
| `Hours_Studied`         |              4 |
| `Attendance`            |              4 |
| `Previous_Score`        |              4 |
| `Assignments_Completed` |              4 |
| `Sleep_Hours`           |              4 |
| `Student_ID`            |              0 |
| `Final_Marks`           |              0 |
| `Result`                |              0 |

---

## 4.5 Handling Missing Values

```python
df = df.fillna(df.mean(numeric_only=True))
```

Mean values used:

| Column                  |  Mean |
| ----------------------- | ----: |
| `Hours_Studied`         |  4.89 |
| `Attendance`            | 70.15 |
| `Previous_Score`        | 64.72 |
| `Assignments_Completed` |  5.06 |
| `Sleep_Hours`           |  6.48 |

Check again:

```python
df.isnull().sum().sum()
# 0
```

> 💡 All missing values have now been handled.

---

## 4.6 Fixing Data Types

After mean filling, `Assignments_Completed` became a float.

```python
df["Assignments_Completed"] = (
    df["Assignments_Completed"]
    .round()
    .astype(int)
)
```

> 💡 Since assignments are counted as whole numbers, the column should use integers.

---

## 4.7 Checking Duplicate Rows

```python
df.duplicated().sum()
# 0
```

If duplicates existed:

```python
df = df.drop_duplicates().reset_index(drop=True)
```

---

## 4.8 Sanity Check with `describe()`

```python
df.describe()
```

This helps check whether the values are within reasonable ranges.

For example:

* `Attendance` should normally be within 0–100.
* `Hours_Studied` should not be negative.
* Assignment counts should stay within their expected range.

---

## 4.9 Splitting Features and Target

```python
feature_cols = [
    "Hours_Studied",
    "Attendance",
    "Previous_Score",
    "Assignments_Completed",
    "Sleep_Hours"
]

X = df[feature_cols]
y = df["Result"]

X.shape, y.shape
# ((300, 5), (300,))
```

### Meaning

**X = Features / inputs**

**y = Target / output**

> 💡 The model learns from `X` and tries to predict `y`.

---

## 4.10 ⚠️ Avoiding Data Leakage

`Result` is derived from `Final_Marks`.

Therefore, using `Final_Marks` as an input feature would reveal information that directly determines the target.

### Wrong

```python
X = df[["Hours_Studied", "Final_Marks"]]
```

### Correct

```python
X = df[feature_cols]
```

> ⚠️ **Data leakage** happens when information that should not be available to the model is accidentally included during training.

---

## 4.11 Saving the Clean Dataset

```python
df.to_csv("student_performance_clean.csv", index=False)
```

In Google Colab:

```python
from google.colab import files

files.download("student_performance_clean.csv")
```

---

## 4.12 Full Data Cleaning Pipeline

```python
df = pd.read_csv("student_performance_large.csv")

df = df.fillna(df.mean(numeric_only=True))

df["Assignments_Completed"] = (
    df["Assignments_Completed"]
    .round()
    .astype(int)
)

df = df.drop_duplicates().reset_index(drop=True)

feature_cols = [
    "Hours_Studied",
    "Attendance",
    "Previous_Score",
    "Assignments_Completed",
    "Sleep_Hours"
]

X = df[feature_cols]
y = df["Result"]

df.to_csv("student_performance_clean.csv", index=False)
```

### Data Cleaning Flow

```text
Load Dataset
     ↓
Check Missing Values
     ↓
Fill Missing Values
     ↓
Fix Data Types
     ↓
Check / Remove Duplicates
     ↓
Separate Features and Target
     ↓
Save Clean Dataset
```

---

# Day 5 — Exploring the Data (EDA)

## 5.1 Where We Left Off

| Column                  | Status                                   |
| ----------------------- | ---------------------------------------- |
| `Student_ID`            | Not used as a feature                    |
| `Hours_Studied`         | Missing values filled                    |
| `Attendance`            | Missing values filled                    |
| `Previous_Score`        | Missing values filled                    |
| `Assignments_Completed` | Filled and converted to integer          |
| `Sleep_Hours`           | Missing values filled                    |
| `Final_Marks`           | Not used as a feature because of leakage |
| `Result`                | Target                                   |

---

## 5.2 What Is EDA?

**EDA = Exploratory Data Analysis**

The goal is to understand the dataset before training a machine learning model.

Questions we may ask:

* Are Pass and Fail students different?
* Which feature is most related to Result?
* Is the dataset balanced?
* Are there unusual values or outliers?

---

## 5.3 Loading the Clean Dataset

```python
from google.colab import files

uploaded = files.upload()

import pandas as pd
import numpy as np

df = pd.read_csv("student_performance_clean.csv")

df.shape
# (300, 8)
```

---

## 5.4 Confirming the Data Is Clean

```python
df.isnull().sum().sum()
# 0

df.duplicated().sum()
# 0

df.dtypes
```

---

## 5.5 Checking Class Balance

```python
df["Result"].value_counts()
```

Output:

```text
Pass    249
Fail     51
```

| Result | Count | Percentage |
| ------ | ----: | ---------: |
| Pass   |   249 |      83.0% |
| Fail   |    51 |      17.0% |

---

## 5.6 Why Class Imbalance Matters

Suppose a model predicts **Pass for every student**.

Because 83% of the students are Pass, such a model could still achieve around 83% accuracy without correctly learning the Fail class.

That is why accuracy alone should not always be the only evaluation metric.

---

## 5.7 Grouping by Result

```python
df.groupby("Result")[
    [
        "Hours_Studied",
        "Attendance",
        "Previous_Score",
        "Assignments_Completed",
        "Sleep_Hours"
    ]
].mean().round(2)
```

| Result | Hours Studied | Attendance | Previous Score | Assignments | Sleep Hours |
| ------ | ------------: | ---------: | -------------: | ----------: | ----------: |
| Fail   |          1.93 |      63.64 |          55.48 |        3.55 |        6.07 |
| Pass   |          5.50 |      71.48 |          66.61 |        5.37 |        6.56 |

> 💡 `groupby("Result").mean()` is useful for comparing average feature values between Pass and Fail students.

---

## 5.8 What the Averages Show

From the averages:

* Pass students have higher average `Hours_Studied`.
* Pass students have higher average `Previous_Score`.
* Pass students have higher average `Attendance`.
* Pass students complete more assignments on average.
* Pass students also have slightly higher average `Sleep_Hours`.

These are descriptive observations from the dataset, not proof of causation.

---

## 5.9 Visualizing Differences

```python
import matplotlib.pyplot as plt

df.groupby("Result")["Hours_Studied"].mean().plot(kind="bar")

plt.title("Average Hours Studied: Pass vs Fail")
plt.ylabel("Hours Studied")
plt.show()
```

> 💡 A bar chart makes average differences easier to see.

---

## 5.10 Looking at Distributions

```python
df["Hours_Studied"].hist(bins=15)

plt.title("Distribution of Hours_Studied")
plt.xlabel("Hours Studied")
plt.ylabel("Frequency")
plt.show()
```

Boxplot:

```python
df.boxplot(column="Attendance", by="Result")

plt.show()
```

> 💡 Histograms show distributions, while boxplots help compare spread and identify possible outliers.

---

## 5.11 Correlation Check

Convert Pass/Fail into numbers:

```python
df["Result_num"] = (
    df["Result"] == "Pass"
).astype(int)
```

Then:

```python
df[
    feature_cols + ["Result_num"]
].corr()["Result_num"]
```

Example correlation values from the dataset:

| Feature                 | Correlation with Result |
| ----------------------- | ----------------------: |
| `Hours_Studied`         |                    0.46 |
| `Assignments_Completed` |                    0.21 |
| `Previous_Score`        |                    0.21 |
| `Attendance`            |                    0.17 |
| `Sleep_Hours`           |                    0.13 |

> 💡 Correlation indicates the strength and direction of a linear relationship between numeric variables. It does not by itself prove causation.

---

## 5.12 Understanding Feature Relationships

In this dataset, `Hours_Studied` has the largest correlation with the numeric Result representation.

Other features also show positive relationships, though weaker.

This information can be useful during EDA, but feature selection should ultimately consider the model, validation results, domain knowledge, and possible leakage.

---

## 5.13 Full EDA Pipeline

```python
df = pd.read_csv("student_performance_clean.csv")

df.isnull().sum().sum()
df.duplicated().sum()

df["Result"].value_counts(normalize=True) * 100

df.groupby("Result")[feature_cols].mean().round(2)

df["Result_num"] = (
    df["Result"] == "Pass"
).astype(int)

df[
    feature_cols + ["Result_num"]
].corr()["Result_num"]
```

---

# Day 6 — Scaling & Train/Test Split

## 6.1 Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Meaning of Parameters

| Parameter         | Meaning                      |
| ----------------- | ---------------------------- |
| `test_size=0.2`   | 20% data for testing         |
| `random_state=42` | Reproducible split           |
| `stratify=y`      | Maintains class distribution |

---

## 6.2 Checking Split Shapes

```python
X_train.shape, X_test.shape
```

Expected:

```text
((240, 5), (60, 5))
```

| Set   | Rows | Percentage |
| ----- | ---: | ---------: |
| Train |  240 |        80% |
| Test  |   60 |        20% |

---

## 6.3 Why Stratify?

The dataset contains:

* 249 Pass
* 51 Fail

Using:

```python
stratify=y
```

helps maintain a similar Pass/Fail ratio in both the training and testing sets.

---

## 6.4 Train/Test Class Distribution

```text
Original:
Pass = 83%
Fail = 17%

Train:
Pass ≈ 83%
Fail ≈ 17%

Test:
Pass ≈ 83%
Fail ≈ 17%
```

---

## 6.5 Why Features Need Scaling?

The features have different numeric ranges.

| Feature                 | Approx. Range |
| ----------------------- | ------------- |
| `Hours_Studied`         | 0–10          |
| `Attendance`            | 0–100         |
| `Previous_Score`        | 30–100        |
| `Assignments_Completed` | 0–10          |
| `Sleep_Hours`           | 4–9           |

When using algorithms that are sensitive to feature scale, normalization or standardization can help.

---

## 6.6 Scaling with `StandardScaler`

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### `fit_transform()`

```python
scaler.fit_transform(X_train)
```

This:

1. Learns the mean and standard deviation from training data.
2. Applies the transformation.

### `transform()`

```python
scaler.transform(X_test)
```

Uses the scaling parameters already learned from training data.

---

## 6.7 ⚠️ Avoiding Scaling Leakage

### Wrong

```python
scaler.fit(X)

X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
```

### Correct

```python
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

> ⚠️ The test set should not be used to calculate the scaling parameters.

---

## 6.8 Checking Scaled Values

```python
X_train_scaled.mean(axis=0).round(2)
```

Approximately:

```text
[0. 0. 0. 0. 0.]
```

And:

```python
X_train_scaled.std(axis=0).round(2)
```

Approximately:

```text
[1. 1. 1. 1. 1.]
```

---

# Day 7 — Training the Model

## 7.1 Training Data

```text
X_train_scaled → (240, 5)
X_test_scaled  → (60, 5)

y_train → 240 labels
y_test  → 60 labels
```

---

## 7.2 Choosing Logistic Regression

For a binary classification problem such as **Pass / Fail**, Logistic Regression is a common starting model.

It can estimate the probability of each class and convert that probability into a classification.

---

## 7.3 Fitting the Model

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)

model.fit(
    X_train_scaled,
    y_train
)
```

> 💡 `fit()` is the training step. The model learns patterns from the training data.

---

## 7.4 Making Predictions

```python
y_pred = model.predict(X_test_scaled)

print(y_pred[:5])
```

Example:

```text
['Pass' 'Pass' 'Fail' 'Pass' 'Pass']
```

> 💡 `predict()` generates class predictions for unseen input data.

---

## 7.5 Checking Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
```

Example:

```text
0.917
```

This means **91.7% of test predictions were correct** for that particular split and model run.

> 💡 Accuracy should be interpreted along with other metrics, especially when the classes are imbalanced.

---

## 7.6 Fail Recall

For this project, recall for the **Fail** class can be important because it measures how many actual Fail students were correctly identified.

Example:

```text
Fail Recall = 0.80
```

This means the model correctly identified **80% of the actual Fail cases** in that evaluation.

---

## 7.7 Looking at Model Coefficients

```python
coefs = pd.Series(
    model.coef_[0],
    index=feature_cols
)

coefs.sort_values(ascending=False)
```

Example coefficients:

| Feature                 | Coefficient |
| ----------------------- | ----------: |
| `Hours_Studied`         |        2.51 |
| `Assignments_Completed` |        1.27 |
| `Previous_Score`        |        0.90 |
| `Attendance`            |        0.73 |
| `Sleep_Hours`           |        0.27 |

> 💡 For this model, a positive coefficient indicates that increasing that feature is associated with a higher model score for the positive class, holding other features constant.

---

## 7.8 Full Training Pipeline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

model = LogisticRegression(random_state=42)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=["Pass", "Fail"]
    )
)

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

---

## 7.9 Predicting a Brand-New Student

A completely new student's data must go through the same preprocessing pipeline.

```python
new_student_scaled = scaler.transform(new_student)

prediction = model.predict(
    new_student_scaled
)

probability = model.predict_proba(
    new_student_scaled
)

print("Prediction:", prediction[0])
print("Probability:", probability[0])
```

> 💡 The new student must have the same features used during model training.

---

## 7.10 `predict()` vs `predict_proba()`

### `predict()`

Returns the predicted class:

```text
Pass
```

or

```text
Fail
```

### `predict_proba()`

Returns class probabilities.

For example:

```text
[0.10, 0.90]
```

This means the model estimated approximately:

```text
Fail = 10%
Pass = 90%
```

The exact order depends on `model.classes_`.

---

# Day 8 — Confusion Matrix & Metrics

## 8.1 What Is a Confusion Matrix?

A confusion matrix compares the **actual class** with the **predicted class**.

It helps show:

* Correct predictions
* False alarms
* Missed cases
* Correctly rejected cases

---

## 8.2 Basic Confusion Matrix Terms

|                     | Predicted Positive  | Predicted Negative  |
| ------------------- | ------------------- | ------------------- |
| **Actual Positive** | True Positive (TP)  | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN)  |

### Meaning

**True Positive (TP):** Predicted positive and actually positive.

**True Negative (TN):** Predicted negative and actually negative.

**False Positive (FP):** Predicted positive but actually negative.

**False Negative (FN):** Predicted negative but actually positive.

---

## 8.3 Example — 100 Emails

|                     | Predicted Spam | Predicted Not Spam |
| ------------------- | -------------: | -----------------: |
| **Actual Spam**     |        45 (TP) |             5 (FN) |
| **Actual Not Spam** |         8 (FP) |            42 (TN) |

---

## 8.4 Evaluation Metrics

| Metric        | Formula                                           | Meaning                              |
| ------------- | ------------------------------------------------- | ------------------------------------ |
| **Accuracy**  | `(TP + TN) / Total`                               | Overall correctness                  |
| **Precision** | `TP / (TP + FP)`                                  | Correctness of positive predictions  |
| **Recall**    | `TP / (TP + FN)`                                  | How many actual positives were found |
| **F1-Score**  | `2 × (Precision × Recall) / (Precision + Recall)` | Balance between precision and recall |

For the example:

```text
TP = 45
TN = 42
FP = 8
FN = 5
```

### Accuracy

```text
(45 + 42) / 100
= 87%
```

### Precision

```text
45 / (45 + 8)
≈ 84.9%
```

### Recall

```text
45 / (45 + 5)
= 90%
```

### F1-Score

```text
≈ 87.3%
```

---

## 8.5 Using `confusion_matrix()`

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Pass", "Fail"]
)

print(cm)
```

---

## 8.6 Using `classification_report()`

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

It provides:

* Precision
* Recall
* F1-score
* Support

for each class.

---

# 📋 Quick Reference — All Commands

| Command                   | What it means                    |
| ------------------------- | -------------------------------- |
| `arr.shape` / `df.shape`  | Dimensions / rows and columns    |
| `arr.ndim`                | Number of dimensions             |
| `arr.reshape(r, c)`       | Change array shape               |
| `arr.flatten()`           | Convert to 1D copy               |
| `arr.ravel()`             | Convert to 1D view when possible |
| `arr.sum(axis=0/1)`       | Sum along an axis                |
| `np.mean()`               | Average                          |
| `np.median()`             | Median                           |
| `np.min()` / `np.max()`   | Minimum / maximum                |
| `np.std()`                | Standard deviation               |
| `arr[arr > x]`            | Boolean filtering                |
| `np.where()`              | Conditional selection            |
| `np.random.randint()`     | Random integers                  |
| `np.random.rand()`        | Random decimals                  |
| `df.head()`               | First rows                       |
| `df.tail()`               | Last rows                        |
| `df.info()`               | Dataset information              |
| `df.describe()`           | Numeric summary                  |
| `df.isnull().sum()`       | Missing values                   |
| `df.fillna()`             | Fill missing values              |
| `df.duplicated().sum()`   | Count duplicates                 |
| `df.drop_duplicates()`    | Remove duplicates                |
| `df.round().astype(int)`  | Round and convert to integer     |
| `df["col"]`               | Select one column                |
| `df[["a","b"]]`           | Select multiple columns          |
| `df.loc[]`                | Select using labels              |
| `df.iloc[]`               | Select using positions           |
| `df.sort_values()`        | Sort rows                        |
| `df.groupby()`            | Group data                       |
| `df.corr()`               | Correlation matrix               |
| `train_test_split()`      | Split dataset                    |
| `test_size=0.2`           | 20% test data                    |
| `random_state=42`         | Reproducible results             |
| `stratify=y`              | Maintain class proportions       |
| `StandardScaler()`        | Standardize features             |
| `fit_transform()`         | Learn parameters and transform   |
| `transform()`             | Apply learned transformation     |
| `LogisticRegression()`    | Classification algorithm         |
| `model.fit()`             | Train model                      |
| `model.predict()`         | Predict classes                  |
| `model.predict_proba()`   | Predict probabilities            |
| `accuracy_score()`        | Accuracy                         |
| `confusion_matrix()`      | TP/FP/FN/TN counts               |
| `classification_report()` | Precision, Recall, F1            |
| `model.coef_`             | Model coefficients               |

---

# 🔄 Complete ML Workflow

```text
Raw Dataset
     ↓
Load Dataset
     ↓
Understand Data
     ↓
Handle Missing Values
     ↓
Fix Data Types
     ↓
Remove Duplicates
     ↓
EDA
     ↓
Select Features and Target
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
Train Model
     ↓
Make Predictions
     ↓
Evaluate Model
     ↓
Use Model on New Data
```

---

<div align="center">

### 📊 Student Result Prediction

**Technologies Used**

`Python` • `NumPy` • `Pandas` • `Matplotlib` • `Scikit-learn`

### Concepts Covered

**NumPy → Pandas → Data Cleaning → EDA → Train/Test Split → Scaling → Logistic Regression → Confusion Matrix → Precision → Recall → F1-Score**

</div>
