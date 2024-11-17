

# Student Performance Prediction

This project leverages machine learning to predict the **Performance Index** of students based on several factors, such as **hours studied**, **previous scores**, **extracurricular activities**, **sleep hours**, and **practice of sample question papers**. We have used a **Linear Regression** model to predict the student's performance based on these features.

---

### **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Installation Requirements](#installation-requirements)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [Prediction Example](#prediction-example)
8. [Usage](#usage)
9. [Conclusion](#conclusion)

---

### **Project Overview**

This project aims to predict a student’s **Performance Index** based on various input features. We use **Linear Regression** for regression tasks where the target variable is continuous. The dataset consists of 10,000 student records and contains 6 columns:

- **Hours Studied**
- **Previous Scores**
- **Extracurricular Activities** (Yes/No)
- **Sleep Hours**
- **Sample Question Papers Practiced**
- **Performance Index** (target variable)

---

### **Dataset Description**

The dataset contains the following columns:

1. **Hours Studied**: The number of hours a student spends studying.
2. **Previous Scores**: The student's previous scores on exams.
3. **Extracurricular Activities**: Whether the student participates in extracurricular activities (encoded as 0 = No, 1 = Yes).
4. **Sleep Hours**: The number of hours a student sleeps per day.
5. **Sample Question Papers Practiced**: The number of question papers the student practices before exams.
6. **Performance Index**: The target variable representing the student's overall performance.

#### **Data Preview**
| Hours Studied | Previous Scores | Extracurricular Activities | Sleep Hours | Sample Question Papers Practiced | Performance Index |
|---------------|-----------------|----------------------------|-------------|----------------------------------|-------------------|
| 7             | 99              | Yes                        | 9           | 1                                | 91.0              |
| 4             | 82              | No                         | 4           | 2                                | 65.0              |
| 8             | 51              | Yes                        | 7           | 2                                | 45.0              |
| 5             | 52              | Yes                        | 5           | 2                                | 36.0              |
| 7             | 75              | No                         | 8           | 5                                | 66.0              |

---

### **Installation Requirements**

To run this project, you need to install the following Python libraries:

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`

You can install these dependencies using pip:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

---

### **Data Preprocessing**

In this step:
- **Label Encoding** is applied to the `Extracurricular Activities` column to convert categorical values (Yes/No) to numerical values (1/0) using **LabelEncoder** from **scikit-learn**.
- **Train-Test Split**: The data is divided into training and testing datasets using **train_test_split** from **scikit-learn**.

```python
from sklearn.preprocessing import LabelEncoder

# Apply Label Encoding to 'Extracurricular Activities'
label_encoder = LabelEncoder()
stud_perf["Extracurricular Activities"] = label_encoder.fit_transform(stud_perf["Extracurricular Activities"])
```

---

### **Modeling**

We use **Linear Regression** to predict the **Performance Index** of a student based on the features:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = stud_perf.drop("Performance Index", axis=1)
y = stud_perf["Performance Index"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the model
lnrg = LinearRegression()
lnrg.fit(X_train, y_train)
```

---

### **Evaluation**

After training the model, we evaluate its performance using the following metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **R² Score**: Indicates the proportion of variance in the target variable that can be explained by the features.
- **Percentage Error**: The percentage of the MSE relative to the range of the target variable.

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = lnrg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) * 100
percentage_error = (mse / range_y_test) * 100

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2:.2f}%")
print(f"Percentage Error of MSE relative to range of y test: {percentage_error:.2f}%")
```

---

### **Prediction Example**

You can use the trained model to predict the performance index for a new student by providing input features (such as hours studied, previous scores, extracurricular activity, etc.):

```python
custom_input = [[10, 96, 1, 7, 8]]
predicted_performance = lnrg.predict(custom_input)
print("Performance of student will be: ", predicted_performance)
```

**Output:**
```
Performance of student will be: [97.69720803]
```

---

### **Usage**

1. Clone the repository or download the **Student_Performance.csv** dataset.
2. Install required libraries.
3. Run the `student_performance_prediction.py` script or use Jupyter Notebook to explore the code.
4. Input new data to predict the performance index for students.

---

### **Conclusion**

This project demonstrates how to use machine learning techniques to predict student performance based on various features. The **Linear Regression** model is highly accurate (with an R² score of 98.88%) and can be further improved by exploring other algorithms or refining the dataset.

---

