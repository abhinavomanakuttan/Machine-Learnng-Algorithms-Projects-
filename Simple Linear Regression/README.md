# 📈 Simple Linear Regression – Height vs Weight Prediction

This project demonstrates the implementation of **Simple Linear Regression** using Python and Scikit-Learn.
The model predicts a person’s **Height** based on their **Weight** and visualizes relationships, predictions, and residuals.

This project is ideal for beginners learning **machine learning fundamentals, regression modeling, and data visualization**.

---

## 🚀 Project Overview

* 📊 Load and explore the dataset (`height-weight.csv`)
* 📈 Visualize the relationship between height and weight
* 🔀 Split data into training and testing sets
* 🤖 Train a Linear Regression model
* 📏 Evaluate predictions and residuals
* 📉 Visualize residual distribution and error behavior

---

## 🧰 Technologies Used

* **Python**
* **Pandas** – Data handling
* **NumPy** – Numerical computation
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-Learn** – Machine Learning model

---

## 📂 Dataset

**File:** `height-weight.csv`

The dataset contains:

* **Weight** → Independent variable (Feature)
* **Height** → Dependent variable (Target)

Sample format:

| Weight | Height |
| ------ | ------ |
| 45     | 150    |
| 60     | 165    |
| 72     | 175    |

---

## ⚙️ Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/simple-linear-regression.git
```

2. Navigate to the project directory:

```bash
cd simple-linear-regression
```

3. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Open the notebook:

```bash
jupyter notebook "Simple Linear Regression.ipynb"
```

---

## 📝 Workflow

1. **Import Libraries**

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```

2. **Load Dataset**

```python
df = pd.read_csv('height-weight.csv')
```

3. **Data Visualization**

* Scatter plot between Weight and Height.

4. **Feature Selection**

```python
X = df[['Weight']]
y = df['Height']
```

5. **Train-Test Split**

```python
from sklearn.model_selection import train_test_split
```

6. **Model Training**

```python
from sklearn.linear_model import LinearRegression
```

7. **Prediction & Evaluation**

* Generate predictions
* Calculate residuals

8. **Residual Analysis**

* Distribution plot
* Scatter plot of residuals vs predictions

---

## 📊 Outputs

✔️ Scatter plot of Height vs Weight
✔️ Regression predictions
✔️ Residual distribution plot
✔️ Error visualization

These plots help analyze:

* Model accuracy
* Linearity assumption
* Error distribution

---

## 🎯 Learning Objectives

* Understand Simple Linear Regression
* Learn feature-target separation
* Practice data visualization
* Learn model training and prediction
* Interpret residuals and errors

---

## 👨‍💻 Author

**Abhinav Omanakuttan**
B.Tech – Artificial Intelligence & Data Science
Aspiring Data Scientist | Machine Learning Enthusiast

---
