# 🤖 SOFTDSNL Activity 4: Comparing Different Machine Learning Classifiers

## 🎯 Objective

In this activity, you will experiment with multiple machine learning classification models and compare their performance using **accuracy, confusion matrix, and visualizations**.

You will:
- Use an existing dataset (e.g., Iris, Wine, or any dataset with at least 50 rows and 3 features)
- Train **at least 4 different classifiers**
- Compare their performance using various metrics and visual tools
- Visualize at least **one decision boundary** and **one decision tree** (if applicable)

---

## 📚 Classifiers to Use

Choose and compare at least 4 of the following:

- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)

---

## 🛠 Requirements

### ✅ Dataset

You may use a built-in dataset from `sklearn.datasets` (e.g., Iris, Wine, Breast Cancer) or load a CSV with:
- At least **50 rows**
- At least **3 numerical features**
- One target label (classification)

### ✅ Classifier Evaluation

For each classifier, you must:
- Split the dataset using `train_test_split`
- Train the model
- Predict and evaluate using:
  - Accuracy
  - Confusion Matrix
  - Classification Report (optional)

### ✅ Visualizations (Required)

Include **at least 3 visualizations** total:
1. **Bar chart** comparing classifier accuracy
2. **At least 1 decision boundary plot** (for classifiers with 2 features)
3. **At least 1 decision tree plot** (for Decision Tree or Random Forest)

---

## 📁 Recommended Folder Structure

```
ml-classifier-comparison/
├── dataset.csv                  # (if using custom data)
├── compare_classifiers.py       # Main script
├── visualizations/              # Plots and images
│   ├── accuracy_bar.png
│   ├── confusion_matrix_rf.png
│   ├── decision_boundary_knn.png
│   ├── decision_tree.png
├── output_summary.pdf           # 📄 Your report
├── requirements.txt
├── README.md                    # This file
```

---

## 📈 Sample Visuals

### Accuracy Bar Chart

```python
plt.bar(models, accuracies)
plt.title("Classifier Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig("visualizations/accuracy_bar.png")
```

### Decision Tree Plot

```python
plot_tree(trained_decision_tree, feature_names=feature_names, class_names=class_names, filled=True)
plt.savefig("visualizations/decision_tree.png")
```

### Decision Boundary Plot

```python
disp = DecisionBoundaryDisplay.from_estimator(knn_model, X_train[:, :2], response_method="predict")
disp.ax_.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.savefig("visualizations/decision_boundary_knn.png")
```

---

## 🧾 `requirements.txt`

```
pandas
numpy
matplotlib
scikit-learn
```

---

## 💡 Template Classifier Script (`compare_classifiers.py`)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load your dataset
df = pd.read_csv('dataset.csv')

# Encode categorical labels if needed
label_encoder = LabelEncoder()
if df['target'].dtype == 'object':
    df['target'] = label_encoder.fit_transform(df['target'])

X = df.drop('target', axis=1)
y = df['target']

# Normalize features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

results = {}
os.makedirs("visualizations", exist_ok=True)

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

# Accuracy bar chart
plt.figure()
plt.bar(results.keys(), results.values(), color='skyblue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/accuracy_bar.png")

# Confusion matrix for Random Forest
rf_model = models["Random Forest"]
rf_preds = rf_model.predict(X_test)
cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.savefig("visualizations/confusion_matrix_rf.png")

# Decision tree diagram
plt.figure(figsize=(12, 8))
plot_tree(models["Decision Tree"], feature_names=X.columns, class_names=[str(c) for c in np.unique(y)], filled=True)
plt.title("Decision Tree Visualization")
plt.savefig("visualizations/decision_tree.png")

# Decision boundary (if only 2 features)
if X.shape[1] == 2:
    from matplotlib.colors import ListedColormap

    h = 0.02
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = models["KNN"].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor="k", cmap=ListedColormap(["red", "green", "blue"]))
    plt.title("Decision Boundary (KNN)")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.savefig("visualizations/decision_boundary_knn.png")
```

---

## 🔌 Django API

Expose your model via an API:

- Set up Django (`django-admin startproject ml_api_project`)
- Create a view to accept data via `POST` and return a prediction
- Test your API using Postman and include screenshots

---

## 📦 How to Run

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Run the script:

```bash
python compare_classifiers.py
```

---

## 📝 What to Submit

You must submit the following via your LMS or upload system:

1. ✅ A **PDF report** named `output_summary.pdf` that includes:
   - Screenshot of the accuracy bar chart
   - Screenshots of decision boundary and tree visualizations
   - Screenshot of postman results. (At least 3).
   - Summary table comparing classifiers
   - Short reflection on model performance
   - **Link to your GitHub repository fork**

2. ✅ A link to your public GitHub fork containing:
   - `compare_classifiers.py`
   - `requirements.txt`
   - Visualizations folder with PNGs
   - Your custom dataset (if any)
   - `README.md` (this file)

---

## 📋 Sample Evaluation Table

| Classifier        | Accuracy | Notes                      |
|-------------------|----------|----------------------------|
| KNN               | 0.94     | Good but sensitive to k    |
| Decision Tree     | 0.91     | Easy to visualize          |
| Random Forest     | 0.96     | Best overall               |
| SVM               | 0.93     | Works well with small data |

---

## 💯 Grading Rubric (50 pts)

| Criteria                             | Points |
|--------------------------------------|--------|
| At least 4 classifiers used          | 10     |
| Accuracy results summarized          | 10     |
| Confusion matrix for 1+ models       | 5      |
| Decision tree plot (for DT or RF)    | 5      |
| Decision boundary plot (2D)          | 10     |
| Accuracy bar chart                   | 5      |
| PDF report with GitHub link          | 5      |
| **TOTAL**                            | **50** |

---

## 🧾 `requirements.txt`

```txt
scikit-learn
matplotlib
pandas
numpy
```

---

Good luck, and enjoy comparing the brains behind the models! 🧠📊🚀
