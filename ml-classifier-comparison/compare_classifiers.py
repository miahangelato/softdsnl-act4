# compare_classifiers.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ==== Load dataset ====
df = pd.read_csv('dataset.csv')
df = df.drop(df.columns[0], axis=1)  # Drop the index column

# Set target column explicitly
target_col = 'Type'

# Encode target if categorical
target_encoder = LabelEncoder()
if df[target_col].dtype == 'object':
    df[target_col] = target_encoder.fit_transform(df[target_col])

X = df.drop([target_col], axis=1)
y = df[target_col]

# Encode categorical features (object or string columns)
encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

# ==== Normalize features ====
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ==== Train-test split ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==== Models ====
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

results = {}
os.makedirs("visualizations", exist_ok=True)

# ==== Train & evaluate ====
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# ==== Accuracy bar chart ====
plt.figure()
plt.bar(results.keys(), results.values(), color='skyblue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/accuracy_bar.png")

# ==== Confusion matrix (Random Forest) ====
rf_model = models["Random Forest"]
rf_preds = rf_model.predict(X_test)
cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.savefig("visualizations/confusion_matrix_rf.png")

# ==== Decision tree visualization ====
plt.figure(figsize=(12, 8))
plot_tree(models["Decision Tree"],
          feature_names=X.columns,
          class_names=[str(c) for c in np.unique(y)],
          filled=True)
plt.title("Decision Tree Visualization")
plt.savefig("visualizations/decision_tree.png")

# ==== Decision boundary for Logistic Regression and KNN (first 2 features) ====
from matplotlib.colors import ListedColormap

X_vis = X.iloc[:, :2]
y_vis = y

models_vis = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier()
}

for name, model in models_vis.items():
    model.fit(X_vis, y_vis)
    h = 0.02
    x_min, x_max = X_vis.iloc[:, 0].min() - 1, X_vis.iloc[:, 0].max() + 1
    y_min, y_max = X_vis.iloc[:, 1].min() - 1, X_vis.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X_vis.iloc[:, 0], X_vis.iloc[:, 1], c=y_vis, edgecolor="k",
                cmap=ListedColormap(["red", "green", "blue"]))
    plt.title(f"Decision Boundary ({name}, first 2 features)")
    plt.xlabel(X_vis.columns[0])
    plt.ylabel(X_vis.columns[1])
    plt.tight_layout()
    plt.savefig(f"visualizations/decision_boundary_{name.lower().replace(' ', '_')}.png")
    plt.close()

# ==== Save all models, encoders, and scaler ====
for name, model in models.items():
    filename = name.lower().replace(" ", "_") + "_model.pkl"
    joblib.dump(model, filename)
    print(f"✅ Saved {name} as {filename}")

# Save encoders for each categorical column
joblib.dump(encoders, "feature_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("\n✅ All visualizations and objects saved in 'visualizations/' folder.")
