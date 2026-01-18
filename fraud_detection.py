import pandas as pd

# Load dataset
df = pd.read_csv("creditcard.csv")

# Basic inspection
print(df.head())
print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(df["Class"].value_counts())


######### DATA PREPROCESSING ##################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)



###################### BASELINE MODEL — LOGISTIC REGRESSION ###############
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Train model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



#################### SECOND MODEL — RANDOM FOREST #####################
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Results")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))



############ ANOMALY DETECTION — ISOLATION FOREST ################
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    contamination=0.0017,  # approx fraud rate
    random_state=42
)

iso.fit(X_train)

y_pred_iso = iso.predict(X_test)
y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]

print("Isolation Forest Results")
print(confusion_matrix(y_test, y_pred_iso))
print(classification_report(y_test, y_pred_iso))


############# ROC CURVE ################

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get probabilities
y_prob_lr = lr.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_lr)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Fraud Detection")
plt.legend()
plt.show()

############# COMPARE MODELS ###############################
from sklearn.metrics import precision_score, recall_score, f1_score

models = {
    "Logistic Regression": y_pred,
    "Random Forest": y_pred_rf,
    "Isolation Forest": y_pred_iso
}

print("\nModel Comparison:")
for name, preds in models.items():
    print(f"{name}")
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1-score:", f1_score(y_test, preds))
    print("-" * 30)
