import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# --- Repréparer les données ---
df = pd.read_csv("creditcard.csv")
df["hour"] = (df["Time"] / 3600) % 24
df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
df["high_amount"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)
scaler = StandardScaler()
df["amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df = df.drop(columns=["Time", "Amount"])

X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
modele = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss")
modele.fit(X_train_sm, y_train_sm)
predictions = modele.predict(X_test)

# --- Dashboard 3 graphiques ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Système de Détection de Fraude - Résultats", fontsize=16, fontweight="bold")

# 1. Matrice de confusion
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Légitime", "Fraude"],
            yticklabels=["Légitime", "Fraude"],
            ax=axes[0])
axes[0].set_title("Matrice de Confusion")
axes[0].set_ylabel("Réel")
axes[0].set_xlabel("Prédit")

# 2. Importance des features
importances = pd.DataFrame({
    "feature":    X.columns,
    "importance": modele.feature_importances_
}).sort_values("importance", ascending=False).head(10)

axes[1].barh(importances["feature"], importances["importance"], color="steelblue")
axes[1].set_title("Top 10 Features")
axes[1].set_xlabel("Importance")
axes[1].invert_yaxis()

# 3. Distribution fraudes par heure
fraudes = df[df["Class"] == 1]
legitimes = df[df["Class"] == 0]
axes[2].hist(legitimes["hour"], bins=24, alpha=0.5, color="steelblue", label="Légitime", density=True)
axes[2].hist(fraudes["hour"], bins=24, alpha=0.7, color="coral", label="Fraude", density=True)
axes[2].set_title("Fraudes par heure de la journée")
axes[2].set_xlabel("Heure")
axes[2].set_ylabel("Densité")
axes[2].legend()

plt.tight_layout()
plt.savefig("fraud_detection_results.png")
plt.show()
print("Dashboard sauvegardé")