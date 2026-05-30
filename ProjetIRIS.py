import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Charger
df = pd.read_csv("iris.csv")
df = df.drop(columns=["Id"])  # ← colonne inutile supprimée
print(f"Dataset chargé : {df.shape[0]} fleurs, {df.shape[1]} colonnes")

# 2. Visualiser
couleurs = {"Iris-setosa": "steelblue", "Iris-versicolor": "coral", "Iris-virginica": "green"}
for espece, couleur in couleurs.items():
    subset = df[df["Species"] == espece]
    plt.scatter(subset["PetalLengthCm"], subset["PetalWidthCm"], color=couleur, label=espece, s=60)
plt.title("Pétales — Longueur vs Largeur", fontsize=14)
plt.xlabel("Longueur pétale (cm)")
plt.ylabel("Largeur pétale (cm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Préparer
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entraîner
modele = RandomForestClassifier(n_estimators=100, random_state=42)
modele.fit(X_train, y_train)

# 5. Évaluer
predictions = modele.predict(X_test)
print(f"Précision : {accuracy_score(y_test, predictions) * 100:.1f}%")

importances = pd.DataFrame({
    "feature":    X.columns,
    "importance": modele.feature_importances_
}).sort_values("importance", ascending=False)
print("\n--- Importance des features ---")
print(importances)

# Visualiser importance
plt.figure(figsize=(8, 4))
plt.barh(importances["feature"], importances["importance"], color="steelblue")
plt.title("Quelle mesure compte le plus ?", fontsize=14)
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 6. Prédire
nouvelle_fleur = pd.DataFrame({
    "SepalLengthCm": [5.5],
    "SepalWidthCm":  [3.0],
    "PetalLengthCm": [4.5],
    "PetalWidthCm":  [1.3]
})
prediction = modele.predict(nouvelle_fleur)
proba = modele.predict_proba(nouvelle_fleur)

print(f"\nEspèce prédite : {prediction[0]}")
print("Probabilités :")
for espece, prob in zip(modele.classes_, proba[0]):
    print(f"  {espece} : {prob * 100:.1f}%")

# comparer prediction à la réalité
resultats = pd.DataFrame({
    "réel":   list(y_test)[:5],
    "prédit": list(predictions)[:5]
})
print(resultats)

# prediction nouvelle fleur
print(f"\nNouvelle fleur : {nouvelle_fleur.iloc[0].to_dict()}")
print(f"Espèce prédite : {prediction[0]}")
