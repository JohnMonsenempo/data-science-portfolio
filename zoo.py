import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Charger
df = pd.read_csv("zoo.csv")
print(f"Dataset chargé : {df.shape[0]} animaux, {df.shape[1]} colonnes ")

# 2. Visualiser
couleurs = {1: "steelblue", 2: "coral", 3: "green",
            4: "purple", 5: "orange", 6: "cyan", 7: "magenta"}
for classe, couleur in couleurs.items():
    subset = df[df["class_type"] == classe]
    plt.scatter(subset["legs"], subset["tail"],
                color=couleur, label=f"Classe {classe}", s=60)
plt.xlabel("Nombre de pattes")
plt.ylabel("Queue (0/1)")
plt.title("Répartition des animaux par classe")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Préparer
X = df.drop(columns=["animal_name", "class_type"])
y = df["class_type"]
print(f"\nFeatures utilisées : {X.columns.tolist()}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entraîner
modele = RandomForestClassifier(n_estimators=100, random_state=42)
modele.fit(X_train, y_train)

# 5. Évaluer
predictions = modele.predict(X_test)
print(f"\nPrécision : {accuracy_score(y_test, predictions) * 100:.1f}%")

importances = pd.DataFrame({
    "feature":    X.columns,
    "importance": modele.feature_importances_
}).sort_values("importance", ascending=False)
print("\n--- Importance des features ---")
print(importances)

plt.figure(figsize=(8, 5))
plt.barh(importances["feature"], importances["importance"], color="steelblue")
plt.title("Quelle caractéristique compte le plus ?", fontsize=14)
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 6. Prédire un nouvel animal
nouvel_animal = pd.DataFrame([{col: 0 for col in X.columns}])
nouvel_animal["hair"]     = 1
nouvel_animal["toothed"]  = 1
nouvel_animal["backbone"] = 1
nouvel_animal["breathes"] = 1
nouvel_animal["legs"]     = 4
nouvel_animal["tail"]     = 1

prediction = modele.predict(nouvel_animal)
proba = modele.predict_proba(nouvel_animal)

noms_classes = {
    1: "Mammifère", 2: "Oiseau", 3: "Reptile",
    4: "Poisson", 5: "Amphibien", 6: "Insecte", 7: "Invertébré"
}

print(f"\nAnimal prédit : {noms_classes[prediction[0]]}")
print("Probabilités :")
for classe, prob in zip(modele.classes_, proba[0]):
    print(f"  {noms_classes[classe]} : {prob * 100:.1f}%")
