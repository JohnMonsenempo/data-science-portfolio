import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "mois": ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
             "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"],
    "ventes":    [15000, 18200, 12800, 21500, 19300, 23100,
                  25000, 28000, 22000, 26000, 30000, 35000],
    "depenses":  [9000, 10500, 8200, 12000, 11800, 13200,
                  14000, 15500, 12500, 14800, 17000, 19500],
    "employes":  [5, 5, 4, 6, 6, 7, 7, 8, 7, 8, 9, 10],
    "clients":   [120, 145, 98, 167, 155, 189,
                  210, 235, 185, 220, 260, 310]
}

df = pd.DataFrame(data)

# 1. Analyse
df["profit"] = df["ventes"] - df["depenses"]
df["croissance"] = df["ventes"].pct_change() * 100

record_ventes = df["ventes"].max()
record_mois = df[df["ventes"] == record_ventes]["mois"].values[0]
print(f"Mois record : {record_mois} avec {record_ventes}€")
print(f"Profit total : {df['profit'].sum()}€")
print(f"Moyenne ventes : {df['ventes'].mean():.0f}€")

# 2. Dashboard 4 graphiques
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Dashboard Ventes 2024", fontsize=16, fontweight="bold")

axes[0, 0].plot(df["mois"], df["ventes"], marker="o", color="steelblue")
axes[0, 0].set_title("Ventes mensuelles")
axes[0, 0].set_ylabel("Ventes (€)")

axes[0, 1].plot(df["mois"], df["depenses"], marker="o", color="coral")
axes[0, 1].set_title("Dépenses mensuelles")
axes[0, 1].set_ylabel("Dépenses (€)")

axes[1, 0].plot(df["mois"], df["profit"], marker="o", color="green")
axes[1, 0].set_title("Profit mensuel")
axes[1, 0].set_ylabel("Profit (€)")

axes[1, 1].plot(df["mois"], df["clients"], marker="o", color="orange")
axes[1, 1].set_title("Nombre de clients")
axes[1, 1].set_ylabel("Clients")

for ax in axes.flat:
    ax.set_xlabel("Mois")
    ax.grid(True)

plt.tight_layout()

# 3. Prédiction 3 mois
X = np.arange(len(df)).reshape(-1, 1)
y = df["ventes"].values
modele = LinearRegression()
modele.fit(X, y)

prochains_mois = np.array([[12], [13], [14]])
predictions = modele.predict(prochains_mois)
noms_mois = ["Jan 2025", "Fév 2025", "Mar 2025"]
print("\n--- Prédictions ---")
for mois, pred in zip(noms_mois, predictions):
    print(f"Ventes prédites {mois} : {pred:.0f}€")

# 4. Sauvegarder
fig.savefig("dashboard_ventes.png")   # avant show()
plt.show()
df.to_csv("rapport_ventes.csv", index=False)
print("\n Dashboard sauvegardé !")
