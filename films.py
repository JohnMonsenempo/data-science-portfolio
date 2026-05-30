import pandas as pd

data = {
    "titre":  ["Matrix", "Deadpool", "I am Legend", "Creed", "Hitch", "hitman's bodyguard"],
    "annee":  [1999, 2016, 2014, 2015, 2005, 2017],
    "note":   [9.2, 8.0, 8.6, 7.1, 9.0, 7.5],
    "genre":  ["Sci-Fi", "Super-Héros", "Sci-Fi", "Sport", "Comédie", "Action"],
}

df = pd.DataFrame(data)

# 1. Films avec note > 8
bons_films = df[df["note"] > 8]
print("--- Films avec note > 8 ---")
print(bons_films)

# 2. Meilleur film
meilleur = df[df["note"] == df["note"].max()]
print("\n--- Meilleur film ---")
print(meilleur)

# 3. Colonne recommande
df["recommande"] = df["note"].apply(lambda x: "Oui" if x >= 7 else "Non")
print("\n--- DataFrame complet ---")
print(df)

# 4. Sauvegarder
df.to_csv("films.csv", index=False)
print("\n✅ Sauvegardé dans films.csv !")

# Afficher la note moyenne de tous vos films
print(f"Note moyenne : {df['note'].mean():.2f}")