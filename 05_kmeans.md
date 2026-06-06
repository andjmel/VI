# 📊 K-Means Klasterizacija – Cheatsheet (FON VI)

## Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn.cluster as cluster
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.mstats import winsorize
from scipy.stats import shapiro
```

---

## ⚠️ Klasterizacija ne koristi izlaznu promenljivu (unsupervised)!

---

## 1. Priprema podataka za klasterizaciju

```python
# a) Učitavanje i filtriranje
df = pd.read_csv('data.csv')
cl_data = df[df['speechiness'] <= 60].copy()

# b) Izbaci ne-numeričke i irelevantne kolone
cl_data.drop(columns=['track_name', 'instrumentalness'], inplace=True)

# c) Konvertuj object kolone u numeričke
cl_data['in_spotify_charts'] = pd.to_numeric(cl_data['in_spotify_charts'], errors='coerce')

# d) Mapiraj binarne kategoričke kolone
cl_data['mode'] = cl_data['mode'].map({'Major': 0, 'Minor': 1})
cl_data['Sex']  = cl_data['Sex'].map({'male': 0, 'female': 1})

# e) One-Hot Encoding za nominalne
cl_data[['Cherbourg','Queenstown','Southampton']] = pd.get_dummies(
    cl_data['Embarked'], dtype=int
)
cl_data.drop(columns='Embarked', inplace=True)
```

---

## 2. Obrada NA vrednosti

```python
cl_data.isnull().sum()

# Ako malo NA vrednosti (< ~1%) → zameni medijanom
cl_data.loc[cl_data['in_spotify_charts'].isnull(), 'in_spotify_charts'] = \
    cl_data['in_spotify_charts'].median()

# Ili izbaci redove
cl_data.dropna(inplace=True)
```

---

## 3. Problem sa nulama u rang-listama

```python
# 0 = "nije na listi", ali 0 < 1 → KMeans bi smatrao da su 0 BOLJE pesme!
# Rešenje: zameni 0 sa max vrednošću (= najlošije rangirana)
cl_data.loc[cl_data['in_spotify_charts'] == 0, 'in_spotify_charts'] = \
    cl_data['in_spotify_charts'].max()
cl_data.loc[cl_data['in_apple_charts'] == 0, 'in_apple_charts'] = \
    cl_data['in_apple_charts'].max()
```

---

## 4. Korelacije – izbaci visoko korelisane

```python
plt.figure(figsize=(8, 6))
sb.heatmap(cl_data.corr(), cmap='Blues', annot=True)
plt.show()

# |r| > 0.7 između prediktora → izbaci jednu od visoko korelisanih
# Primer: streams, in_apple_playlists i in_spotify_playlists su korelisane
cl_data.drop(columns=['in_apple_playlists', 'streams'], inplace=True)
```

---

## 5. Outlieri

```python
# Boxplot
plt.boxplot(cl_data, tick_labels=cl_data.columns)
plt.show()

# Winsorize kolona sa outlierima
from scipy.stats.mstats import winsorize

# Pravilo:
# Ako winsorize > 10% → IZBACI kolonu
# Ako winsorize ≤ 10% → ZAMENI outliere winsorize vrednostima

wins = winsorize(cl_data['bpm'].copy(), limits=[0.00, 0.02])  # gornja 2%
cl_data['bpm'] = wins

# Proba winsorize i proveri boxplot
wins_test = winsorize(cl_data['col'].copy(), limits=[0.00, 0.10])
plt.boxplot(wins_test); plt.show()
# Ako IMA OUTLIERA posle winsorize(10%) → izbaci kolonu
cl_data.drop(columns=['problematic_col'], inplace=True)
```

---

## 6. Normalizacija – OBAVEZNA za KMeans

```python
# MinMaxScaler → sve promenljive u raspon [0, 1]
temp = MinMaxScaler().fit_transform(cl_data)
cl_data_scaled = pd.DataFrame(temp, columns=cl_data.columns)

cl_data_scaled.describe()
# sve min = 0, sve max = 1 → OK
```

---

## 7. Elbow metoda – određivanje optimalnog K

```python
wcss = []

for i in range(1, 11):
    km = cluster.KMeans(
        n_clusters=i,
        init='k-means++',
        n_init=1000,
        max_iter=100,
        random_state=0
    ).fit(cl_data_scaled)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss, 'bo-')
plt.xlabel("Broj klastera (K)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow metoda")
plt.show()

# "Lakat" (nagli pad prestaje) → optimalno K
# Često 2 lakta → probati oba K
```

---

## 8. Klasterizacija

```python
# K = 2
kmeans = cluster.KMeans(
    n_clusters=2,
    init='k-means++',
    n_init=1000,
    max_iter=100,
    random_state=0
)
kmeans.fit(cl_data_scaled)

twocl = cl_data_scaled.copy()
twocl['cluster'] = kmeans.labels_

# Posebni DataFrame po klasteru
cl1 = twocl[twocl['cluster'] == 0].drop(columns='cluster').copy()
cl2 = twocl[twocl['cluster'] == 1].drop(columns='cluster').copy()
```

---

## 9. Vizualizacija i tumačenje klastera

```python
# Boxplot za 2 klastera
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].boxplot(cl1, tick_labels=cl1.columns)
ax[0].set_title(f"Klaster 1 (N={cl1.shape[0]})")

ax[1].boxplot(cl2, tick_labels=cl2.columns)
ax[1].set_title(f"Klaster 2 (N={cl2.shape[0]})")

plt.tight_layout()
plt.show()

# Boxplot za 4 klastera
cl1 = twocl[twocl['cluster']==0].drop(columns='cluster')
cl2 = twocl[twocl['cluster']==1].drop(columns='cluster')
cl3 = twocl[twocl['cluster']==2].drop(columns='cluster')
cl4 = twocl[twocl['cluster']==3].drop(columns='cluster')

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
short = cl1.columns  # ili krace oznake

ax[0,0].boxplot(cl1, tick_labels=short); ax[0,0].set_xlabel(f"Klaster 1 ({cl1.shape[0]})")
ax[0,1].boxplot(cl2, tick_labels=short); ax[0,1].set_xlabel(f"Klaster 2 ({cl2.shape[0]})")
ax[1,0].boxplot(cl3, tick_labels=short); ax[1,0].set_xlabel(f"Klaster 3 ({cl3.shape[0]})")
ax[1,1].boxplot(cl4, tick_labels=short); ax[1,1].set_xlabel(f"Klaster 4 ({cl4.shape[0]})")

plt.tight_layout()
plt.show()
```

---

## 10. Veličina klastera

```python
print("Klaster 1, N =", twocl[twocl['cluster']==0]['cluster'].count())
print("Klaster 2, N =", twocl[twocl['cluster']==1]['cluster'].count())
```

---

## 11. Ponovna klasterizacija (izbaci staru 'cluster' kolonu!)

```python
cl_data_scaled.drop(columns='cluster', inplace=True)

# Pa pokreni KMeans ponovo sa novim K
kmeans = cluster.KMeans(n_clusters=4, init='k-means++',
                         max_iter=100, n_init=1000, random_state=0)
kmeans.fit(cl_data_scaled)
cl_data_scaled['cluster'] = kmeans.labels_
```

---

## Šablon tumačenja klastera

```
1. Veličina klastera:
   Klaster X ima N instanci – svi klasteri su/nisu značajne veličine.

2. Centri klastera (šta ih razlikuje):
   Klaster 1: [opis karakteristika – visoki/niski koji atribut]
   Klaster 2: ...

3. Disperzija (homogenost):
   Klaster X je najhomogeniji po atributu Y (mala disperzija / kratke brke).
   Klaster Z ima veliku disperziju po atributu W.
```

---

## Napomene

| Pojam | Šta znači |
|-------|-----------|
| `init='k-means++'` | Pametna inicijalizacija centroida (bolje od random) |
| `n_init=1000` | Broj random restarta – veće = stabilnije |
| `inertia_` | WCSS – ukupna suma kvadratnih odstupanja od centroida |
| Elbow metoda | Tražimo K gde WCSS prestaje naglnao da opada |
| MinMaxScaler | Obavezan jer KMeans koristi euklidsko rastojanje |
| Visoka korelacija | Visoko korelisane varijable dominiraju klasterizacijom – izbaci |
