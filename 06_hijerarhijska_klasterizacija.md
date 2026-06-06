# 🌿 Hijerarhijska Klasterizacija – Cheatsheet (FON VI)

## Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats.mstats import winsorize
```

---

## ⚠️ Priprema podataka je ista kao za KMeans (vidi kmeans cheatsheet)!

---

## 1. Priprema podataka (ista kao KMeans)

```python
# Učitaj, filtriraj, mapiraj kategoričke, one-hot, obradi NA,
# obradi outliere, korelacije, MinMaxScaler
# → pogledaj 05_kmeans.md za detalje

# Finalni skalirani DataFrame:
temp = MinMaxScaler().fit_transform(cl_data)
cl_data_scaled = pd.DataFrame(temp, columns=cl_data.columns)
```

---

## 2. Dendrogram – određivanje optimalnog K i linkage metode

```python
from scipy.cluster.hierarchy import linkage, dendrogram

# Probaj sve 4 linkage metode
for method in ['ward', 'single', 'complete', 'average']:
    dendrogram(linkage(cl_data_scaled, method=method, metric='euclidean'))
    plt.title(f"{method.capitalize()} linkage dendrogram")
    plt.show()
```

### Tumačenje dendrograma

```
- Ward linkage:   minimizuje ukupnu varijansu unutar klastera (najčešće se bira)
- Single linkage: minimalna razdaljina između tačaka (sklon "chaining efektu")
- Complete linkage: maksimalna razdaljina između tačaka
- Average linkage: prosečna razdaljina između tačaka

Kako odrediti K:
→ Nacrtaj horizontalnu liniju kroz najduži vertikalni segment (koji se ne seče)
→ Broj presečenih vertikalnih linija = optimalan K
→ Ako postoje 2 moguća K (npr. 2 ili 4) – probaj oba!
```

---

## 3. Hijerarhijska klasterizacija – AgglomerativeClustering

```python
# K = 2, Ward linkage, euklidska udaljenost
hclust = cluster.AgglomerativeClustering(
    n_clusters=2,
    linkage='ward',     # 'ward', 'single', 'complete', 'average'
    metric='euclidean'  # 'euclidean', 'manhattan', 'cosine'
).fit(cl_data_scaled)

# Dodaj labele u dataset
twocl = cl_data_scaled.copy()
twocl['cluster'] = hclust.labels_

# Posebni DataFrameovi po klasteru
cl1 = twocl[twocl['cluster'] == 0].drop(columns='cluster').copy()
cl2 = twocl[twocl['cluster'] == 1].drop(columns='cluster').copy()
```

---

## 4. Vizualizacija klastera – Boxplot

```python
# 2 klastera
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].boxplot(cl1, tick_labels=cl1.columns)
ax[0].set_title(f"Klaster 1 (N={cl1.shape[0]})")

ax[1].boxplot(cl2, tick_labels=cl2.columns)
ax[1].set_title(f"Klaster 2 (N={cl2.shape[0]})")

plt.tight_layout()
plt.show()

# 3 klastera
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for i, (cl, ax_i) in enumerate(zip([cl1, cl2, cl3], ax), 1):
    ax_i.boxplot(cl, tick_labels=cl.columns)
    ax_i.set_title(f"Klaster {i} (N={cl.shape[0]})")
plt.tight_layout()
plt.show()

# 4 klastera
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
for idx, (cl, (r, c)) in enumerate(
        zip([cl1, cl2, cl3, cl4], [(0,0),(0,1),(1,0),(1,1)]), 1):
    ax[r,c].boxplot(cl, tick_labels=cl.columns)
    ax[r,c].set_xlabel(f"Klaster {idx} ({cl.shape[0]})")
plt.tight_layout()
plt.show()
```

---

## 5. Ponovna klasterizacija (sa K=4)

```python
# Izbaci staru 'cluster' kolonu pre ponovnog treniranja!
cl_data_scaled.drop(columns='cluster', inplace=True)

hclust = cluster.AgglomerativeClustering(
    n_clusters=4,
    linkage='ward',
    metric='euclidean'
).fit(cl_data_scaled)

cl_data_scaled['cluster'] = hclust.labels_
```

---

## 6. Veličina klastera

```python
for i in range(n_clusters):
    n = cl_data_scaled[cl_data_scaled['cluster'] == i]['cluster'].count()
    print(f"Klaster {i+1}, N = {n}")
```

---

## KMeans vs Hijerarhijska – razlike

| Osobina | KMeans | Hijerarhijska |
|---------|--------|---------------|
| K se zadaje unapred | Da (+ Elbow metoda) | Da (+ Dendrogram) |
| Odredjivanje K | Elbow metoda | Dendrogram |
| Deterministički | Delimično (n_init) | Da (Ward je deterministički) |
| Efikasnost | Brži za velike skupove | Sporiji za velike skupove |
| Prikazivanje strukture | Ne | Da (dendrogram) |
| Linkage | Nema | ward/single/complete/average |

---

## Linkage metode – tumačenje u praksi

```
Ward:     Najčešće se bira – daje kompaktne, balansovane klastere.
          Bira se kada su klasteri jasno odvojeni na dendrogramu.

Complete: Dobar za okrugle, kompaktne klastere.

Average:  Kompromis između Single i Complete.

Single:   Sklon "chaining" efektu – ne preporučuje se u praksi.
          Jedan klaster može biti jako velik a ostali mali.
```

---

## Šablon tumačenja klastera

```
1. Veličina klastera:
   Klaster X ima N instanci – svi/nisu značajne veličine.

2. Centri klastera:
   Klaster 1: [opis – visoki/niski koji atribut, npr. "isključivo žene, niže klase"]
   Klaster 2: [opis]

3. Disperzija (homogenost):
   Klaster X je najhomogeniji po atributu Y.
   Klaster Z ima veliku disperziju po atributu W.
   Razlike/sličnosti između klastera po ostalim atributima.
```
