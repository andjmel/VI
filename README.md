# 📚 Veštačka Inteligencija – Cheatsheets


## 📁 Fajlovi

| Fajl | Oblast |
|------|--------|
| [`00_priprema_podataka.md`](00_priprema_podataka.md) | Učitavanje, NA, outlieri, skaliranje, podela train/test, metrike |
| [`01_linearna_regresija.md`](01_linearna_regresija.md) | OLS, summary, dijagnostički grafikoni, VIF, RMSE, R² |
| [`02_decision_tree.md`](02_decision_tree.md) | DecisionTree, plot_tree, ccp_alpha pruning, GridSearchCV |
| [`03_knn.md`](03_knn.md) | KNN, RobustScaler, optimalno k, GridSearchCV |
| [`04_neuronske_mreze.md`](04_neuronske_mreze.md) | MLPClassifier, arhitektura, loss kriva, solver/activation |
| [`05_kmeans.md`](05_kmeans.md) | KMeans++, Elbow metoda, MinMaxScaler, tumačenje klastera |
| [`06_hijerarhijska_klasterizacija.md`](06_hijerarhijska_klasterizacija.md) | AgglomerativeClustering, dendrogram, linkage metode |

---

## 🔄 Opšti tok rešavanja zadatka

### Regresija (LR)
```
Učitaj → EDA (head/info/describe) → Korelaciona matrica →
Izbor prediktora → Train/Test split (stratify po kvantilima) →
smf.ols().fit() → summary() → Predikcija → R², RMSE →
Dijagnostički grafikoni → VIF (multikolinearnost)
```

### Klasifikacija (DT / KNN / MLP)
```
Učitaj → Napravi izlaznu promenljivu (q3) → Mapiraj kategoričke →
EDA (KDE / countplot za selekciju prediktora) →
Outlieri → Skaliranje (DT: nije potrebno, KNN/MLP: obavezno) →
Train/Test split (stratify=y) → Model → Predikcija →
Confusion matrix → Accuracy/Precision/Recall/F1
```

### Klasterizacija (KMeans / Hijerarhijska)
```
Učitaj → Izbaci irelevantne kolone → Konvertuj u numeričke →
One-Hot / mapiranje → Obrada NA → Nule u rang-listama →
Korelacije (izbaci visoko korelisane) → Outlieri + Winsorize →
MinMaxScaler → Elbow / Dendrogram → Klasterizacija →
Boxplot po klasterima → Tumačenje (veličina, centri, disperzija)
```

---

## ⚡ Najčešće greške

1. **Skaliranje**: `fit_transform` samo na train, `transform` na test!
2. **KNN bez skaliranja** → loši rezultati
3. **Neuronske mreže bez StandardScaler** → ne konvergira
4. **KMeans bez MinMaxScaler** → dominiraju promenljive sa većim rasponom
5. **Zaboraviti izbaciti 'cluster' kolonu** pre ponovne klasterizacije
6. **Nule u rang-listama** = "nije rangiran", NE "prvi na listi"!
7. **`fit_transform` na test skupu** → curite informacije iz test skupa (data leakage)

---

## 📋 Brza referenca – Scaleri

| Scaler | Kada | Zašto |
|--------|------|-------|
| `StandardScaler` | Normalna raspodela, MLP | Mean=0, Std=1 |
| `RobustScaler` | Nenormalna raspodela, KNN | Koristi medijanu i IQR, otporan na outliere |
| `MinMaxScaler` | Klasterizacija | Sve u [0,1], euklidsko rastojanje je fer |

## 📋 Brza referenca – Izbor metoda

| Metoda | Tip problema | Skaliranje |
|--------|-------------|-----------|
| Linearna regresija | Regresija | Nije obavezno |
| Decision Tree | Klasifikacija | Nije potrebno |
| KNN | Klasifikacija | Obavezno (Robust/Standard) |
| MLP | Klasifikacija | Obavezno (Standard) |
| KMeans | Klasterizacija | Obavezno (MinMax) |
| Hijerarhijska | Klasterizacija | Obavezno (MinMax) |
