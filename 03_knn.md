# 🔍 K-Nearest Neighbors (kNN) – Cheatsheet (FON VI)

## Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score)
```

---

## ⚠️ KNN zahteva standardizaciju! (za razliku od DT)

---

## 1. Test normalnosti i odabir skalera

```python
from scipy.stats import shapiro

shapiro_results = df[numeric_cols].apply(lambda x: shapiro(x.dropna())[1])
print(shapiro_results)

# p < 0.05 → nije normalna → RobustScaler
# p > 0.05 → normalna    → StandardScaler
```

---

## 2. Standardizacija

```python
from sklearn.preprocessing import RobustScaler, StandardScaler

not_normal      = ['Income', 'Advertising', 'Population']
normally_dist   = ['CompPrice', 'Price']

df_st = pd.DataFrame()

# Nenormalne promenljive
robust = RobustScaler()
df_st[not_normal] = robust.fit_transform(df[not_normal])

# Normalne promenljive
standard = StandardScaler()
df_st[normally_dist] = standard.fit_transform(df[normally_dist])

# Binarne / kategoričke → samo mapiraj, ne skalirati
df_st['Urban'] = df['Urban'].map({'No': 0, 'Yes': 1})
df_st['HighSales'] = df['HighSales'].map({'No': 0, 'Yes': 1})

# VAŽNO: fit na train, transform na test
scaler = RobustScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)    # NE fit_transform!
```

---

## 3. Podela podataka

```python
X = df_st.drop(columns='HighSales')
y = df_st['HighSales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## 4. KNN sa fiksnim k (knn1)

```python
knn1 = KNeighborsClassifier(n_neighbors=5)
knn1.fit(X_train, y_train)
y_pred1 = knn1.predict(X_test)
```

---

## 5. Optimalno k – GridSearchCV (knn2)

```python
knn = KNeighborsClassifier()

# Neparne vrednosti (da postoji dominantna klasa)
param_grid = {'n_neighbors': list(range(3, 26, 2))}

cv = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=10,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
cv.fit(X_train, y_train)

best_k = cv.best_params_['n_neighbors']
print("Optimalna vrednost za k:", best_k)
```

---

## 6. Treniranje sa optimalnim k

```python
knn2 = KNeighborsClassifier(n_neighbors=best_k)
knn2.fit(X_train, y_train)
y_pred2 = knn2.predict(X_test)
```

---

## 7. Matrica konfuzije i metrike

```python
cm = confusion_matrix(y_test, y_pred2)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrica konfuzije – knn2")
plt.show()

def compute_eval_metrics(y_true, y_pred, pos_label=0):
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': float(precision_score(y_true, y_pred, pos_label=pos_label)),
        'recall':    float(recall_score(y_true, y_pred, pos_label=pos_label)),
        'f1':        float(f1_score(y_true, y_pred, pos_label=pos_label)),
    }

metrics2 = compute_eval_metrics(y_test, y_pred2)
print(metrics2)
```

---

## 8. Poređenje modela

```python
metrics1 = compute_eval_metrics(y_test, y_pred1)
metrics2 = compute_eval_metrics(y_test, y_pred2)

eval_df = pd.DataFrame(
    [metrics1, metrics2],
    index=['knn1 (k=5)', f'knn2 (k={best_k})']
)
print(eval_df)
```

---

## 9. Selekcija prediktora za KNN

```python
# KDE plot za numeričke
for col in numeric_cols:
    sns.kdeplot(data=df, x=col, hue='HighSales', fill=True, alpha=0.5)
    plt.title(col); plt.show()
    # Ako se krive preklapaju → nije koristan → izbaci

# Count plot za kategoričke
for col in cat_cols:
    sns.countplot(data=df, x=col, hue='HighSales', palette='pastel')
    plt.title(col); plt.show()
```

---

## 10. Outlieri za KNN

```python
# Detekcija (IQR metoda)
def count_outliers(col):
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    return ((col < q1 - 1.5*iqr) | (col > q3 + 1.5*iqr)).sum()

outliers = df[numeric_cols].apply(count_outliers)
print(outliers)

# Winsorize (ako ima outliere)
from scipy.stats.mstats import winsorize
df['col'] = winsorize(df['col'].copy(), limits=[0, 0.05])
```

---

## Napomene

| Pojam | Šta znači |
|-------|-----------|
| KNN mora biti skaliran | Euklid rastojanje je osetljivo na raspon |
| Neparne k vrednosti | Uvek postoji dominantna klasa |
| RobustScaler | Koristi medijanu i IQR, otporan na outliere |
| StandardScaler | Koristi mean i std, za normalne raspodele |
| `fit_transform` na train | Naučiti parametre skaliranja samo na train! |
| `transform` na test | Samo primeniti naučene parametre |
| Ako krive KDE se preklapaju | Prediktor ne doprinosi – izbaci ga |
