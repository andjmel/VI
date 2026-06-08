# 📋 Priprema Podataka – Cheatsheet (FON VI)

## 1. Učitavanje i prvi pregled

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import shapiro

df = pd.read_csv('data/podaci.csv', na_values=['-', ' ', ''])

df.head()          # prvih 5 redova
df.info()          # tipovi kolona, broj non-null vrednosti
df.describe()      # deskriptivna statistika
df.isna().sum()    # broj NA po koloni
df.isna().mean() * 100  # procenat NA po koloni (za odluku o izbacivanju, granicu uzeti >20%)
```
df_subset=df[~df['rm'].astype(str).str.contains(r"[-/ ]", regex=True)]
invalid_rm = df[
    df['rm'].astype(str).str.contains("-") |
    df['rm'].astype(str).str.contains("/") |
    df['rm'].astype(str).str.contains(" ")
    ]
---

## 2. Kreiranje izlazne promenljive (klasifikacija)

```python
# Na osnovu kvartila
q3 = df['Sales'].quantile(0.75)
df['HighSales'] = np.where(df['Sales'] > q3, 'Yes', 'No')

# Mapiranje string -> broj
df['HighSales'] = df['HighSales'].map({'Yes': 1, 'No': 0})

# Ukloni originalu kolonu
df.drop(columns=['Sales'], inplace=True)
```

---

## 3. Binarno mapiranje kategorijskih promenljivih

```python
df['Urban'] = df['Urban'].map({'Yes': 1, 'No': 0})
df['US']    = df['US'].map({'Yes': 1, 'No': 0})
df['Sex']   = df['Sex'].map({'male': 0, 'female': 1})

# Ordinalna (postoji redosled)
df['ShelveLoc'] = df['ShelveLoc'].map({'Bad': 0, 'Medium': 1, 'Good': 2})
```

---

## 4. One-Hot Encoding (nominalna, bez redosleda)

```python
# Scipy/Pandas varijanta
df[['Cherbourg', 'Queenstown', 'Southampton']] = pd.get_dummies(df['Embarked'], dtype=int)
df.drop(columns='Embarked', inplace=True)

# Sklearn varijanta (drop='first' uklanja prvu kolonu da izbegne multikolinearnost)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded = ohe.fit_transform(df[['Geography', 'Gender']])
encoded_cols = ohe.get_feature_names_out(['Geography', 'Gender'])
df_ohe = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
df = df.drop(columns=['Geography', 'Gender']).join(df_ohe)
```

---

## 5. Konverzija u numerički tip

```python
# Pogrešno unete vrednosti postaju NaN
df['rm'] = pd.to_numeric(df['rm'], errors='coerce')

# Proveri nevalidne karaktere
invalid = df[df['rm'].astype(str).str.contains(r'[-/ ]', regex=True)]

# Ukloni redove sa nevalidnim karakterima
df = df[~df['rm'].astype(str).str.contains(r'[-/ ]', regex=True)]
```

---

## 6. Obrada nedostajućih vrednosti (NA)

```python
# Izbaci redove sa NA
df.dropna(inplace=True)

# Zameni medijanom (kad raspodela NIJE normalna)
median = df['Age'].median()
df['Age'] = df['Age'].fillna(median)

# Zameni modom (kategorijska promenljiva)
df['Geography'] = df['Geography'].fillna(df['Geography'].mode()[0])

# Zameni medijanom uz loc
df.loc[df['col'].isnull(), 'col'] = df['col'].median()

# Kolone sa >20% NA → izbaci kolonu
df.drop(columns='calcium', inplace=True)
```

---

## 7. Test normalnosti (Shapiro-Wilk)

```python
from scipy.stats import shapiro

stat, p = shapiro(df['Age'].dropna())
print(f"p = {p:.5f} → {'nije normalna' if p < 0.05 else 'normalna'} distribucija")

# Za sve numeričke kolone
shapiro_results = df.select_dtypes('number').apply(lambda x: shapiro(x.dropna())[1])
print(shapiro_results)
# p < 0.05 → nije normalna → koristiti RobustScaler / medijanu
# p > 0.05 → normalna    → koristiti StandardScaler / mean
```

---

## 8. Outlieri – detekcija i obrada

```python
# Broj outliera po koloni (IQR metoda)
def count_outliers(col):
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    return ((col < q1 - 1.5*iqr) | (col > q3 + 1.5*iqr)).sum()

outliers = df.select_dtypes('number').apply(count_outliers)
print(outliers)

# Vizualizacija
plt.boxplot(df[numeric_cols], tick_labels=numeric_cols)
plt.show()

# Winsorize (zamena outliera granicom percentila)
from scipy.stats.mstats import winsorize
# limits=[donja, gornja] – procenat koji se zamenjuje
wins = winsorize(df['Fare'].copy(), limits=[0, 0.08])   # gorna 8%
plt.boxplot(wins); plt.show()
df['Fare'] = wins

# Pravilo: ako >10% vrednosti su outlieri → IZBACI kolonu
#           ako <10%                      → Winsorize
```

---

## 9. Standardizacija / Skaliranje

```python
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# StandardScaler – za normalno distribuirane promenljive
scaler = StandardScaler()
df[normally_dist_cols] = scaler.fit_transform(df[normally_dist_cols])

# RobustScaler – za nenormalne promenljive (koristi medijanu i IQR)
robust = RobustScaler()
df[not_normal_cols] = robust.fit_transform(df[not_normal_cols])

# MinMaxScaler – za klasterizaciju (normalizacija na 0-1)
from sklearn.preprocessing import MinMaxScaler
df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)

# VAŽNO: fit samo na train, transform i na test!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)        # NE fit_transform!
```

---

## 10. Korelaciona matrica

```python
import seaborn as sb

corr = df.corr(numeric_only=True).round(2)
plt.figure(figsize=(10, 8))
sb.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Korelaciona matrica")
plt.tight_layout()
plt.show()

# Pravilo za regresiju: |r| > 0.5 → dobar prediktor (okvirno)
# Za klasterizaciju: |r| > 0.7 između prediktora → multikolinearnost → izbaci jednu
```

---

## 11. Vizualizacija za selekciju prediktora (klasifikacija)

```python
import seaborn as sns

# Numerički prediktor vs klasa → KDE plot
sns.kdeplot(data=df, x='Age', hue='Stayed', fill=True, alpha=0.55)
plt.show()
# Ako se krive PREKLAPAJU → prediktor nije koristan → izbaci

# Kategorijski prediktor vs klasa → Count plot
sns.countplot(data=df, x='IsActiveMember', hue='Stayed', palette='pastel')
plt.show()
# Ako su proporcije iste za obe klase → prediktor nije koristan → izbaci
```

---

## 12. Podela na train/test skup

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns='target')
y = df['target']

# Za klasifikaciju – stratify čuva raspodelu klasa
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Za regresiju – stratify po kvantilima
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, shuffle=True,
    stratify=pd.qcut(df['medv'], q=10, duplicates='drop')
)

# Proveri raspodelu klasa
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
```

---

## 13. Evaluacione metrike (klasifikacija)

```python
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score)

# Matrica konfuzije
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=['Stvarno:0','Stvarno:1'],
                       columns=['Pred:0','Pred:1']))

# Prikaz
ConfusionMatrixDisplay(cm, display_labels=['No','Yes']).plot(cmap=plt.cm.Blues)
plt.show()

# Metrike
def compute_eval_metrics(y_true, y_pred, pos_label=1):
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=pos_label),
        'recall':    recall_score(y_true, y_pred, pos_label=pos_label),
        'f1':        f1_score(y_true, y_pred, pos_label=pos_label),
    }

# Manuelno iz matrice konfuzije
TP = cm[1,1]; TN = cm[0,0]; FP = cm[0,1]; FN = cm[1,0]
accuracy  = (TP + TN) / cm.sum()
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

---

## 14. Evaluacione metrike (regresija)

```python
y_true = test_df['medv']
y_pred = model.predict(test_df)

rss  = ((y_pred - y_true) ** 2).sum()
tss  = ((y_true - y_true.mean()) ** 2).sum()
r2   = 1 - rss / tss
rmse = np.sqrt(rss / len(y_true))

print(f"RSS  = {rss:.2f}")   # → 0
print(f"R²   = {r2:.3f}")    # → 1
print(f"RMSE = {rmse:.2f}")  # → 0
```
