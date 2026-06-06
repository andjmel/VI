# 🌳 Stablo Odlučivanja (Decision Tree) – Cheatsheet (FON VI)

## Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
```

---

## 1. Podela podataka

```python
X = df.drop(columns='HighSales')
y = df['HighSales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Proveri raspodelu klasa
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
```

---

## 2. Treniranje – bez podešavanja (tree1)

```python
tree1 = DecisionTreeClassifier(random_state=42)
tree1.fit(X_train, y_train)
```

---

## 3. Vizualizacija stabla

```python
plt.figure(figsize=[30, 10])
plot_tree(
    tree1,
    feature_names=X_train.columns,
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Stablo odlučivanja - tree1")
plt.show()
```

---

## 4. Predikcija i matrica konfuzije

```python
y_pred1 = tree1.predict(X_test)

# Matrica konfuzije
cm1 = confusion_matrix(y_test, y_pred1)
cm1_df = pd.DataFrame(
    cm1,
    index=['Stvarno:No', 'Stvarno:Yes'],
    columns=['Predviđeno:No', 'Predviđeno:Yes']
)
print(cm1_df)

# Prikaz
ConfusionMatrixDisplay(cm1, display_labels=['No', 'Yes']).plot(cmap=plt.cm.Blues)
plt.show()
```

---

## 5. Evaluacione metrike

```python
# Sklearn varijanta
def compute_eval_metrics(y_true, y_pred, pos_label=0):
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=pos_label),
        'recall':    recall_score(y_true, y_pred, pos_label=pos_label),
        'f1':        f1_score(y_true, y_pred, pos_label=pos_label),
    }

metrics1 = compute_eval_metrics(y_test, y_pred1)
print(metrics1)

# Manuelna varijanta iz cm
def compute_basic_eval_measures(cm, model_name=""):
    from pandas import Series
    TP = cm[1,1]; TN = cm[0,0]; FP = cm[0,1]; FN = cm[1,0]
    accuracy  = (TP + TN) / cm.sum()
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    F1        = 2 * precision * recall / (precision + recall)
    return Series([accuracy, precision, recall, F1],
                  index=['accuracy', 'precision', 'recall', 'F1'],
                  name=model_name)

eval1 = compute_basic_eval_measures(cm1, 'tree1')
```

---

## 6. Podrezivanje (Pruning) – GridSearch za ccp_alpha (tree2)

```python
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
param_grid = {'ccp_alpha': np.arange(0.0, 0.05, 0.0025)}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy'
)
grid.fit(X_train, y_train)

best_ccp_alpha = grid.best_params_['ccp_alpha']
print("Najbolja vrednost ccp_alpha:", best_ccp_alpha)
```

---

## 7. Treniranje sa podrezivanjem (tree2)

```python
tree2 = DecisionTreeClassifier(ccp_alpha=best_ccp_alpha, random_state=42)
tree2.fit(X_train, y_train)

y_pred2 = tree2.predict(X_test)
metrics2 = compute_eval_metrics(y_test, y_pred2)
```

---

## 8. Poređenje modela

```python
eval_df = pd.DataFrame(
    [metrics1, metrics2],
    index=['Tree1 (bez pruning)', 'Tree2 (sa pruning)']
)
print(eval_df)
# Tree2 obično ima manje čvorova i bolju generalizaciju
```

---

## 9. Selekcija prediktora (vizualizacija)

```python
# Numerički prediktor vs klasa → KDE plot
sns.kdeplot(data=df, x='Age', hue='Stayed', fill=True, alpha=0.55)
plt.show()
# Ako se krive preklapaju → nije koristan prediktor

# Kategorički prediktor vs klasa → Count plot
sns.countplot(data=df, x='IsActiveMember', hue='Stayed', palette='pastel')
plt.show()
# Ako su proporcije iste za obe klase → nije koristan prediktor
```

---

## 10. Napomene o pripremi za DT

```python
# DT ne zahteva skaliranje podataka
# Kategoričke promenljive MORAJU biti numeričke (map ili OneHotEncoder)
# Nominalne → One-Hot Encoding
# Ordinalne → map({'Bad': 0, 'Medium': 1, 'Good': 2})

# One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded = ohe.fit_transform(df[['Geography', 'Gender']])
encoded_cols = ohe.get_feature_names_out(['Geography', 'Gender'])
df_ohe = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
df = df.drop(columns=['Geography', 'Gender']).join(df_ohe)
```

---

## Napomene

| Pojam | Šta znači |
|-------|-----------|
| `ccp_alpha=0` | Bez podrezivanja (stablo može biti preveliko) |
| Veće `ccp_alpha` | Agresivnije podrezivanje (manje stablo) |
| `stratify=y` | Čuva raspodelu klasa u train/test |
| `pos_label` | Klasa koja se smatra "pozitivnom" u metrikama |
| Precision | TP / (TP + FP) – koliko je predikcija ispravno |
| Recall | TP / (TP + FN) – koliko je stvarnih pozitivnih uhvaćeno |
| F1 | Harmonijska sredina Precision i Recall |
