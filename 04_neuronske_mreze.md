# 🧠 Neuronske Mreže (MLP) – Cheatsheet (FON VI)

## Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score)
```

---

## ⚠️ Neuronske mreže zahtevaju StandardScaler!

---

## 1. Priprema i standardizacija

```python
X = df.drop(columns='isBenign')
y = df['isBenign']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# StandardScaler – fit samo na train!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)     # NE fit_transform!
```

---

## 2. Kreiranje i treniranje MLP modela

```python
# Osnovna arhitektura: 2 skrivena sloja (64 i 32 neurona)
mlp1 = MLPClassifier(
    hidden_layer_sizes=(64, 32),   # arhitektura skrivenih slojeva
    max_iter=1000,                 # maksimalan broj epoha
    random_state=42                # reproduktivnost
)
mlp1.fit(X_train, y_train)

# Sa SGD solverom i tanh aktivacijom
mlp2 = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=1000,
    random_state=42,
    activation='tanh',             # 'relu' (default), 'tanh', 'logistic'
    learning_rate_init=0.0001,     # početna stopa učenja
    solver='sgd'                   # 'adam' (default), 'sgd', 'lbfgs'
)

# Jedan skriveni sloj sa 128 neurona
mlp3 = MLPClassifier(
    hidden_layer_sizes=(128,),     # tuple sa jednim elementom!
    max_iter=1000,
    random_state=42,
    activation='tanh',
    learning_rate_init=0.0001,
    solver='sgd'
)
```

---

## 3. Praćenje konvergencije – loss kriva

```python
plt.figure()
plt.plot(mlp1.loss_curve_)
plt.xlabel("Iteracija")
plt.ylabel("Loss")
plt.title("Promena funkcije greške tokom treninga")
plt.grid()
plt.show()
# Loss treba da opada i da se "slegne" (konvergira)
# Ako ne opada → povećaj max_iter ili promeni solver/lr
```

---

## 4. Predikcija

```python
y_pred  = mlp1.predict(X_test)          # klase (0 ili 1)
y_proba = mlp1.predict_proba(X_test)    # verovatnoće za svaku klasu

# Primeri
for i in range(5):
    print("Predikcija:", y_pred[i], "| Verovatnoće:", y_proba[i])
```

---

## 5. Matrica konfuzije

```python
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['malignant', 'benign'])
disp.plot()
plt.show()
```

---

## 6. Evaluacione metrike

```python
def compute_eval_metrics(y_true, y_pred, pos_label=0):
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': float(precision_score(y_true, y_pred, pos_label=pos_label)),
        'recall':    float(recall_score(y_true, y_pred, pos_label=pos_label)),
        'f1':        float(f1_score(y_true, y_pred, pos_label=pos_label)),
    }

mlp1_eval = compute_eval_metrics(y_test, y_pred)
print(mlp1_eval)
```

---

## 7. Poređenje više modela

```python
y_pred1 = mlp1.predict(X_test)
y_pred2 = mlp2.predict(X_test)
y_pred3 = mlp3.predict(X_test)

eval_df = pd.DataFrame(
    [compute_eval_metrics(y_test, y_pred1),
     compute_eval_metrics(y_test, y_pred2),
     compute_eval_metrics(y_test, y_pred3)],
    index=['mlp1 (64,32) relu/adam',
           'mlp2 (64,32) tanh/sgd',
           'mlp3 (128,)  tanh/sgd']
)
print(eval_df)
```

---

## 8. Učitavanje sklearn dataseta (za vežbu)

```python
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target
print(cancer_data.DESCR)
```

---

## Parametri MLPClassifier – najvažniji

| Parametar | Opcije | Default |
|-----------|--------|---------|
| `hidden_layer_sizes` | `(64,)`, `(64,32)`, `(128,64,32)` | `(100,)` |
| `activation` | `'relu'`, `'tanh'`, `'logistic'` | `'relu'` |
| `solver` | `'adam'`, `'sgd'`, `'lbfgs'` | `'adam'` |
| `learning_rate_init` | `0.001`, `0.0001`, ... | `0.001` |
| `max_iter` | `500`, `1000`, ... | `200` |
| `random_state` | `42` | `None` |

---

## Napomene

| Pojam | Šta znači |
|-------|-----------|
| `StandardScaler` obavezan | MLP je osjetljiv na razmeru ulaza |
| `fit_transform` na train | Naučiti parametre skaliranja samo na train |
| `transform` na test | Primeniti iste parametre na test |
| `loss_curve_` | Lista vrednosti loss-a po epohi (samo za `adam` i `sgd`) |
| `predict_proba` | Verovatnoća za svaku klasu |
| `hidden_layer_sizes=(64,)` | Tuple s jednim elementom – tačka na kraju! |
| Ako loss ne konvergira | Povećaj `max_iter`, smanji `learning_rate_init` |
