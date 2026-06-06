# 📈 Linearna Regresija (LR) – Cheatsheet (FON VI)

## Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
```

---

## 1. Podela podataka

```python
# Regresija – stratify po kvantilima ciljne promenljive
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, shuffle=True,
    stratify=pd.qcut(df['medv'], q=10, duplicates='drop')
)
```

---

## 2. Treniranje modela

```python
# Jednostavna linearna regresija (jedan prediktor)
model = smf.ols(formula='medv ~ lstat', data=train_df).fit()

# Višestruka linearna regresija
model = smf.ols(formula='medv ~ lstat + rm + ptratio', data=train_df).fit()

# Svi numerički prediktori automatski
all_num = train_df.select_dtypes('number').columns.drop('medv')
model = smf.ols('medv ~ ' + ' + '.join(all_num), data=train_df).fit()
```

---

## 3. Analiza modela – summary

```python
print(model.summary())
# Ključne stvari u summary:
#   R-squared    → koliko % varijanse model objašnjava (veće = bolje)
#   Adj. R²      → korigovani R² (za višestruku regresiju)
#   coef         → koeficijenti (za svaki prediktor)
#   P>|t|        → p-vrednost (< 0.05 → prediktor je statistički značajan)
#   F-statistic  → ukupna značajnost modela
#   Cond. No.    → visok broj → moguća multikolinearnost

# Koeficijenti
print(model.params)

# 95% konfidencioni intervali
print(model.conf_int(alpha=0.05))

# RSS
rss = sum(model.resid ** 2)
print(f"RSS: {rss:.2f}")
```

---

## 4. Predikcija

```python
# Na test skupu
test_df['predicted'] = model.predict(test_df)

# Metrike
y_true = test_df['medv']
y_pred = test_df['predicted']

rss  = ((y_pred - y_true) ** 2).sum()
tss  = ((y_true - y_true.mean()) ** 2).sum()
r2   = 1 - rss / tss
rmse = np.sqrt(rss / len(y_true))

print(f"R²   = {r2:.3f}")
print(f"RMSE = {rmse:.2f}")
```

---

## 5. Dijagnostički grafikoni

```python
# 1) Residuals vs Fitted – proverava linearnost (Pretpostavka 1)
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Fitted values"); plt.ylabel("Residuals")
plt.title("Residuals vs Fitted"); plt.show()
# OK ako tačke nasumično raspoređene oko 0 (bez obrasca)

# 2) Normal Q-Q – normalnost reziduala (Pretpostavka 2)
sm.qqplot(model.resid, line='45', fit=True)
plt.title("Normal Q-Q"); plt.show()
# OK ako tačke leže duž dijagonale

# 3) Scale-Location – homoskedastičnost (Pretpostavka 3)
influence = model.get_influence()
std_resid = influence.resid_studentized_internal
abs_sqrt_resid = np.sqrt(np.abs(std_resid))
plt.scatter(model.fittedvalues, abs_sqrt_resid)
plt.axhline(np.mean(abs_sqrt_resid), color='gray', linestyle='--')
plt.title("Scale-Location"); plt.show()
# OK ako tačke ravnomerno raspoređene (bez trenda)

# 4) Residuals vs Leverage – uticajne tačke (Pretpostavka 4)
leverage = influence.hat_matrix_diag
stud_resid = influence.resid_studentized_external
cooks_d = influence.cooks_distance[0]
n = len(model.model.endog)
thresh = 4 / n

plt.scatter(leverage, stud_resid, s=1000*cooks_d, alpha=0.5)
plt.axhline(0, color='gray', linestyle='--')
plt.axhline(2, color='red', linestyle='--')
plt.axhline(-2, color='red', linestyle='--')
plt.axvline(thresh, color='red', linestyle='--')
plt.xlabel("Leverage"); plt.ylabel("Studentized Residuals")
plt.title("Residuals vs Leverage"); plt.show()
# Tačke izvan Cook's distance linije → uticajne tačke (outlieri)

# Alternativno (kraće):
sb.residplot(x=model.fittedvalues, y=model.resid, lowess=True,
             line_kws={'color': 'red'})
plt.axhline(0, color='black'); plt.title('Residuals vs Fitted'); plt.show()

sm.graphics.influence_plot(model, criterion='cooks')
plt.title('Residuals vs Leverage'); plt.show()
```

---

## 6. Multikolinearnost – VIF

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

# Pripremi X sa konstantom
X = sm.add_constant(train_df.drop(columns='medv'))

# √VIF za svaki prediktor
sqrt_vif = pd.Series(
    [vif(X.values, i) for i in range(X.shape[1])],
    index=X.columns
) ** 0.5

print(sqrt_vif.sort_values(ascending=False))
# √VIF > 2 → potencijalna multikolinearnost → izbaci jednu promenljivu i ponovi

# Nakon izbacivanja npr. 'tax':
X2 = sm.add_constant(train_df.drop(columns=['medv', 'tax']))
sqrt_vif2 = pd.Series(
    [vif(X2.values, i) for i in range(X2.shape[1])],
    index=X2.columns
) ** 0.5
print(sqrt_vif2.sort_values(ascending=False))
```

---

## 7. Scatter plot pre modeliranja

```python
plt.scatter(df['lstat'], df['medv'])
plt.xlabel('lstat'); plt.ylabel('medv')
plt.title('lstat vs medv'); plt.show()
```

---

## 8. Prikaz regresione linije

```python
plt.figure(figsize=(8, 6))
plt.scatter(train_df['lstat'], train_df['medv'],
            edgecolor='black', facecolor='none', label='Podaci')
plt.plot(train_df['lstat'], model.fittedvalues,
         color='red', label='Regresiona linija')
plt.xlabel("lstat"); plt.ylabel("medv")
plt.title("Linearna regresija: medv ~ lstat")
plt.legend(); plt.grid(True); plt.show()
```

---

## 9. Poređenje više modela

```python
lm1 = smf.ols('medv ~ lstat + rm', data=train_df).fit()
lm2 = smf.ols('medv ~ lstat + rm + ptratio', data=train_df).fit()

test_pred = test_df.copy()
test_pred['pred_lm1'] = lm1.predict(test_pred)
test_pred['pred_lm2'] = lm2.predict(test_pred)

# Poređenje R² i RMSE za svaki model
for name, pred_col in [('lm1', 'pred_lm1'), ('lm2', 'pred_lm2')]:
    rss  = ((test_pred[pred_col] - test_pred['medv']) ** 2).sum()
    tss  = ((test_pred['medv'] - test_pred['medv'].mean()) ** 2).sum()
    r2   = 1 - rss / tss
    rmse = np.sqrt(rss / len(test_pred))
    print(f"{name}: R²={r2:.3f}, RMSE={rmse:.2f}")
```

---

## Napomene

| Pojam | Šta znači |
|-------|-----------|
| R² blizu 1 | Model dobro objašnjava varijansu |
| p-vrednost < 0.05 | Prediktor je statistički značajan |
| √VIF > 2 | Moguća multikolinearnost – razmotri izbacivanje |
| Cook's D > 4/n | Uticajna tačka (outlier koji kvari model) |
| RSS → 0 | Bolje fitovanje |
| RMSE → 0 | Manja greška predikcije |
