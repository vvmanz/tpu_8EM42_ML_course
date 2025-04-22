import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


model = joblib.load("decision_tree_optuna.pkl")

df = pd.read_csv("df_final_for_lr.csv")
X = df.drop(columns=["rating_y"]).values
y = df["rating_y"].values

y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\n📦 Метрики на всех данных (загруженной моделью):")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))
r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print("\n📉 Кросс-валидация (5-Fold):")
print("MAE: %.4f ± %.4f" % (mae_scores.mean(), mae_scores.std()))
print("RMSE: %.4f ± %.4f" % (rmse_scores.mean(), rmse_scores.std()))
print("R²: %.4f ± %.4f" % (r2_scores.mean(), r2_scores.std()))

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y, y=y_pred, alpha=0.5)
plt.xlabel("🎯 Настоящие значения")
plt.ylabel("📈 Предсказанные значения")
plt.title("💡 Реальное vs Предсказанное")
plt.grid(True)
plt.plot([min(y), max(y)], [min(y), max(y)], color="red", linestyle="--", label="Идеал")
plt.legend()
plt.tight_layout()
plt.show()

residuals = y - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=50, kde=True, color="tomato")
plt.title("🧾 Распределение остатков (y - y_pred)")
plt.xlabel("Ошибка")
plt.ylabel("Количество")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.boxplot(y=residuals, color="skyblue")
plt.title("📦 Boxplot ошибок предсказания")
plt.ylabel("Остатки")
plt.grid(True)
plt.tight_layout()
plt.show()