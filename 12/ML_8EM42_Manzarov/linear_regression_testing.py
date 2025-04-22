import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")
df_final = pd.read_csv("df_final_for_lr.csv")

X_scaled = df_final.drop(columns=["rating_y"]).values
y = df_final["rating_y"].values

y_pred = model.predict(X_scaled)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", lw=2)
plt.xlabel("Реальные значения")
plt.ylabel("Предсказанные значения")
plt.title("Реальные vs Предсказанные значения")
plt.show()

residuals = y - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color="red", lw=2)
plt.xlabel("Предсказанные значения")
plt.ylabel("Остатки")
plt.title("Предсказанные значения vs Остатки")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor="black")
plt.xlabel("Остатки")
plt.ylabel("Частота")
plt.title("Гистограмма остатков")
plt.show()