import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("df_final_for_lr_cleaned.csv")
X = df.drop(columns=["rating_y"]).values
y = df["rating_y"].values
feature_names = df.drop(columns=["rating_y"]).columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 10, 500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_strength": trial.suggest_float("random_strength", 0, 1),
        "verbose": 0
    }

    model = CatBoostRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
print("Лучшие параметры:", study.best_params)
best_model = CatBoostRegressor(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# === Метрики ===
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Метрики CatBoost:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

joblib.dump(best_model, "catboost_optuna.pkl")


plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Настоящие значения (y_test)")
plt.ylabel("Предсказанные значения (y_pred)")
plt.title("📍 CatBoost: настоящие vs предсказанные")
plt.grid(True)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
plt.tight_layout()
plt.show()

importances = best_model.get_feature_importance()
sorted_idx = np.argsort(importances)[::-1]
top_n = 20

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx][:top_n], y=np.array(feature_names)[sorted_idx][:top_n])
plt.title("📌 Важность признаков (CatBoost, Top 20)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()