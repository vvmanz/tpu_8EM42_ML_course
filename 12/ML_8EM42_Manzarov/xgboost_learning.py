import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
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
        "n_estimators": trial.suggest_int("n_estimators", 10, 600),
        "max_depth": trial.suggest_int("max_depth", 5, 9),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "verbosity": 0
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
print("Лучшие параметры:", study.best_params)
best_model = XGBRegressor(**study.best_params)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Метрики XGBoost:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

joblib.dump(best_model, "xgboost_optuna.pkl")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Настоящие значения (y_test)")
plt.ylabel("Предсказанные значения (y_pred)")
plt.title("📍 XGBoost: настоящие vs предсказанные")
plt.grid(True)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
plt.tight_layout()
plt.show()

importances = best_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
top_n = 20

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx][:top_n], y=np.array(feature_names)[sorted_idx][:top_n])
plt.title("📌 Важность признаков (XGBoost, Top 20)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()