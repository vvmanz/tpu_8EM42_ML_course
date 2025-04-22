import optuna
import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df_final = pd.read_csv("df_final_for_lr.csv")

X_scaled = df_final.drop(columns=["rating_y"]).values
y = df_final["rating_y"].values

scaler = joblib.load("scaler.pkl")

def objective(trial):
    alpha = trial.suggest_loguniform('alpha', 0.1, 100)
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    mae = mean_absolute_error(y, y_pred)
    return mae

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Лучшие гиперпараметры:", best_params)
best_model = Ridge(alpha=best_params['alpha'])
best_model.fit(X_scaled, y)

y_pred = best_model.predict(X_scaled)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

joblib.dump(best_model, "best_linear_model_optuna.pkl")