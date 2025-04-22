import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("df_final_for_lr.csv")
X = df.drop(columns=["rating_y"]).values
y = df["rating_y"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.15, random_state=42)

def objective(trial):
    n_layers = trial.suggest_int("n_layers", 2,4)
    hidden_layer_sizes = tuple(
        trial.suggest_int(f"n_units_l{i}", 4, 32, step=8) for i in range(n_layers)
    )
    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": trial.suggest_categorical("activation", ["relu"]),
        "solver": "adam",
        "alpha": trial.suggest_float("alpha", 1e-4, 1e-2, log=True),
        "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
        "early_stopping": True,
        "max_iter": 100,
        "random_state": 42
    }
    model = MLPRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
best_params = study.best_params.copy()
n_layers = best_params.pop("n_layers")
hidden_layer_sizes = tuple(best_params.pop(f"n_units_l{i}") for i in range(n_layers))
best_params["hidden_layer_sizes"] = hidden_layer_sizes

best_model = MLPRegressor(**best_params, random_state=42, max_iter=100, early_stopping=True)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Метрики нейросети (MLPRegressor):")
print("Лучшие параметры:", study.best_params)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

joblib.dump(best_model, "mlp_regressor_optuna.pkl")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Истинные значения")
plt.ylabel("Предсказания")
plt.title("Предсказание vs Истинное")

plt.subplot(1, 2, 2)
sns.histplot(y_test - y_pred, bins=30, kde=True)
plt.xlabel("Ошибка")
plt.title("Распределение ошибок")

plt.tight_layout()
plt.show()