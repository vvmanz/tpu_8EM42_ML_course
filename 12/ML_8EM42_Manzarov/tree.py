import optuna
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("df_final_for_lr.csv")
X = df.drop(columns=["rating_y"]).values
y = df["rating_y"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    }
    model = DecisionTreeRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)  # минимизируем MAE

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("Лучшие параметры:", study.best_params)
best_model = DecisionTreeRegressor(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Метрики дерева решений:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

joblib.dump(best_model, "decision_tree_optuna.pkl")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Настоящие значения (y_test)")
plt.ylabel("Предсказанные значения (y_pred)")
plt.title("🎯 Сравнение настоящих и предсказанных значений")
plt.grid(True)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
plt.tight_layout()
plt.show()

importances = best_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
top_n = 20

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx][:top_n], y=np.array(feature_names)[sorted_idx][:top_n])
plt.title("🔥 Важность признаков (Top 20)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()