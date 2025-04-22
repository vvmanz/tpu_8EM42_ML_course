import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import plot_tree
from catboost import CatBoostRegressor
import xgboost as xgb
import os

df = pd.read_csv("df_final_for_lr.csv")
X = df.drop(columns=["rating_y"]).values
y = df["rating_y"].values

models = {
    "LinearRegression": joblib.load("linear_model.pkl"),
    "DecisionTree": joblib.load("decision_tree_optuna.pkl"),
    "CatBoost": joblib.load("catboost_optuna.pkl"),
    "XGBoost": joblib.load("xgboost_optuna.pkl"),
    "MLP": joblib.load("mlp_regressor_optuna.pkl")
}

results = []

def get_metrics(model_name, model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    results.append({
        "Model": model_name,
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    })
    print(f"==== {model_name} ====")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return y_pred

y_pred_lr = get_metrics("LinearRegression", models["LinearRegression"], X, y)
weights = models["LinearRegression"].coef_
plt.figure(figsize=(10, 4))
sns.barplot(x=weights, y=df.drop(columns=["rating_y"]).columns)
plt.title("Весы линейной регрессии")
plt.tight_layout()
plt.savefig("linear_weights.png")
plt.close()

y_pred_dt = get_metrics("DecisionTree", models["DecisionTree"], X, y)
plt.figure(figsize=(12, 6))
plot_tree(models["DecisionTree"], feature_names=df.drop(columns=["rating_y"]).columns, filled=True, max_depth=2)
plt.title("Первые узлы дерева решений")
plt.savefig("decision_tree_nodes.png")
plt.close()

y_pred_cb = get_metrics("CatBoost", models["CatBoost"], X, y)
feature_importance_cb = models["CatBoost"].get_feature_importance()
cb_importance_df = pd.DataFrame({
    "Feature": df.drop(columns=["rating_y"]).columns,
    "Importance": feature_importance_cb
}).sort_values("Importance", ascending=False)
cb_importance_df.to_csv("catboost_feature_importance.csv", index=False)

y_pred_xgb = get_metrics("XGBoost", models["XGBoost"], X, y)
xgb.plot_importance(models["XGBoost"], importance_type='weight', max_num_features=10)
plt.title("XGBoost Feature Importance")
plt.savefig("xgboost_feature_importance.png")
plt.close()

y_pred_mlp = get_metrics("MLP", models["MLP"], X, y)

mlp_model = models["MLP"]
for i, coefs in enumerate(mlp_model.coefs_):
    plt.figure()
    plt.hist(coefs.flatten(), bins=30)
    plt.title(f"Гистограмма весов слоя {i} (вход → скрытый / скрытый → выход)")
    plt.savefig(f"mlp_weights_layer_{i}.png")
    plt.close()

for i, biases in enumerate(mlp_model.intercepts_):
    plt.figure()
    plt.hist(biases.flatten(), bins=30)
    plt.title(f"Гистограмма смещений слоя {i}")
    plt.savefig(f"mlp_biases_layer_{i}.png")
    plt.close()

df_results = pd.DataFrame(results)
df_results.to_csv("model_metrics_summary.csv", index=False)
print("\nСводная таблица:")
print(df_results)

best_model_row = df_results.sort_values("R2", ascending=False).iloc[0]
print(f"\nЛучшая модель: {best_model_row['Model']} (R2 = {best_model_row['R2']:.4f})")