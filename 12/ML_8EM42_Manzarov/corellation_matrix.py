import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)

df = pd.read_csv("df.csv", low_memory=False)

df_filtered = df[(df["rating_y"] != -1) & (~df["rating_y"].isna())]

# Удалим неинформативный признак
df_filtered = df_filtered.drop(columns=["name"])

# Обработка 'episodes' — превращаем в число, Unknown -> NaN
df_filtered["episodes"] = pd.to_numeric(df_filtered["episodes"], errors='coerce')

# Разбиваем 'genre' на бинарные признаки
# Разделяем строки с жанрами
df_filtered["genre"] = df_filtered["genre"].fillna("")
genre_dummies = df_filtered["genre"].str.get_dummies(sep=", ")

# Кодируем 'type' (TV, Movie и т.п.)
type_dummies = pd.get_dummies(df_filtered["type"], prefix="type")

# Объединяем всё в одну таблицу
df_final = pd.concat([df_filtered.drop(columns=["genre", "type"]), genre_dummies, type_dummies], axis=1)

# Удаляем строки с NaN (например, если episodes не смогли преобразоваться)
df_final = df_final.dropna()

# Корреляционная матрица
corr_matrix = df_final.corr()

# Корреляции с целевой переменной
target_corr = corr_matrix["rating_y"].sort_values(ascending=False)

# Печать корреляций
print("Корреляция с rating_y:")
print(target_corr)

# Тепловая карта (по желанию)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Корреляционная матрица (включая жанры и типы)")
plt.show()




