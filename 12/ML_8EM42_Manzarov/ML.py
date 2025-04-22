import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

ratings = pd.read_csv("rating.csv")
anime = pd.read_csv("anime.csv")


df = ratings.merge(anime, on="anime_id", how="left")
df['num_words_in_name'] = df['name'].astype(str).apply(lambda x: len(x.split()))
df['genre'] = df['genre'].astype(str).apply(lambda x: [g.strip() for g in x.split(',')])

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
genre_ohe = pd.DataFrame(mlb.fit_transform(df['genre']), columns=mlb.classes_, index=df.index)

df = pd.concat([df, genre_ohe], axis=1)
df = pd.get_dummies(df, columns=['type'], prefix='type')
exclude_cols = ['name', 'genre','rating_x','user_id','anime_id']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
X = X.drop(columns=['nan'])
X = df.drop(columns=['rating_y','genre', 'name','user_id','anime_id','rating_x'])  # Признаки
Y = df['rating_y']

X.to_csv('X_cleaned.csv', index=False)
Y.to_csv('Y_cleaned.csv', index=False)
print("Сохранено в X_cleaned.csv и Y_cleaned.csv")

print(X.head())
