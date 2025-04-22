import pandas as pd

df = pd.read_csv("df_final_for_lr.csv")
df_cleaned = df.drop(columns=['episodes', 'members'])
df_cleaned.to_csv("df_final_for_lr_cleaned.csv", index=False)