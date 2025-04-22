import pandas as pd

pd.set_option('display.max_columns', None)
df = pd.read_csv("df_final_for_lr.csv")

print(df.head(5))