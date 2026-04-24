import pandas as pd
df = pd.read_csv("data/processed/cleaned_data.csv")
df['Risk'] = (
    (df['BMXWAIST'] > 90) |
    (df['BMXHIP'] > 100)
).astype(int)
print(df['Risk'].value_counts(normalize=True))
