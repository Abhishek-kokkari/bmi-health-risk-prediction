import pandas as pd
import os


# Load Data
def load_data():
    female = pd.read_csv(
        "data/raw/nhanes_adult_female_bmx_2020.csv",
        comment='#',        # ignore metadata lines
        quotechar='"'       # handle quoted header
    )
    female['Gender'] = 1  # 1 for Female

    male = pd.read_csv(
        "data/raw/nhanes_adult_male_bmx_2020.csv",
        comment='#',
        quotechar='"'
    )
    male['Gender'] = 0  # 0 for Male

    # Combine datasets
    df = pd.concat([female, male], ignore_index=True)

    print("Data Loaded:", df.shape)
    print("Columns:", df.columns)

    return df


# Clean Data
def clean_data(df):
    # Remove missing values
    df = df.dropna()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Create BMI (IMPORTANT)
    if 'BMXWT' in df.columns and 'BMXHT' in df.columns:
        df['BMI'] = df['BMXWT'] / ((df['BMXHT'] / 100) ** 2)
        print("BMI column created")
    else:
        raise Exception("Required columns BMXWT/BMXHT not found")

    return df


# Save Cleaned Data
def save_data(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned_data.csv", index=False)
    print("Cleaned data saved at data/processed/")


# 🚀 MAIN EXECUTION
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    save_data(df)