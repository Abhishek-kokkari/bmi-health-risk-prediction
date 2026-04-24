import os
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Manually set to avoid physical core detection warning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle


def load_data():
    df = pd.read_csv("data/processed/cleaned_data.csv")
    return df


def prepare_data(df):
    df = df.dropna()
    df.columns = df.columns.str.strip()

    df['Risk'] = (
        (df['BMXWAIST'] > 90) |
        (df['BMXHIP'] > 100)
    ).astype(int)

    features = ['BMXWT', 'BMXHT', 'BMI', 'BMXARML', 'BMXLEG', 'BMXARMC', 'Gender']
    X = df[features]
    y = df['Risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train):
    # Ensemble of two powerful models
    clf1 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)
    clf2 = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    
    model = VotingClassifier(
        estimators=[('hgb', clf1), ('rf', clf2)],
        voting='soft'
    )
    
    model.fit(X_train, y_train)
    print("Model trained using Voting Ensemble (HistGradientBoosting + RandomForest)")
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def save_model(model, scaler):
    os.makedirs("models", exist_ok=True)
    pickle.dump(model, open("models/model.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    print("Model saved")


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
    save_model(model, scaler)