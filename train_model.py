import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import optuna

DATASET_PATH = "training_data/training_dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def encode_labels(df):
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    return df, le

def objective(trial, X, y, n_classes):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'objective': 'multi:softprob' if n_classes > 2 else 'binary:logistic',
        'eval_metric': 'mlogloss' if n_classes > 2 else 'logloss',
        'base_score': 0.5
    }
    model = XGBClassifier(**params)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    return f1_score(y_valid, y_pred, average='weighted')

def train_model_for_symbol(symbol, df):
    try:
        df_symbol = df[df['symbol'] == symbol].dropna()
        if len(df_symbol) < 50:
            print(f"⛔ Dataset terlalu sedikit untuk {symbol}, dilewati.")
            return

        df_symbol, label_encoder = encode_labels(df_symbol)

        features = ['rsi', 'macd', 'signal_line', 'support', 'resistance']
        X = df_symbol[features]
        y = df_symbol['label']
        n_classes = len(set(y))

        if len(df_symbol) > 500:
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, X, y, n_classes), n_trials=10)
            best_params = study.best_params
            best_params['objective'] = 'multi:softprob' if n_classes > 2 else 'binary:logistic'
            best_params['eval_metric'] = 'mlogloss' if n_classes > 2 else 'logloss'
            best_params['base_score'] = 0.5
            model = XGBClassifier(**best_params)
        else:
            model = XGBClassifier(
                objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
                eval_metric='mlogloss' if n_classes > 2 else 'logloss',
                base_score=0.5,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1
            )

        model.fit(X, y)

        name = symbol.replace("/", "")
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}_model.pkl"))
        joblib.dump(label_encoder, os.path.join(MODEL_DIR, f"{name}_label_encoder.pkl"))
        print(f"✅ Model untuk {symbol} berhasil disimpan.")
    except Exception as e:
        print(f"❌ Gagal melatih model untuk {symbol}: {e}")

def main():
    if not os.path.exists(DATASET_PATH):
        print("❌ File dataset tidak ditemukan.")
        return

    df = pd.read_csv(DATASET_PATH)
    if 'symbol' not in df.columns:
        print("❌ Kolom 'symbol' tidak ditemukan dalam dataset.")
        return

    symbols = df['symbol'].unique()
    for symbol in symbols:
        train_model_for_symbol(symbol, df)

if __name__ == "__main__":
    main()
