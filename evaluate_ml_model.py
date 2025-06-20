import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import joblib
import argparse

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

TRADES_FILE = 'Output/trade.csv'
FEATURES = ['price_change', 'volume_spike', 'RSI', 'MACD_cross', 'score']

def get_model(model_name):
    if model_name == 'logistic':
        return LogisticRegression(max_iter=1000)
    elif model_name == 'rf':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'xgb' and XGB_AVAILABLE:
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unsupported or unavailable model: {model_name}")

def evaluate_model(model_name):
    if not os.path.exists(TRADES_FILE):
        print(f"Missing trade file: {TRADES_FILE}")
        return

    df = pd.read_csv(TRADES_FILE)
    df = df.dropna(subset=FEATURES + ['result'])

    # Preprocess
    df['MACD_cross'] = df['MACD_cross'].astype(int)
    df['target'] = df['result'].apply(lambda r: 1 if r == 'WIN' else 0)

    X = df[FEATURES]
    Y = df['target']

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Model
    model = get_model(model_name) 
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Save model to disk
    model_path = f'Output/{model_name}_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Report
    report = classification_report(Y_test, Y_pred, target_names=['LOSS', 'WIN'])
    print(report)
    report_path = f'Output/ml_classification_report_{model_name}.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['LOSS', 'WIN'])
    disp.plot(cmap='Blues')
    plt.title(f"{model_name.upper()} Confusion Matrix")
    cm_path = f'Output/ml_confusion_matrix_{model_name}.png'
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # Save feature importance
    if model_name == 'logistic':
        importance = model.coef_[0]
    elif model_name == 'rf' or model_name == 'xgb':
        importance = model.feature_importances_
    else:
        importance = [0] * len(FEATURES)

    # Feature Importance
    importance_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': importance 
    }).sort_values(by='Importance', key=abs, ascending=False)

    importance_path = f'Output/ml_feature_importance_{model_name}.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")
    print(importance_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logistic", "rf", "xgb"], default="logistic")
    args = parser.parse_args()

    if args.model == 'xgb' and not XGB_AVAILABLE:
        print("xgboost not installed. Try: pip install xgboost")
    else:
        evaluate_model(args.model)
