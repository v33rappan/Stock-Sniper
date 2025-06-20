import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import joblib

TRADES_FILE = 'Output/trade.csv'
MODEL_REPORT = 'Output/ml_classification_report.txt'
CONF_MATRIX_PNG = 'Output/ml_confusion_matrix.png'
FEATURE_IMPORTANCE_CSV = 'Output/ml_feature_importance.csv'
MODEL_FILE = 'Output/logistic_model.pkl'

def train_ml_model():
    if not os.path.exists(TRADES_FILE):
        print(f"Missing trade file: {TRADES_FILE}")
        return

    df = pd.read_csv(TRADES_FILE)
    df = df.dropna(subset=['price_change', 'volume_spike', 'RSI', 'MACD_cross', 'score', 'result'])

    # Preprocess
    df['MACD_cross'] = df['MACD_cross'].astype(int)
    df['target'] = df['result'].apply(lambda r: 1 if r == 'WIN' else 0)

    features = ['price_change', 'volume_spike', 'RSI', 'MACD_cross', 'score']
    X = df[features]
    Y = df['target']

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Save model to disk
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    # Report
    report = classification_report(Y_test, Y_pred, target_names=['LOSS', 'WIN'])
    print(report)
    with open(MODEL_REPORT, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {MODEL_REPORT}")

    # Confusion Matrix
    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['LOSS', 'WIN'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(CONF_MATRIX_PNG)
    print(f"Confusion matrix saved to {CONF_MATRIX_PNG}")

    # Feature Importance
    importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', key=abs, ascending=False)
    importance.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
    print(f"Feature importance saved to {FEATURE_IMPORTANCE_CSV}")
    print(importance)

if __name__ == '__main__':
    train_ml_model()
