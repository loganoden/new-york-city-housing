import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import os


def main(csv_path='nyc_housing_base.csv', bldgclass='D4', sample_n=20000):
    df = pd.read_csv(csv_path)
    df = df[df['sale_price']>0]
    sub = df[df['bldgclass'] == bldgclass]
    if len(sub) < 200:
        print('Subset too small:', len(sub))
        return

    numeric = sub.select_dtypes(include=[np.number]).columns.tolist()
    if 'sale_price' in numeric:
        numeric.remove('sale_price')

    if 'latitude' in sub.columns and 'longitude' in sub.columns:
        numeric += ['latitude', 'longitude']

    X = sub[numeric].select_dtypes(include=[np.number]).fillna(sub.median(numeric_only=True))
    y = sub['sale_price'].values

    if sample_n and len(X) > sample_n:
        X = X.sample(sample_n, random_state=1)
        y = X['sale_price'] if 'sale_price' in X.columns else y
        # ensure y aligns
        # simpler: resample sub earlier

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # train on log target using sklearn API for easy feature importances
    params = {'objective': 'reg:squarederror', 'learning_rate': 0.05, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8, 'n_estimators': 1000, 'random_state': 1, 'n_jobs': -1}
    model = xgb.XGBRegressor(**params)
    # fit without early stopping to avoid API differences
    model.fit(X_train_s, np.log1p(y_train))
    preds_log = model.predict(X_test_s)
    preds = np.expm1(preds_log)
    r2 = r2_score(y_test, preds)
    print(f'XGBoost on bldgclass={bldgclass}: N={len(sub)} Test R2={r2:.4f}')

    # ensure output dir
    os.makedirs('plots', exist_ok=True)

    # parity plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, s=10, alpha=0.6)
    mn = min(y_test.min(), preds.min())
    mx = max(y_test.max(), preds.max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('Actual sale_price')
    plt.ylabel('Predicted sale_price')
    plt.title(f'XGBoost parity — bldgclass={bldgclass} R²={r2:.3f}')
    plt.tight_layout()
    parity_path = os.path.join('plots', f'xgb_{bldgclass}_parity.png')
    plt.savefig(parity_path, dpi=150)
    plt.close()

    # feature importance (use feature names)
    feat_names = X.columns.tolist()
    importances = model.feature_importances_
    # sort
    idx = np.argsort(importances)[::-1]
    topk = min(20, len(feat_names))
    plt.figure(figsize=(8, 6))
    plt.barh([feat_names[i] for i in idx[:topk]][::-1], importances[idx[:topk]][::-1])
    plt.xlabel('Importance')
    plt.title(f'Feature importance — {bldgclass}')
    plt.tight_layout()
    fi_path = os.path.join('plots', f'xgb_{bldgclass}_feat_importance.png')
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f'Saved parity plot to {parity_path} and feature importance to {fi_path}')


if __name__ == '__main__':
    main()
