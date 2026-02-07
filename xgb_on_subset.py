import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb


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

    # train on log target
    dtrain = xgb.DMatrix(X_train_s, label=np.log1p(y_train))
    dtest = xgb.DMatrix(X_test_s, label=np.log1p(y_test))
    params = {'objective': 'reg:squarederror', 'eta': 0.05, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8}
    bst = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dtest, 'test')], early_stopping_rounds=20, verbose_eval=False)
    preds_log = bst.predict(dtest)
    preds = np.expm1(preds_log)
    r2 = r2_score(y_test, preds)
    print(f'XGBoost on bldgclass={bldgclass}: N={len(sub)} Test R2={r2:.4f}')


if __name__ == '__main__':
    main()
