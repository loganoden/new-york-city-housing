import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def main():
    np.random.seed(1)
    df = pd.read_csv('nyc_housing_base.csv')
    df = df[df['sale_price']>0]
    boroughs = df['borough_y'].dropna().unique()
    top_classes = df['bldgclass'].value_counts().nlargest(20).index.tolist()

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'sale_price' in numeric:
        numeric.remove('sale_price')

    candidates = []
    for b in boroughs:
        for c in top_classes:
            sub = df[(df['borough_y']==b) & (df['bldgclass']==c)]
            if len(sub) < 200:
                continue
            candidates.append((b, c, len(sub), float(sub['sale_price'].std()/sub['sale_price'].mean())))

    candidates = sorted(candidates, key=lambda x: x[3])[:20]
    print('Top candidate groups (by lowest CV):')
    for t in candidates:
        print(t)

    best = (None, -9)
    for b, c, sz, cv in candidates:
        sub = df[(df['borough_y']==b) & (df['bldgclass']==c)].copy()
        feats = [f for f in numeric if f in sub.columns]
        if 'latitude' in sub.columns and 'longitude' in sub.columns:
            feats += ['latitude', 'longitude']
        if len(feats) == 0:
            continue
        X = sub[feats].fillna(sub.median())
        y = sub['sale_price'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, max_depth=10, random_state=1)
        model.fit(X_train_s, np.log1p(y_train))
        yp = np.expm1(model.predict(X_test_s))
        r2 = r2_score(y_test, yp)
        print(f'Group {b},{c} size={sz} CV={cv:.3f} R2_logmodel={r2:.4f}')
        if r2 > best[1]:
            best = ((b, c, sz, cv), r2)

    print('\nBest group result: ', best)


if __name__ == '__main__':
    main()
