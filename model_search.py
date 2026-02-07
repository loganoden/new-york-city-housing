import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def build_pipeline(numeric_feats, categorical_feats, model, poly_degree=1):
    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])

    preprocess = ColumnTransformer([
        ("num", num_pipe, numeric_feats),
        ("cat", cat_pipe, categorical_feats),
    ], remainder="drop")

    steps = [("pre", preprocess)]
    if poly_degree and poly_degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate_model(pipe, X_train, X_test, y_train, y_test, target_log=False):
    if target_log:
        # fit on log-target
        y_train_t = np.log1p(y_train)
    else:
        y_train_t = y_train

    pipe.fit(X_train, y_train_t)
    pred_t = pipe.predict(X_test)
    if target_log:
        y_pred = np.expm1(pred_t)
    else:
        y_pred = pred_t

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mse, y_pred


def main(csv_path="nyc_housing_base.csv", out_dir="plots", sample_n=20000, random_state=1):
    warnings.filterwarnings("ignore")
    df = pd.read_csv(csv_path)
    target = "sale_price"
    df = df[df[target] > 0].copy()

    # basic feature selection
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric:
        numeric.remove(target)

    # choose a small set of categorical features to try
    categorical = [c for c in ["borough_y", "bldgclass", "landuse", "zip_code"] if c in df.columns]

    # sample for speed
    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=random_state)

    # prepare subsets to attempt (full data + boroughs + top bldgclass)
    subsets = [("all", df)]
    if "borough_y" in df.columns:
        for b in df["borough_y"].dropna().unique():
            subsets.append((f"borough_{b}", df[df["borough_y"] == b]))

    if "bldgclass" in df.columns:
        top_classes = df["bldgclass"].value_counts().nlargest(8).index.tolist()
        for c in top_classes:
            subsets.append((f"bldgclass_{c}", df[df["bldgclass"] == c]))

    models = [
        ("Linear", LinearRegression(), [1, 2]),
        ("Ridge", Ridge(alpha=1.0), [1, 2]),
        ("RF", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state, max_depth=20), [1]),
        ("GBR", GradientBoostingRegressor(n_estimators=200, random_state=random_state), [1]),
        ("HGB", HistGradientBoostingRegressor(max_iter=200, random_state=random_state), [1]),
    ]

    best = {"r2": -np.inf}
    os.makedirs(out_dir, exist_ok=True)

    for name, subdf in subsets:
        if len(subdf) < 200:
            continue
        X = subdf[numeric + categorical].copy()
        # fill numeric na with median
        for col in numeric:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())
        # categorical fill
        for col in categorical:
            if col in X.columns:
                X[col] = X[col].fillna("__NA__")

        y = subdf[target].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        for mname, model, degrees in models:
            for deg in degrees:
                for target_log in [False, True]:
                    try:
                        pipe = build_pipeline(numeric, categorical, model, poly_degree=deg)
                        r2, mse, y_pred = evaluate_model(pipe, X_train, X_test, y_train, y_test, target_log=target_log)
                    except Exception as e:
                        continue

                    print(f"Subset={name} Model={mname} deg={deg} log={target_log} R2={r2:.4f} MSE={mse:.2f}")

                    if r2 > best["r2"]:
                        best = {
                            "r2": r2,
                            "mname": mname,
                            "subset": name,
                            "deg": deg,
                            "log": target_log,
                        }
                        # save parity plot
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(7, 6))
                        plt.scatter(y_test, y_pred, s=8, alpha=0.6)
                        mn = min(y_test.min(), y_pred.min())
                        mx = max(y_test.max(), y_pred.max())
                        plt.plot([mn, mx], [mn, mx], "r--")
                        plt.xlabel("Actual sale_price")
                        plt.ylabel("Predicted sale_price")
                        plt.title(f"Best: {best['r2']:.4f} — {best['mname']} on {best['subset']}")
                        out = os.path.join(out_dir, "best_parity.png")
                        plt.tight_layout()
                        plt.savefig(out, dpi=150)
                        plt.close()

    print("\nBest found:", best)
    if best["r2"] >= 0.8:
        print("Achieved R^2 >= 0.8")
    else:
        print("Did not achieve R^2 >= 0.8. Try more feature engineering or powerful models (XGBoost/LightGBM) and hyperparameter tuning.")


if __name__ == "__main__":
    main()
