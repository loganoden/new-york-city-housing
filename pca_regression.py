import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error


def main(csv_path="nyc_housing_base.csv", out_dir="plots", random_state=1):
    df = pd.read_csv(csv_path)
    target = "sale_price"

    num = df.select_dtypes(include=[np.number]).copy()
    if target not in num.columns:
        raise RuntimeError(f"Target column {target} not found")

    features = num.drop(columns=[target])
    features = features.dropna(axis=1, how="all")
    nunique = features.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        features = features.drop(columns=const_cols)

    data = pd.concat([features, num[target]], axis=1).dropna()

    X = data[features.columns].values
    y = data[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    max_components = min(15, X.shape[1])
    comp_grid = [2, 3, 5, 8, 10, max_components]
    comp_grid = sorted(set([c for c in comp_grid if c <= max_components]))
    degrees = [1, 2, 3]
    alphas = [0.0, 0.1, 1.0, 10.0]

    best = {
        "cv_mean": -np.inf,
        "params": None,
        "model": None,
    }

    scaler = StandardScaler()

    for n_comp in comp_grid:
        pca = PCA(n_components=n_comp, random_state=random_state)
        # transform training data to compute scale for arrow sizing later
        Xs_train = scaler.fit_transform(X_train)
        Xp_train = pca.fit_transform(Xs_train)

        for deg in degrees:
            for alpha in alphas:
                steps = []
                steps.append(("scaler", StandardScaler()))
                steps.append(("pca", PCA(n_components=n_comp, random_state=random_state)))
                if deg > 1:
                    steps.append(("poly", PolynomialFeatures(degree=deg, include_bias=False)))
                if alpha == 0.0:
                    steps.append(("reg", LinearRegression()))
                else:
                    steps.append(("reg", Ridge(alpha=alpha)))

                pipe = Pipeline(steps)

                try:
                    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
                except Exception:
                    continue

                cv_mean = float(np.mean(scores))

                if cv_mean > best["cv_mean"]:
                    best["cv_mean"] = cv_mean
                    best["params"] = {"n_components": n_comp, "degree": deg, "alpha": alpha}
                    # fit final model on full training set
                    pipe.fit(X_train, y_train)
                    best["model"] = pipe

                print(f"n_comp={n_comp} deg={deg} alpha={alpha} CV_R2={cv_mean:.4f}")

    os.makedirs(out_dir, exist_ok=True)

    print("\nBest CV result:")
    print(best["params"], "CV_R2=", best["cv_mean"])

    if best["model"] is None:
        print("No successful model found")
        return

    # evaluate on test set
    y_pred = best["model"].predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    print(f"Test R^2 = {test_r2:.4f}, MSE = {test_mse:.2f}")

    out_image = os.path.join(out_dir, "pca_regression_best.png")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, s=10, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual sale_price")
    plt.ylabel("Predicted sale_price")
    plt.title(f"Actual vs Predicted (Test) — R^2={test_r2:.4f}")
    plt.tight_layout()
    plt.savefig(out_image, dpi=150)
    plt.close()
    print(f"Saved parity plot to {out_image}")

    # report whether target reached
    target_r2 = 0.90
    if test_r2 >= target_r2 or best["cv_mean"] >= target_r2:
        print(f"Success: achieved R^2 >= {target_r2}")
    else:
        print(f"Did not reach R^2 >= {target_r2}. Best CV={best['cv_mean']:.4f}, Test={test_r2:.4f}")


if __name__ == "__main__":
    main()
