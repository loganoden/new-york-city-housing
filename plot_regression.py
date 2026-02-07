import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np


def main(csv_path="nyc_housing_base.csv", out_dir="plots", sample_n=10000):
    df = pd.read_csv(csv_path)
    # target column
    ycol = "sale_price"

    # features to plot (other than bldgarea)
    features = [
        "lotarea",
        "resarea",
        "comarea",
        "unitsres",
        "unitstotal",
        "numfloors",
        "building_age",
        "yearbuilt",
    ]

    # readable labels for x axis
    xlabel_map = {
        "lotarea": "Lot area (sq ft)",
        "resarea": "Residential area (sq ft)",
        "comarea": "Commercial area (sq ft)",
        "unitsres": "Residential units",
        "unitstotal": "Total units",
        "numfloors": "Number of floors",
        "building_age": "Building age (years)",
        "yearbuilt": "Year built",
    }

    # ensure output directory exists
    import os
    os.makedirs(out_dir, exist_ok=True)

    for xcol in features:
        if xcol not in df.columns or ycol not in df.columns:
            print(f"Skipping {xcol}: column not found in CSV")
            continue

        # keep only positive numeric values
        sub = df[[xcol, ycol]].dropna()
        # for yearbuilt allow > 0, for others require > 0
        try:
            sub = sub[(sub[xcol] > 0) & (sub[ycol] > 0)]
        except Exception:
            print(f"Skipping {xcol}: non-numeric data")
            continue

        # optionally sample for speed/clarity
        if sample_n and len(sub) > sample_n:
            sub = sub.sample(sample_n, random_state=1)

        X = sub[[xcol]].values.reshape(-1, 1)
        y = sub[ycol].values

        # fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)

        # build plot
        plt.figure(figsize=(8, 6))
        sns.regplot(x=xcol, y=ycol, data=sub, scatter_kws={"s": 10, "alpha": 0.3}, line_kws={"color": "red"})
        plt.xlabel(xlabel_map.get(xcol, xcol))
        plt.ylabel("Sale price (USD)")
        plt.title(f"Sale price vs {xlabel_map.get(xcol, xcol)} — slope={slope:,.2f}, R²={r2:.3f}")

        out_image = os.path.join(out_dir, f"regression_{xcol}.png")
        plt.tight_layout()
        plt.savefig(out_image, dpi=150)
        plt.close()
        print(f"Saved regression plot to {out_image}")
        print(f"Model: sale_price = {slope:.4f} * {xcol} + {intercept:.2f}")
        print(f"R^2 = {r2:.4f}")


if __name__ == "__main__":
    main()
