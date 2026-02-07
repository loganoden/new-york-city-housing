import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np


def main(csv_path="nyc_housing_base.csv", out_image="regression_plot.png", sample_n=10000):
    df = pd.read_csv(csv_path)
    # choose predictors and target
    xcol = "bldgarea"
    ycol = "sale_price"

    # keep only positive numeric values
    df = df[[xcol, ycol]].dropna()
    df = df[(df[xcol] > 0) & (df[ycol] > 0)]

    # optionally sample for speed/clarity
    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=1)

    X = df[[xcol]].values.reshape(-1, 1)
    y = df[ycol].values

    # fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    # build plot
    plt.figure(figsize=(8, 6))
    sns.regplot(x=xcol, y=ycol, data=df, scatter_kws={"s": 10, "alpha": 0.3}, line_kws={"color": "red"})
    plt.xlabel("Building area (sq ft)")
    plt.ylabel("Sale price (USD)")
    plt.title(f"Sale price vs Building area — slope={slope:,.2f}, R²={r2:.3f}")

    # improve layout and save
    plt.tight_layout()
    plt.savefig(out_image, dpi=150)
    print(f"Saved regression plot to {out_image}")
    print(f"Model: sale_price = {slope:.4f} * bldgarea + {intercept:.2f}")
    print(f"R^2 = {r2:.4f}")


if __name__ == "__main__":
    main()
