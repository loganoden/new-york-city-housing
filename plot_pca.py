import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main(csv_path="nyc_housing_base.csv", out_image="plots/pca_scatter.png", sample_n=10000, random_state=1):
    df = pd.read_csv(csv_path)

    # target for coloring
    target = "sale_price"

    # select numeric features and drop the target
    num = df.select_dtypes(include=[np.number]).copy()
    if target not in num.columns:
        raise RuntimeError(f"Target column {target} not found in CSV")

    features = num.drop(columns=[target])

    # drop columns with all NaN or constant values
    features = features.dropna(axis=1, how="all")
    nunique = features.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        features = features.drop(columns=const_cols)

    # align target and features, drop rows with any NA
    data = pd.concat([features, num[target]], axis=1).dropna()

    # optionally sample for speed
    if sample_n and len(data) > sample_n:
        data = data.sample(sample_n, random_state=random_state)

    X = data[features.columns].values
    y = data[target].values

    # scale features before PCA
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=random_state)
    pcs = pca.fit_transform(Xs)

    evr = pca.explained_variance_ratio_

    # output dir
    os.makedirs(os.path.dirname(out_image) or ".", exist_ok=True)

    # build scatter plot colored by sale_price (log scale for readability)
    plt.figure(figsize=(10, 8))
    cmap = sns.color_palette("viridis", as_cmap=True)
    sc = plt.scatter(pcs[:, 0], pcs[:, 1], c=np.log1p(y), s=20, cmap=cmap, alpha=0.7)
    plt.xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    plt.title("PCA projection of numeric features colored by log(sale_price)")
    cbar = plt.colorbar(sc)
    cbar.set_label("log(1 + sale_price)")

    # draw simple biplot arrows for the top contributing features
    loadings = pca.components_.T
    # scale arrows to fit the scatter
    arrow_scale = np.percentile(np.abs(pcs), 90)
    for i, feat in enumerate(features.columns):
        lx = loadings[i, 0]
        ly = loadings[i, 1]
        plt.arrow(0, 0, lx * arrow_scale, ly * arrow_scale, color="red", alpha=0.6, head_width=0.03 * arrow_scale)
        plt.text(lx * arrow_scale * 1.05, ly * arrow_scale * 1.05, feat, color="red", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_image, dpi=150)
    plt.close()
    print(f"Saved PCA scatter to {out_image}")


if __name__ == "__main__":
    main()
