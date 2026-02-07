import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main(csv_path="nyc_housing_base.csv", out_dir="plots/by_borough", sample_n=10000, random_state=1):
    df = pd.read_csv(csv_path)
    if "borough_y" not in df.columns:
        raise RuntimeError("CSV does not contain 'borough_y' column")

    # select numeric features except the target
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    target = "sale_price"
    if target in numeric:
        numeric.remove(target)

    # exclude obvious ID-like columns
    exclude = ["block", "lot"]
    features = [c for c in numeric if c not in exclude]

    os.makedirs(out_dir, exist_ok=True)

    for feat in features:
        sub = df[[feat, target, "borough_y"]].dropna()
        # require positive for both feature and target where numeric
        try:
            sub = sub[(sub[feat] > 0) & (sub[target] > 0)]
        except Exception:
            # non-numeric or non-comparable — skip
            print(f"Skipping {feat}: incompatible values")
            continue

        # optionally sample
        if sample_n and len(sub) > sample_n:
            sub = sub.sample(sample_n, random_state=random_state)

        # seaborn lmplot with hue by borough (separate regression per borough)
        plt.figure(figsize=(8, 6))
        try:
            sns.scatterplot(x=feat, y=target, hue="borough_y", data=sub, alpha=0.5, s=20)
            # draw overall regression line
            sns.regplot(x=feat, y=target, data=sub, scatter=False, color="k", line_kws={"linewidth":1.0})
        except Exception as e:
            print(f"Plot failed for {feat}: {e}")
            continue

        plt.xlabel(feat)
        plt.ylabel("Sale price (USD)")
        plt.title(f"Sale price vs {feat} — colored by borough")
        plt.legend(title="borough", loc="best")
        plt.tight_layout()
        out_image = os.path.join(out_dir, f"regression_{feat}_by_borough.png")
        plt.savefig(out_image, dpi=150)
        plt.close()
        print(f"Saved {out_image}")


if __name__ == "__main__":
    main()
