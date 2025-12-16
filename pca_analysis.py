import os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config import PROCESSED_DATA_PATH, MODELS_DIR, SEED

FEATURES_FILE = os.path.join(PROCESSED_DATA_PATH, "features_training.parquet")
N_COMPONENTS = 12


def main():
    print("Running PCA...")

    df = pd.read_parquet(FEATURES_FILE)

    # Keep numeric features only
    X = (
        df.select_dtypes(include=[np.number])
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0)
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(N_COMPONENTS, X_scaled.shape[1]), random_state=SEED)
    pca.fit(X_scaled)

    os.makedirs(MODELS_DIR, exist_ok=True)

    dump(
        {
            "scaler": scaler,
            "pca": pca,
            "feature_names": X.columns.tolist(),
        },
        os.path.join(MODELS_DIR, "pca_model.joblib"),
    )

    ev = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })

    ev.to_csv(os.path.join(PROCESSED_DATA_PATH, "pca_explained_variance.csv"), index=False)

    print("✅ PCA complete")
    print(ev.head(10))


if __name__ == "__main__":
    main()
