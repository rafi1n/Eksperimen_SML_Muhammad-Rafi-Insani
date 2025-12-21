import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

TARGET = "Survived"

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET not in df.columns:
        raise ValueError("Kolom 'Survived' tidak ditemukan. Pastikan memakai Titanic train.csv Kaggle.")

    # pisahkan fitur dan target
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].copy()

    # drop kolom yang biasanya tidak dipakai
    drop_cols = [c for c in ["Name", "Ticket", "Cabin"] if c in X.columns]
    X = X.drop(columns=drop_cols)

    # pisahkan numerik vs kategorikal
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )

    # fit-transform (untuk Kriteria 1)
    X_processed = preprocessor.fit_transform(X)

    # nama fitur hasil preprocessing
    feature_names = []
    feature_names.extend(num_cols)

    if len(cat_cols) > 0:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(list(ohe.get_feature_names_out(cat_cols)))

    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    # gabungkan kembali dengan target
    processed_df = pd.concat(
        [y.reset_index(drop=True), X_processed_df.reset_index(drop=True)],
        axis=1
    )
    return processed_df

def main():
    # input raw
    raw_path = os.environ.get("RAW_PATH", "titanic_raw/titanic.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset tidak ditemukan: {raw_path}")

    df = pd.read_csv(raw_path)
    processed_df = preprocess(df)

    # output folder sesuai rubric
    out_dir = os.environ.get("OUT_DIR", "preprocessing/titanic_preprocessing")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "titanic_processed.csv")
    processed_df.to_csv(out_csv, index=False)

    # simpan daftar kolom fitur untuk Kriteria 2
    feat_path = os.path.join(out_dir, "feature_columns.txt")
    feature_cols = [c for c in processed_df.columns if c != TARGET]
    with open(feat_path, "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(col + "\n")

    print("Saved:", out_csv)
    print("Saved:", feat_path)
    print("Shape:", processed_df.shape)

if __name__ == "__main__":
    main()
