import os
import pandas as pd

from automl_eda.data_loader.dataset_manager import get_datasets
from automl_eda.meta_dataset_builder.feature_extractor import get_meta_features

path = "meta_datasets/data.csv"
os.makedirs("meta_datasets", exist_ok=True)

def build_dataset():
    datasets = get_datasets()
    all_meta = []

    for name, df in datasets.items():
        target_name = df.columns[-1]
        meta_df = get_meta_features(df, target_name, name)

        if "regression" in name:
            meta_df["task_type"] = "regression"
        else:
            meta_df["task_type"] = "classification"

        all_meta.append(meta_df)

    final_meta = pd.concat(all_meta, ignore_index=True)
    final_meta.to_csv(path, index=False)
    print(f"Saved column-level meta-dataset to {path}")

if __name__ == "__main__":
    build_dataset()
