import openml
import pandas as pd
import os

path = "src/automl-eda/datasets/openml_datasets"
os.makedirs(path, exist_ok=True) 

def get_dataset(name: str, dataset_id: int):
    dataset_path = os.path.join(path, f"{name}_{dataset_id}.csv")

    if os.path.exists(dataset_path):
        print(f"Loading cached dataset {name} (ID {dataset_id})...")
        df = pd.read_csv(dataset_path)
    else:
        print(f"Downloading dataset {name} (ID {dataset_id})")
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        df = X.copy()
        df[dataset.default_target_attribute] = y
        df.to_csv(dataset_path, index=False)

    return df

