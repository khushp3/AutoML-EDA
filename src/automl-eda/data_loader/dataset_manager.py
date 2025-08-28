from .loader import get_dataset

def get_datasets():
    datasets = {}
    openml_datasets = [
        ("adult_classification", 1590), 
        ("iris_classification", 61),    
        ("abalone_regression", 183),    
        ("boston_regression", 531),     
    ]

    for name, dataset_id in openml_datasets:
        datasets[name] = get_dataset(name, dataset_id)

    return datasets
