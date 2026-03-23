from dataset.cwru.sehri_et_al import get_list_of_papers_X_y
import numpy as np

def print_meta_data():
    list_of_X_y = get_list_of_papers_X_y()
    for i, (X, y) in enumerate(list_of_X_y):
        labels, counts = np.unique(y, return_counts=True)
        label_counts = {label: int(count) for label, count in zip(labels.tolist(), counts.tolist())}
        print(f"Fold {i}: X shape = {X.shape}, y shape = {y.shape}")
        print(f"Labels = {label_counts}")