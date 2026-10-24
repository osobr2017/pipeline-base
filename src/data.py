import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def get_dataloaders(batch_size=16, val_size=0.2, seed=42):
    X, y = load_iris(return_X_y=True)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_tr)   # solo con train
    X_tr, X_val = scaler.transform(X_tr), scaler.transform(X_val)

    def to_loader(Xn, yn, shuffle):
        ds = TensorDataset(torch.tensor(Xn, dtype=torch.float32),
                           torch.tensor(yn, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (to_loader(X_tr, y_tr, True), to_loader(X_val, y_val, False),
            X.shape[1], len(np.unique(y)))