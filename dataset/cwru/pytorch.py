import torch
from torch.utils.data import Dataset

class CWRUDataset(Dataset):
    """PyTorch Dataset wrapper for CWRU bearing fault data."""
    
    def __init__(self, X, y, transform=None):
        """
        Args:
            X: Input features from get_list_of_papers_X_y
            y: Labels from get_list_of_papers_X_y
            transform: Optional transforms to apply to samples
        """
        self.X = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X
        self.y = torch.LongTensor(y) if not isinstance(y, torch.Tensor) else y
        self.transform = transform
        
        assert len(self.X) == len(self.y), "X and y must have same length"
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
