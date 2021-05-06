from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, labels):
        """Initialization"""
        self.labels = labels
        self.data = data

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""
        X = self.data[index]
        y = self.labels[index]

        return X, y
