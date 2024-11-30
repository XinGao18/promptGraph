from torch_geometric.data import Dataset


# Basic Dataset template should implement
class basicDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 transform_args=None, pre_transform_args=None):
        super(basicDataset, self).__init__(root, transform, pre_transform)
        self.transform = transform
        self.pre_transform = pre_transform
        self.transform_args = transform_args
        self.pre_transform_args = pre_transform_args

    def download(self):
        pass

    def process(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
