import torch
import pandas as pd
from preprocess.basicDataset import basicDataset
from typing import Callable, List, Optional
from torch_geometric.data import Data, download_url


class Yelp(basicDataset):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['user_list.txt', 'item_list.txt', 'train.txt', 'test.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self) -> None:
        data = Data()
        # Process number of nodes:
        num_users, num_items = self.getNumber()
        data.num_nodes = num_users + num_items
        # Process edge information for training and testing:
        edge_index, edge_label_index = [], []
        for path in self.raw_paths[2:]:
            rows, cols = [], []
            with open(path) as f:
                lines = f.readlines()
            for line in lines:
                indices = line.strip().split(' ')
                for dst in indices[1:]:
                    rows.append(int(indices[0]))
                    cols.append(int(dst) + num_users)  # Offset item indices
            if path == self.raw_paths[2]:  # train.txt
                edge_index.extend([rows, cols])
            else:  # test.txt
                edge_label_index.extend([rows, cols])
        data.edge_index = torch.tensor(edge_index)
        data.edge_label_index = torch.tensor(edge_label_index)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(data, self.processed_paths[0])

    def getNumber(self):
        num_users = len(pd.read_csv(self.raw_paths[0], sep=' ', header=0))
        num_items = len(pd.read_csv(self.raw_paths[1], sep=' ', header=0))
        return num_users, num_items

    def len(self) -> int:
        return 1

    def get(self) -> Data:
        return self.data
