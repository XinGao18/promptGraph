import torch
import pandas as pd
from preprocess.basicDataset import basicDataset
from typing import Callable, List, Optional
from torch_geometric.data import Data, download_url, extract_zip
import os
import os.path as osp

MOVIE_HEADERS = ["movieId", "title", "genres"]
USER_HEADERS = ["userId", "gender", "age", "occupation", "zipCode"]
RATING_HEADERS = ['userId', 'movieId', 'rating', 'timestamp']

class MovieLens1M(basicDataset):
    r"""The MovieLens 1M rating dataset, assembled by GroupLens Research from the MovieLens web site,
    consisting of movies (3,883 nodes) and users (6,040 nodes) with approximately 1 million ratings between them.
    User ratings for movies are available as ground truth labels.
    Features of users and movies are encoded according to the "Inductive Matrix Completion Based on Graph Neural Networks" paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in a
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None) -> None:
        self.root = osp.join(root, 'ml-1m')  # 更新 root 路径
        super().__init__(self.root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['movies.dat', 'users.dat', 'ratings.dat']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.remove(path)
        # 创建 ml-1m 文件夹（如果不存在）
        ml1m_dir = osp.join(self.root, 'ml-1m')
        os.makedirs(ml1m_dir, exist_ok=True)
        # 创建 raw 文件夹并移动原始数据
        raw_dir = osp.join(ml1m_dir, 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        for file in self.raw_file_names:
            os.rename(osp.join(self.root, file), osp.join(raw_dir, file))
        # 创建 processed 文件夹
        processed_dir = osp.join(ml1m_dir, 'processed')
        os.makedirs(processed_dir, exist_ok=True)

    def process(self) -> None:
        data = Data()

        # Process movie data:
        movie_df = pd.read_csv(
            self.raw_paths[0],
            sep='::',
            header=None,
            index_col='movieId',
            names=MOVIE_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
        )
        movie_mapping = {idx: i for i, idx in enumerate(movie_df.index)}

        genres = movie_df['genres'].str.get_dummies('|').values
        data.movie_x = torch.from_numpy(genres).to(torch.float)

        # Process user data:
        user_df = pd.read_csv(
            self.raw_paths[1],
            sep='::',
            header=None,
            index_col='userId',
            names=USER_HEADERS,
            dtype='str',
            encoding='ISO-8859-1',
            engine='python',
        )
        user_mapping = {idx: i for i, idx in enumerate(user_df.index)}

        age = user_df['age'].str.get_dummies().values
        gender = user_df['gender'].str.get_dummies().values
        occupation = user_df['occupation'].str.get_dummies().values
        data.user_x = torch.cat([
            torch.from_numpy(age).to(torch.float),
            torch.from_numpy(gender).to(torch.float),
            torch.from_numpy(occupation).to(torch.float)
        ], dim=-1)

        # Process rating data:
        rating_df = pd.read_csv(
            self.raw_paths[2],
            sep='::',
            header=None,
            names=RATING_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
        )

        src = [user_mapping[idx] for idx in rating_df['userId']]
        dst = [movie_mapping[idx] for idx in rating_df['movieId']]
        data.edge_index = torch.tensor([src, dst])

        data.rating = torch.from_numpy(rating_df['rating'].values).to(torch.long)
        data.time = torch.from_numpy(rating_df['timestamp'].values)

        data.num_nodes = len(user_mapping) + len(movie_mapping)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, self.processed_paths[0])

    def len(self) -> int:
        return 1

    def get(self) -> Data:
        return self.data
