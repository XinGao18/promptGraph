import torch
import torch.nn as nn
from typing import Optional, Union


class MF(nn.Module):

    def __init__(
            self,
            num_users: int,
            num_items: int,
            mean: float = 0.,
            embedding_dim: int = 100
    ):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.item_bias = nn.Embedding(num_items, 1)

        # Init embeddings
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.user_bias.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.item_bias.weight)

    def get_embedding(self, prompt: Optional[nn.Module] = None) -> tuple:
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight

        if prompt is not None:
            user_emb = prompt.add(user_emb)
            item_emb = prompt.add(item_emb)

        return user_emb, item_emb

    def forward(
            self,
            src: torch.Tensor,
            dst: torch.Tensor,
            prompt: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Calculates model

        Args:
            src (torch.Tensor): User index
            dst (torch.Tensor): Item index
            prompt (nn.Module, optional)

        Returns:
            torch.Tensor: U-i prediction matrix
        """
        user_emb, item_emb = self.get_embedding(prompt)

        src_emb = user_emb[src]
        src_bias = self.user_bias(src).squeeze()
        dst_emb = item_emb[dst]
        dst_bias = self.item_bias(dst).squeeze()

        return torch.sum(src_emb * dst_emb, dim=1) + src_bias + dst_bias + self.mean

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_users}, '
                f'{self.num_items}, embedding_dim={self.embedding_dim})')