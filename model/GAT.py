import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATRecommenderWithPrompt(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers, heads=1, dropout=0.6, prompt_dim=10):
        super(GATRecommenderWithPrompt, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.prompt_dim = prompt_dim
        
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(embedding_dim, embedding_dim, heads=heads, dropout=dropout))
        
        self.prompt = nn.Parameter(torch.randn(1, prompt_dim))
        self.prompt_proj = nn.Linear(prompt_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.1)
        nn.init.normal_(self.prompt, std=0.1)
        self.prompt_proj.reset_parameters()

    def forward(self, edge_index):
        x = self.embedding.weight
        
        # Add prompt to the initial embeddings
        prompt = self.prompt_proj(self.prompt).expand(x.size(0), -1)
        x = x + prompt
        
        layer_embeddings = [x]
        
        for conv in self.convs:
            x = F.elu(conv(self.dropout(x), edge_index))
            layer_embeddings.append(x)
        
        final_embeddings = torch.stack(layer_embeddings, dim=1).mean(dim=1)
        
        users_emb = final_embeddings[:self.num_users]
        items_emb = final_embeddings[self.num_users:]
        
        return users_emb, items_emb

    def predict_layer(self, edge_index):
        users_emb, items_emb = self.forward(edge_index)
        return torch.matmul(users_emb, items_emb.t())

    @torch.no_grad()
    def recommend(self, edge_index, user_ids, top_k):
        users_emb, items_emb = self.forward(edge_index)
        users_emb = users_emb[user_ids]
        ratings = torch.matmul(users_emb, items_emb.t())
        ratings[ratings < 0] = 0
        _, indices = torch.topk(ratings, k=top_k)
        return indices

    def create_bpr_loss(self, users, pos_items, neg_items, edge_index):
        users_emb, items_emb = self.forward(edge_index)
        
        users_emb = users_emb[users]
        pos_emb = items_emb[pos_items]
        neg_emb = items_emb[neg_items]
        
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/len(users)
        
        return loss, reg_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())