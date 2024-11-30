import numpy as np
import torch
from torch.nn.functional import pairwise_distance


def InBP(user_emb, item_emb, interactions, num_shards, config, max_epochs):
    """
    :param interactions: The size of train interactions is (2, num_interactions)
    """
    with torch.no_grad():
        shards = [[] for _ in range(num_shards)]
        num_users = config['num_users']
        max_shard_size = 1.2 * interactions.size(1) // num_shards
        random_indices = torch.randperm(interactions.size(1))[:num_shards]
        anchors = interactions[:, random_indices]
        # Embedding of anchors
        anchors_emb = []
        for i in range(num_shards):
            temp_u = user_emb[anchors[0][i]]
            temp_i = item_emb[anchors[1][i]-num_users]
            anchors_emb.append(torch.cat((temp_u, temp_i), 0))
        anchors_emb = torch.stack(anchors_emb)

        while max_epochs:
            for shard in shards:
                shard.clear()

            # Calculate distances between interactions and shards' centers
            distances = {}
            for i in range(interactions.size(1)):
                for j in range(num_shards):
                    uid_emb = user_emb[interactions[0][i]]
                    aid_emb = anchors_emb[j][:config['embedding_dim']]
                    score_u = pairwise_distance(uid_emb, aid_emb)
                    iid_emb = item_emb[interactions[1][i] - num_users]
                    aid_emb = anchors_emb[j][config['embedding_dim']:]
                    score_i = pairwise_distance(iid_emb, aid_emb)
                    distances[i, j] = score_u * score_i
            sorted_indices = sorted(distances.items(), key=lambda x: x[1], reverse=True)

            # Assign interactions to shards
            assigned_interactions = set()
            for (i, j), distance in sorted_indices:
                if i in assigned_interactions:
                    continue
                if len(shards[j]) < max_shard_size:
                    shards[j].append(i)  # include the index of interaction rather than two entities
                    assigned_interactions.add(i)

            # Update the centers of shards
            for j in range(num_shards):
                if len(shards[j]) == 0:
                    continue
                shard_emb = []
                for i in shards[j]:
                    uid_emb = user_emb[interactions[0][i]]
                    iid_emb = item_emb[interactions[1][i] - num_users]
                    shard_emb.append(torch.cat((uid_emb, iid_emb), 0))
                shard_emb = torch.stack(shard_emb)
                anchors_emb[j] = torch.mean(shard_emb, 0)
        max_epochs -= 1
    return shards
