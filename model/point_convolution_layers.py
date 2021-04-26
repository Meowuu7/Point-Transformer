import torch
# from torch_cluster import fps
import torch.nn as nn
from .utils import farthest_point_sampling, get_knn_idx, batched_index_select


class TransitionDown(nn.Module):
    def __init__(self, feat_dim: int, out_feat_dim: int, k: int=16):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=out_feat_dim, eps=1e-5, momentum=0.1)
        self.k = k
        self.out_feat_dim = out_feat_dim
        self.mlp = nn.Linear(feat_dim, out_feat_dim, bias=True)
        # TODO: any other proper initialization methods?
        nn.init.xavier_uniform_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)
        self.map_fat_net = nn.Sequential(
            self.mlp,
            self.bn,
            nn.ReLU()
        )

    def forward(self, x: torch.FloatTensor, pos: torch.FloatTensor, n_sampling: int):
        # (1) get fps sampled idx; (2) get knn neighbours for each point; (3) map the original features;
        # (4) aggregate original features
        # pos = pos.float()
        bz, N = x.size(0), x.size(1)
        fps_idx = farthest_point_sampling(pos=pos[:, :, :3], n_sampling=n_sampling)
        # if len(fps_idx.size()) == 1:
        #     fps_idx = fps_idx.view(bz, n_sampling)
        knn_idx = get_knn_idx(pos=pos[:, :, :3], k=self.k, sampled_idx=fps_idx, n_sampling=n_sampling)
        # pos.size() = bz x N x 3
        # knn_idx.size() = bz x N x k
        # fps_idx.size() = bz x n_sampling
        # x.size() = bz x N x feat_dim
        # x = self.map_fat_net(x) # map the feature x
        for net_blk in self.map_fat_net:
            if not isinstance(net_blk, nn.BatchNorm1d):
                x = net_blk(x)
            else:
                x = torch.transpose(x, 1, 2)
                x = net_blk(x)
                x = torch.transpose(x, 1, 2)
        # print(N, n_sampling)
        x_expand = x.view(bz, 1, N, -1).repeat(1, n_sampling, 1, 1)
        gather_knn_x = batched_index_select(values=x_expand, indices=knn_idx, dim=2)
        # gather_knn_x.size() = bz x N x k x feat_dim
        # max pooling to get feature vectors for sampled points
        gather_knn_x, _ = gather_knn_x.max(dim=2, keepdim=False)
        # bz x N x feat_dim

        # sampled_knn_x = batched_index_select(values=gather_knn_x.view(bz * N, -1), indices=fps_idx, dim=0)
        # sampled_knn_x = sampled_knn_x.view(bz, n_sampling, -1)
        sampled_knn_x = gather_knn_x
        # bz x n_sampling x feat_dim
        sampled_pos = batched_index_select(values=pos.view(bz * N, -1), indices=fps_idx, dim=0)
        sampled_pos = sampled_pos.view(bz, n_sampling, -1)
        return sampled_knn_x, sampled_pos # return the sampled knn

