import torch
# from torch_cluster import fps
import torch.nn as nn
from .utils import farthest_point_sampling, get_knn_idx, batched_index_select
# from .utils import three_interpolate, three_nn


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


class TransitionUp(nn.Module):
    def __init__(self, fea_in: int, fea_out: int):
        super(TransitionUp, self).__init__()
        self.mlp_in = nn.Sequential(
            nn.Linear(fea_in, fea_out, bias=True),
            nn.BatchNorm1d(fea_out),
            nn.ReLU()
        )

        self.mlp_out = nn.Sequential(
            nn.Linear(fea_out, fea_out, bias=True),
            nn.BatchNorm1d(fea_out),
            nn.ReLU()
        )

    def apply_module_with_bn(self, x, modle):
        for layer in modle:
            if isinstance(layer, nn.BatchNorm1d):
                x = torch.transpose(x, 1, 2)
                x = layer(x)
                x = torch.transpose(x, 1, 2)
            else:
                x = layer(x)
        return x

    def forward(self, x1, p1, x2, p2):
        x1 = self.apply_module_with_bn(x1, self.mlp_in)
        # dist, idx = three_nn(p2, p1)
        # p2.size() = bz x N2 x 3
        dist = p2[:, :, None, :] - p1[:, None, :, :]
        dist = torch.norm(dist, dim=-1, p=2, keepdim=False)
        # dist.size() = bz x N2 x N1
        # print(dist.size())
        dist, idx = dist.topk(3, dim=-1, largest=False)
        # bz x N2 x 3
        # print(dist.size(), idx.size())
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        # weight.size() = bz x N2 x 3; idx.size() = bz x N2 x 3
        three_nearest_features = batched_index_select(x1, idx, dim=1) # 1 is the idx dimension
        # three_features = bz x N2 x 3 x fea_dim; x1 = bz x N1 x fea_dim;
        # print(three_nearest_features.size(), weight.size())
        interpolated_feats = torch.sum(three_nearest_features * weight[:, :, :, None], dim=2, keepdim=False)
        # interpolated_feats = bz x N2 x fea_dim
        # interpolated_feats = three_interpolate(
        #     x1.transpose(1, 2).contiguous(), idx.contiguous(), weight.contiguous()
        # )
        x2 = self.apply_module_with_bn(x2, self.mlp_out)
        # x2 = self.mlp_out(x2.transpose(1, 2).contiguous())
        # print(interpolated_feats.size(), x2.size())
        y = interpolated_feats + x2 # interpolated_feats.size() and x2.size()
        # y = interpolated_feats.transpose(1, 2) + x2 #.transpose(1, 2)
        return y.contiguous(), p2



