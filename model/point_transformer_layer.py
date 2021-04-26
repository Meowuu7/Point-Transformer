import torch
from torch import nn, einsum
from einops import repeat
from .utils import batched_index_select, index_points
import numpy as np
# helpers

def exists(val):
    return val is not None

def max_value(t):
    return torch.finfo(t.dtype).max

# def batched_index_select(values, indices, dim = 1):
#     value_dims = values.shape[(dim + 1):]
#     values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
#     indices = indices[(..., *((None,) * len(value_dims)))]
#     indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
#     value_expand_len = len(indices_shape) - (dim + 1)
#     values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]
#
#     value_expand_shape = [-1] * len(values.shape)
#     expand_slice = slice(dim, (dim + value_expand_len))
#     value_expand_shape[expand_slice] = indices.shape[expand_slice]
#     values = values.expand(*value_expand_shape)
#
#     dim += value_expand_len
#     return values.gather(dim, indices)

# classes

class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        in_dim,
        dim,
        pos_mlp_hidden_dim = 64,
        attn_mlp_hidden_mult = 4,
        num_neighbors = None,
        dp_ratio=0.5,
        use_abs_pos=False,
        with_normal=False
    ):
        super().__init__()
        self.num_neighbors = num_neighbors

        # add linear transition block
        # self.linear_in = nn.Linear(dim * 2, dim, bias=False)
        # self.linear_out = nn.Linear(dim, dim * 2, bias=False)

        self.linear_in = nn.Linear(in_dim, dim, bias=False)
        self.linear_out = nn.Linear(dim, in_dim, bias=False)

        # nn.init.xavier_uniform_(self.linear_in.weight)
        # nn.init.xavier_uniform_(self.linear_out.weight)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False) # input dims to the transformer network

        pos_input_dim = 3 if (not use_abs_pos) else 6 if (use_abs_pos and (not with_normal)) else 9

        self.use_abs_pos = use_abs_pos
        self.with_normal = with_normal

        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_input_dim, pos_mlp_hidden_dim),
            nn.BatchNorm2d(pos_mlp_hidden_dim, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(p=dp_ratio),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.BatchNorm2d(dim * attn_mlp_hidden_mult, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(p=dp_ratio),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )
        # such two initializations should be kept
        self.init_weight(self.pos_mlp)
        self.init_weight(self.attn_mlp)

    def init_weight(self, module):
        for md in module:
            if isinstance(md, nn.Linear):
                nn.init.xavier_uniform_(md.weight)
                nn.init.zeros_(md.bias)

    def apply_module_with_bn(self, rel_pos_emb, module):
        # bz x N x k x C
        for layer in module:
            if not isinstance(layer, nn.BatchNorm2d):
                rel_pos_emb = layer(rel_pos_emb)
            else:
                rel_pos_emb = torch.transpose(rel_pos_emb, 1, 3)
                rel_pos_emb = layer(rel_pos_emb)
                rel_pos_emb = torch.transpose(rel_pos_emb, 1, 3)
        return rel_pos_emb

    def forward(self, ori_x, pos, mask = None):
        # if isinstance(pos, torch.LongTensor):
        #     pos = pos.float()
        x = self.linear_in(ori_x)
        # x = ori_x
        # pos = pos.float()
        n, num_neighbors = x.shape[1], self.num_neighbors

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # calculate relative positional embeddings
        # pos.size() = (bz, N, 3)
        rel_pos = pos[:, :, None, :3] - pos[:, None, :, :3]
        rel_dist = rel_pos.norm(dim=-1)
        # rel_dist.size() = bz x N x N
        # print(num_neighbors, rel_dist.size())
        num_neighbors = min(num_neighbors, n)
        rel_dist_topk, topk_indices = rel_dist.topk(num_neighbors, largest=False, dim=-1)
        qk_rel_topk = q[:, :, None, :] - index_points(k, topk_indices)

        top_k_pos = index_points(pos[:, :, :], topk_indices)
        rel_pos_topk = pos[:, :, None, :3] - top_k_pos[:, :, :, :3] # index_points(pos[:, :, :3], topk_indices)
        if self.use_abs_pos and (not self.with_normal):
            rel_pos_topk = torch.cat([rel_pos_topk, top_k_pos[:, :, :, :3]], dim=-1)
        elif self.use_abs_pos and self.with_normal:
            rel_pos_topk = torch.cat([rel_pos_topk, top_k_pos], dim=-1)
        # rel_pos_topk_emb = self.pos_mlp(rel_pos_topk)

        rel_pos_topk_emb = self.apply_module_with_bn(rel_pos_topk, self.pos_mlp)
        v = index_points(v, topk_indices)
        # attn_emb = self.attn_mlp(qk_rel_topk + rel_pos_topk_emb)

        attn_emb = self.apply_module_with_bn(qk_rel_topk + rel_pos_topk_emb, self.attn_mlp)
        attn_emb = torch.softmax(attn_emb / np.sqrt(k.size(-1)), dim=-2)
        res = torch.einsum('bmnf,bmnf->bmf', attn_emb, v + rel_pos_topk_emb)
        res = self.linear_out(res) + ori_x
        return res

        # rel_pos_emb = self.pos_mlp(rel_pos)
        # # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than
        # dot product
        # # print("q k size for qk_rel: ", q.size(), k.size())
        # qk_rel = q[:, :, None, :] - k[:, None, :, :]
        #
        # # prepare mask
        # if exists(mask):
        #     mask = mask[:, :, None] * mask[:, None, :]
        #
        # # expand values
        # v = repeat(v, 'b j d -> b i j d', i = n)
        #
        # # determine k nearest neighbors for each point, if specified
        # if exists(num_neighbors) and num_neighbors < n:
        #     rel_dist = rel_pos.norm(dim = -1)
        #
        #     if exists(mask):
        #         mask_value = max_value(rel_dist)
        #         rel_dist.masked_fill_(~mask, mask_value)
        #
        #     dist, indices = rel_dist.topk(num_neighbors, largest = False)
        #
        #     v = batched_index_select(v, indices, dim = 2)
        #     qk_rel = batched_index_select(qk_rel, indices, dim = 2)
        #     rel_pos_emb = batched_index_select(rel_pos_emb, indices, dim = 2)
        #     mask = batched_index_select(mask, indices, dim = 2) if exists(mask) else None
        #
        # # add relative positional embeddings to value
        # v = v + rel_pos_emb
        #
        # # use attention mlp, making sure to add relative positional embedding first
        # sim = self.attn_mlp(qk_rel + rel_pos_emb)
        #
        # # masking
        # if exists(mask):
        #     mask_value = -max_value(sim)
        #     sim.masked_fill_(~mask[..., None], mask_value)
        #
        # # attention
        # attn = sim.softmax(dim = -2)
        #
        # # aggregate
        # agg = einsum('b i j d, b i j d -> b i d', attn, v)
        #
        # agg = self.linear_out(agg)
        #
        # return agg + ori_x
