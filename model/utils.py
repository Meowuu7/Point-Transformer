import torch
from torch_cluster import fps


def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sampling(pos: torch.FloatTensor, n_sampling: int):
    bz, N = pos.size(0), pos.size(1)
    device = pos.device
    sampling_ratio = float(n_sampling / N)
    pos_float = pos.float()
    # [1, bz] x [N, 1]
    # print("fps rate = ", sampling_ratio)
    # batch = torch.arange(bz, dtype=torch.long, device=device).view(1, bz)
    # mult_one = torch.ones((N, ), dtype=torch.long, device=device).view(N, 1)
    # move to cuda
    batch = torch.arange(bz, dtype=torch.long).view(bz, 1).to(device)
    mult_one = torch.ones((N,), dtype=torch.long).view(1, N).to(device)

    batch = batch * mult_one
    batch = batch.view(-1)
    pos_float = pos_float.view(-1, 3) # (bz x N, 3)
    sampling_ratio = torch.tensor([sampling_ratio for _ in range(bz)], dtype=torch.float).to(device)
    # batch = torch.zeros((N, ), dtype=torch.long, device=device)
    sampled_idx = fps(pos_float, batch, ratio=sampling_ratio, random_start=False)
    # shape of sampled_idx?
    return sampled_idx


def get_knn_idx(pos: torch.FloatTensor, k: int, sampled_idx: torch.LongTensor=None, n_sampling: int=None):
    bz, N = pos.size(0), pos.size(1)
    if sampled_idx is not None:
        assert n_sampling is not None
        pos_exp = pos.view(bz * N, -1)
        pos_sampled = pos_exp[sampled_idx, :]
        rel_pos_sampled = pos_sampled.view(bz, n_sampling, 1, -1) - pos.view(bz, 1, N, -1)
        rel_dist = rel_pos_sampled.norm(dim=-1)
        nearest_k_dist, nearest_k_idx = rel_dist.topk(k, dim=-1, largest=False)
        return nearest_k_idx
    # N = pos.size(0)

    rel_pos = pos.view(bz, N, 1, -1) - pos.view(bz, 1, N, -1)
    rel_dist = rel_pos.norm(dim=-1)
    nearest_k_dist, nearest_k_idx = rel_dist.topk(k, dim=-1, largest=False)
    # [bz, N, k]
    # return the nearest k idx
    return nearest_k_idx

