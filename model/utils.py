import torch
from torch_cluster import fps
import warnings
from torch.autograd import Function

# try:
#     import model._ext as _ext
# except ImportError:
#     from torch.utils.cpp_extension import load
#     import glob
#     import os.path as osp
#     import os
#
#     warnings.warn("Unable to load point_transformer_ops cpp extension. JIT Compiling.")
#
#     _ext_src_root = osp.join(osp.dirname(__file__), "_ext-src")
#     _ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
#         osp.join(_ext_src_root, "src", "*.cu")
#     )
#     _ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))
#
#     os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
#     _ext = load(
#         "_ext",
#         sources=_ext_sources,
#         extra_include_paths=[osp.join(_ext_src_root, "include")],
#         extra_cflags=["-O3"],
#         extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
#         with_cuda=True,
#     )


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


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights
        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        ctx.save_for_backward(idx, weight, features)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs
        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features
        None
        None
        """
        idx, weight, features = ctx.saved_tensors
        m = features.size(2)

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, torch.zeros_like(idx), torch.zeros_like(weight)


three_interpolate = ThreeInterpolate.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features
        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)
        dist = torch.sqrt(dist2)

        ctx.mark_non_differentiable(dist, idx)

        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()


three_nn = ThreeNN.apply