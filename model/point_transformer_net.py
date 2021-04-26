import torch
import torch.nn as nn
from .point_transformer_layer import PointTransformerLayer
from .point_convolution_layers import TransitionDown


class PointTransformerObjClassificationNet(nn.Module):
    def __init__(self, n_layers: int, feat_dims: list, n_samples: list, n_class: int, in_feat_dim: int,
                 dp_ratio: float=0.5, attn_mult: int=2, args=None):
        super(PointTransformerObjClassificationNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.feat_map_blocks = nn.ModuleList()
        self.pt_transformer_blocks = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                # TODO: the MLP architecture?
                blk = nn.Sequential(
                    nn.Linear(in_features=in_feat_dim, out_features=feat_dims[0], bias=True),
                    nn.BatchNorm1d(num_features=feat_dims[0], eps=1e-5, momentum=0.1),
                    nn.ReLU(),
                )
            else:
                # feat_dim: int, out_feat_dim: int, k: int=16
                blk = TransitionDown(feat_dim=feat_dims[i-1],
                                     out_feat_dim=feat_dims[i],
                                     k=16)
            # TODO: parameter for pos_mlp_hidden_dim? parameter for attn_mlp_hidden_mult? for num_neighbours?
            # pmhd = 64 if feat_dims[i] < 128 else (feat_dims[i] // 2)
            # ptfl_blk = PointTransformerLayer(dim=feat_dims[i],
            #                                  pos_mlp_hidden_dim=pmhd,
            #                                  attn_mlp_hidden_mult=4,
            #                                  num_neighbors=16)

            # pmhd = 64 if feat_dims[i] < 256 else (feat_dims[i] // 4)
            # TODO: dimensions of pos_mlp_hidden_dim and attn_mlp_hidden_mult?
            ptfl_blk = PointTransformerLayer(in_dim=feat_dims[i],
                                             dim=512,
                                             pos_mlp_hidden_dim=512,
                                             attn_mlp_hidden_mult=attn_mult,
                                             num_neighbors=16,
                                             dp_ratio=dp_ratio,
                                             use_abs_pos=args.use_abs_pos,
                                             with_normal=args.with_normal)

            self.feat_map_blocks.append(blk)
            self.pt_transformer_blocks.append(ptfl_blk)
        # self.cls_layer = nn.Linear(feat_dims[-1], n_class, bias=True)
        self.cls_layer = nn.Sequential(
            nn.Linear(feat_dims[-1], 256, bias=True),
            nn.BatchNorm1d(num_features=256, eps=1e-5, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(p=dp_ratio),
            nn.Linear(256, 64, bias=True),
            nn.BatchNorm1d(num_features=64, eps=1e-5, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(p=dp_ratio),
            nn.Linear(64, n_class)
        )
        # self.cls_layer = nn.Sequential(
        #     nn.Linear(feat_dims[-1], 256, bias=True),
        #     # nn.BatchNorm1d(num_features=256, eps=1e-5, momentum=0.1),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(256, n_class)
        # )
        self.n_samples = n_samples
        # nn.init.xavier_uniform_(self.cls_layer.weight)
        # nn.init.zeros_(self.cls_layer.bias)
        # self.init_weight(self.feat_map_blocks) # initialize weight matrices in nn.Linear in feat_map_blocks
        # self.init_weight(self.cls_layer)
        # TODO: the influence of weight initialization on model's performance and whether easy to train

    def init_weight(self, blocks):
        for module in blocks:
            if isinstance(module, nn.Sequential):
                for subm in module:
                    if isinstance(subm, nn.Linear):
                        nn.init.xavier_uniform_(subm.weight)
                        nn.init.zeros_(subm.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.FloatTensor, pos: torch.FloatTensor):
        for i, (feat_map_blk, pt_trans_blk) in enumerate(zip(self.feat_map_blocks, self.pt_transformer_blocks)):
            if isinstance(feat_map_blk, nn.Sequential):
                for seq_blk in feat_map_blk:
                    if not isinstance(seq_blk, nn.BatchNorm1d):
                        x = seq_blk(x)
                    else:
                        x = torch.transpose(x, 1, 2) # bz x N x feat_dims[0] -> bz x feat_dims[0] x N
                        x = seq_blk(x)
                        x = torch.transpose(x, 1, 2) # bz x feat_dims[0] x N -> bz x N x feat_dims[0]
                # x = feat_map_blk(x)
                x = pt_trans_blk(x, pos) # point attention transformer
            else:
                x, pos = feat_map_blk(x, pos, self.n_samples[i - 1]) # i-th samples
                # print("in %d-th forward block, x.size() = " % i, x.size(), "; pos.size() = ", pos.size())
                x = pt_trans_blk(x, pos)
            # print("in %d-th forward block, x.size() = " % i, x.size(), "; pos.size() = ", pos.size())
        # x.size() = bz x n_samples_last x feat_dim_last
        x = x.mean(dim=1) # mean pooling
        logits = self.cls_layer(x)
        # logits.size() = bz x n_classes
        return logits
