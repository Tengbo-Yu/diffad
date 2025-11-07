import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed
from model.dit_modules.layers import TransformerDecoderLayer
from model.dit_modules.layers import build_mlp
from model.dit_modules.position_embedding import get_2d_sincos_pos_embed
from util.transformer_util import _get_clones
from util.projection_util import img2lidar


class PostProcessNet(nn.Module):

    def __init__(self, config, input_size, in_channels, patch_size, hidden_size, pc_range):
        super().__init__()
        dropout = config['dropout']
        n_head = config['nhead']
        d_ffn = config['dim_feedforward']
        activation = config['activation']
        num_decoder_layers = config['num_decoder_layers']
        decoder_layer = TransformerDecoderLayer(hidden_size, d_ffn, dropout, activation, n_head)
        self.layers = _get_clones(decoder_layer, num_decoder_layers)

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        num_planning_frames, point_size = config['traj_size']
        num_queries = num_planning_frames + 1  # current+future
        self.query = nn.Embedding(num_queries, hidden_size * 2)
        num_fc = config['num_fc']
        self.final_layer = build_mlp(num_fc, hidden_size, hidden_size, point_size)

        self.pc_range = pc_range

        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

    def forward(self, x):
        x = self.x_embedder(x)  # [bs, T, D]
        x = x + self.pos_embed
        bs, _, _ = x.shape

        query = self.query.weight.unsqueeze(0).expand(bs, -1, -1)  # bs, num_queries,d
        query, query_pos = query.chunk(chunks=2, dim=-1)

        output = query
        src = x
        for lid, layer in enumerate(self.layers):
            output = layer(output, query_pos, src)

        output = self.final_layer(output)  # [bs, L, C]
        output = output.unsqueeze(1)  # [bs, 1, T, 2]
        output[..., :2] = output[..., :2].sigmoid()
        output = img2lidar(output, self.pc_range)
        return output

    def loss_func(self, x, plan_traj, plan_traj_mask):
        pred_trajs = self.forward(x)
        loss_tensor = F.mse_loss(pred_trajs, plan_traj, reduction='none')
        loss_mask = plan_traj_mask == 1
        loss = loss_tensor[loss_mask].mean()
        return loss
