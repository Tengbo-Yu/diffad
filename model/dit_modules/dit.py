import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpointing
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed
from timm.models.vision_transformer import Mlp
from model.dit_modules.layers import modulate
from model.dit_modules.layers import Attention
from model.dit_modules.layers import FinalLayer
from model.dit_modules.position_embedding import TimestepEmbedder, Embedder
from model.dit_modules.position_embedding import get_2d_sincos_pos_embed
from model.builder import LDM_MODEL


class DiTBlock(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio=4.0,
            **block_kwargs
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiT(nn.Module):
    def __init__(self, depth, hidden_size, patch_size, num_heads, input_size,
                 in_channels, bevfeat_dim, command_dim=6, mlp_ratio=4.0, learn_sigma=True,
                 dtype=torch.float32, use_gradient_checkpointing=False):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.dtype = dtype
        assert (input_size[0] % patch_size == 0) and (input_size[1] % patch_size == 0)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.task_embedder = Embedder(4, hidden_size)
        self.egostatus_embedder = Embedder(9, hidden_size)
        self.navi_embedder = nn.Embedding(command_dim, hidden_size)
        self.num_patches = self.x_embedder.num_patches
        self.grid_size = self.x_embedder.grid_size

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        # BevNet
        self.bev_embedder = PatchEmbed(input_size, patch_size, bevfeat_dim, hidden_size, bias=True)

        # load pretrained weight
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def forward(self, x, timestep, bevfeature, prop_prev_x_start, task_label, ego_status, command):
        """
        Forward pass of DiT.
        x: (N=bs, C, H, W)
        t: (N=bs,) tensor of diffusion timesteps
        command:(N, C)
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)

        # embedding
        x = self.x_embedder(x)  # [bs, L, D]
        x = x + self.pos_embed

        # timestemp
        t = self.t_embedder(timestep, dtype=x.dtype).unsqueeze(1).repeat(1, self.num_patches, 1)  # [bs*T, L, D]

        # BevNet
        bevfeature = F.interpolate(bevfeature, size=self.input_size, mode='bilinear',
                                   align_corners=False)  # bs*ts,c,h,w
        bevfeature = self.bev_embedder(bevfeature)  # [bs, L, D]

        # last action
        if prop_prev_x_start is not None:
            bevfeature = bevfeature + prop_prev_x_start

        # task_label
        # torch.Size([1, 4])
        task_embed = self.task_embedder(task_label, dtype=x.dtype).unsqueeze(1).repeat(1, self.num_patches, 1)

        # ego_status: torch.Size([1, 9])
        egostatus_embed = self.egostatus_embedder(ego_status, dtype=x.dtype).unsqueeze(1).repeat(1, self.num_patches, 1)

        # driving command
        navi_index = torch.where(command == 1)[1]  # N
        navi_embed = self.navi_embedder.weight[navi_index].unsqueeze(1).repeat(1, self.num_patches, 1)

        cond = t + bevfeature + task_embed + egostatus_embed + navi_embed  # bs,L+T,D
        # torch.Size([1, 512, 1024])

        # # # blocks
        for i, block in enumerate(self.blocks):
            if self.training and self.use_gradient_checkpointing:
                x = gradient_checkpointing(block, x, cond, use_reentrant=False)
            else:
                x = block(x, cond)  # (bs, S, D)

        # final process
        x = self.final_layer(x, cond)  # [bs, L, C]

        x = self.unpatchify(x, self.out_channels,
                            self.x_embedder.patch_size,
                            self.x_embedder.grid_size)  # [bs, C, H, W]

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        w = self.bev_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.bev_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.task_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.task_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.egostatus_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.egostatus_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.navi_embedder.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, out_channels, patch_size, grid_size):
        """
        x: (N, S, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = out_channels
        p, q = patch_size
        h, w = grid_size
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, q, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * q))
        return imgs


@LDM_MODEL.register_function('DiT-XL/2')
def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


@LDM_MODEL.register_function('DiT-XL/4')
def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


@LDM_MODEL.register_function('DiT-XL/8')
def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


@LDM_MODEL.register_function('DiT-L/2')
def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


@LDM_MODEL.register_function('DiT-L/4')
def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


@LDM_MODEL.register_function('DiT-L/8')
def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


@LDM_MODEL.register_function('DiT-B/2')
def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


@LDM_MODEL.register_function('DiT-B/4')
def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


@LDM_MODEL.register_function('DiT-B/8')
def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


@LDM_MODEL.register_function('DiT-S/2')
def DiT_S_2(**kwargs):
    return DiT(depth=1, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


@LDM_MODEL.register_function('DiT-S/4')
def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


@LDM_MODEL.register_function('DiT-S/8')
def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)
