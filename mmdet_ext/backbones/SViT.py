import os
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from mmdet.models.builder import BACKBONES

from .modules import (
    MySequential,
    LayerNorm,
    QuickGELU,
    inflate_weight,
    resize_abs_pos,
    window_partition,
    window_reverse,
    FPNAdapter,
    ResBottleneckBlock,
    DropPath)


class Attention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_type='space',
                 num_frames=1):
        super(Attention, self).__init__()
        assert attn_type in ['space', 'stream']
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if attn_type in ['stream']:
            self.in_proj_weight_t = nn.Parameter(torch.empty((embed_dim, embed_dim)))
            self.in_proj_bias_t = nn.Parameter(torch.empty(embed_dim))
            self.out_proj_t = nn.Linear(embed_dim, embed_dim)
            self.alpha = nn.Parameter(1e-4 * torch.ones((embed_dim)), requires_grad=True)
        self.history_k = []
        self.history_v = []
        self.num_frames = num_frames
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)
        if self.attn_type in ['stream']:
            xavier_uniform_(self.in_proj_weight_t)
            constant_(self.in_proj_bias_t, 0.)
            constant_(self.out_proj_t.bias, 0.)

    def clear_history(self):
        if self.attn_type != 'stream':
            return
        for k in self.history_k:
            del k
        for v in self.history_v:
            del v
        self.history_k.clear()
        self.history_v.clear()
        # torch.cuda.empty_cache()

    def attention(self, q, k, v, mask=None):
        attn = (q @ k.transpose(-2, -1))
        if mask is not None:
            attn = attn.masked_fill(mask, -1e3)
        attn = attn.softmax(-1)
        o = (attn @ v)
        return o

    def _forward_in_frame(self, x):

        x = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = rearrange(x, 'b n (three num_heads head_c) -> three b num_heads n head_c',
                        three=3, num_heads=self.num_heads)
        q, k, v = qkv.unbind(0)
        q = q * self.scale

        x = self.attention(q, k, v)

        x = rearrange(x, 'b num_heads n head_c -> b n (num_heads head_c)')
        x = self.out_proj(x)

        if self.attn_type == 'stream':
            if len(self.history_k) == self.num_frames:
                if self.training:
                    self.clear_history()
                else:
                    # self.history_k.pop(0)
                    # self.history_v.pop(0)

                    k_oldest = self.history_k.pop(0)
                    v_oldest = self.history_v.pop(0)
                    if len(self.history_k) > 0:
                        self.history_k[0] = (self.history_k[0] + k_oldest) / 2
                        self.history_v[0] = (self.history_v[0] + v_oldest) / 2
            self.history_k.append(k.detach())  # [(B hh ww), n_heads, (win_h win_w), head_c]
            self.history_v.append(v.detach())

        return x

    def _forward_cross_frame(self, x, size):
        H, W = size

        q_t = F.linear(x, self.in_proj_weight_t, self.in_proj_bias_t)
        q_t = q_t * self.scale
        q_th = rearrange(q_t, 'b (h w) (num_heads head_c) -> (b w) num_heads h head_c', num_heads=self.num_heads, h=H)
        q_tw = rearrange(q_t, 'b (h w) (num_heads head_c) -> (b h) num_heads w head_c', num_heads=self.num_heads, w=W)

        k_t = torch.stack(self.history_k, dim=1)
        v_t = torch.stack(self.history_v, dim=1)

        k_th = rearrange(k_t, 'b t num_heads (h w) head_c -> (b w) num_heads (t h) head_c', h=H)
        k_tw = rearrange(k_t, 'b t num_heads (h w) head_c -> (b h) num_heads (t w) head_c', w=W)
        v_th = rearrange(v_t, 'b t num_heads (h w) head_c -> (b w) num_heads (t h) head_c', h=H)
        v_tw = rearrange(v_t, 'b t num_heads (h w) head_c -> (b h) num_heads (t w) head_c', w=W)

        o_th = self.attention(q_th, k_th, v_th)
        o_tw = self.attention(q_tw, k_tw, v_tw)

        o_th = rearrange(o_th, '(b w) num_heads h head_c -> b (h w) (num_heads head_c)', w=W)
        o_tw = rearrange(o_tw, '(b h) num_heads w head_c -> b (h w) (num_heads head_c)', h=H)

        x_t = self.out_proj_t(o_th + o_tw)

        return x_t

    def forward(self, x, window_size=0, size=None):
        """[B N C]"""

        if window_size > 0:
            x, pad_size = window_partition(x, size, window_size)

        if self.attn_type == 'space':
            x = self._forward_in_frame(x)
        elif self.attn_type == 'stream':
            x_in = self._forward_in_frame(x)
            x_cross = self._forward_cross_frame(x, (window_size, window_size))
            x = x_in + self.alpha * x_cross
        if window_size > 0:
            x = window_reverse(x, size, window_size, pad_size)

        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_type: str = 'space', num_frames=1, window_size=0, use_residual_block=False,
                 drop_path_rate=0.):
        super().__init__()

        self.attn = Attention(d_model, n_head, attn_type, num_frames)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if self.use_residual_block:
            self.residual_block = ResBottleneckBlock(
                in_channels=d_model,
                out_channels=d_model,
                bottleneck_channels=d_model // 2,
            )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, size):

        x = x + self.drop_path(self.attn(self.ln_1(x), self.window_size, size))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        if self.use_residual_block:
            x = self.residual_block(x, size)
        return x, size

    def clear_history(self):
        self.attn.clear_history()


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_type='space', num_frames=1, window_size=0,
                 window_block_index=(), residual_block_index=(),
                 enable_checkpoint=False,
                 stage_segment=None,
                 drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]
        self.resblocks = MySequential(*[ResidualAttentionBlock(width, heads, attn_type, num_frames,
                                                               window_size if i in window_block_index else 0,
                                                               use_residual_block=i in residual_block_index,
                                                               drop_path_rate=dpr[i]) for i in range(layers)])
        self.enable_checkpoint = enable_checkpoint
        self.stage_segment = stage_segment

    def forward(self, x: torch.Tensor, size):
        if self.stage_segment is None:
            return self.resblocks(x, size)[0]
        else:
            outs = []
            for i, m in enumerate(self.resblocks):
                if self.enable_checkpoint:
                    x, size = checkpoint.checkpoint(m, x, size)
                else:
                    x, size = m(x, size)
                if i in self.stage_segment:
                    outs.append(x)
            return outs

    def clear_history(self):
        for layer in self.resblocks:
            layer.clear_history()

@BACKBONES.register_module()
class SViT(nn.Module):
    def __init__(self,
                 width=768,
                 patch_size=(16, 16),
                 layers=12,
                 heads=12,
                 window_size=0,
                 input_resolution=224,
                 pretrain=None,
                 enable_checkpoint=False,
                 stage_segment=(2, 5, 8, 11),
                 out_indices=(1, 2, 3),
                 fpn_scale=(2.0, 1.0, 0.5),
                 window_block_index=(),
                 residual_block_index=(),
                 drop_path_rate=0.,
                 attn_type='space',
                 num_frames=1,
                 ):
        # input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int
        super().__init__()

        # build model
        assert len(stage_segment) == 4
        self.input_resolution = input_resolution
        self.enable_checkpoint = enable_checkpoint
        self.out_indices = out_indices

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size,
                               bias=False)
        scale = width ** -0.5

        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size[0]) ** 2, width))

        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, attn_type, num_frames,
                                       window_size, window_block_index, residual_block_index,
                                       enable_checkpoint, stage_segment,
                                       drop_path_rate)
        self.ln_post = LayerNorm(width)

        self.ln_post_out0 = LayerNorm(width) if 0 in out_indices else None
        self.ln_post_out1 = LayerNorm(width) if 1 in out_indices else None
        self.ln_post_out2 = LayerNorm(width) if 2 in out_indices else None
        self.fpn_adaptor = FPNAdapter(width, fpn_scale)

        self._initialize_weights()
        self._load_pretrain(pretrain)

    def _initialize_weights(self):
        nn.init.normal_(self.positional_embedding, std=0.02)

    def _load_pretrain(self, pretrain):
        if pretrain is None or not os.path.exists(pretrain):
            return
        print(f'Loading network weights from {pretrain}')
        model_state_dict_3d = self.state_dict()

        try:
            clip_model = torch.jit.load(pretrain, map_location='cpu')
            clip_model = clip_model.visual
            model_state_dict_2d = clip_model.state_dict()
        except:
            checkpoint = torch.load(pretrain, map_location='cpu')
            model_state_dict_2d = checkpoint['model']

        # remove pos embed for class token
        model_state_dict_2d['positional_embedding'] = model_state_dict_2d['positional_embedding'][1:]

        inflated_model_dict = inflate_weight(model_state_dict_2d, model_state_dict_3d)
        msg = self.load_state_dict(inflated_model_dict, strict=False)
        print(msg)
        print('Pretrained network weights loaded.')

    def forward_features(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.conv1(x)
        size = list(x.shape[2:])
        x = rearrange(x, 'b c h w -> b (h w) c', b=B)

        pos_embed = resize_abs_pos(self.positional_embedding.unsqueeze(0), size)

        x = x + pos_embed

        x = self.ln_pre(x)

        xs = self.transformer(x, size)

        norms = [getattr(self, f'ln_post_out{i}') for i in range(3)] + [self.ln_post]
        outs = [rearrange(norms[i](xs[i]), 'b (h w) c -> b c h w', h=size[0]) if i in self.out_indices else None
                for i in range(4)]
        outs = [outs[i] for i in self.out_indices]
        return outs

    def forward(self, x):
        outs = self.forward_features(x)
        outs = self.fpn_adaptor(outs)
        return outs

    def clear_history(self):
        self.transformer.clear_history()


