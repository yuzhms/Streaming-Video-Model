import math

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from einops import rearrange
import fvcore.nn.weight_init as weight_init


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def sequential_checkpoint(functions, segments, *input):
    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    def run_function(start, end, functions):
        def forward(*input):
            for j in range(start, end + 1):
                input = functions[j](*input)
            return input
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        input = checkpoint.checkpoint(run_function(start, end, functions), *input)
    input = run_function(end + 1, len(functions) - 1, functions)(*input)
    return input


def inflate_weight(state_dict_2d, state_dict_3d):
    # copy from slowfast.checkpoint
    from collections import OrderedDict
    state_dict_inflated = OrderedDict()
    for k, v2d in state_dict_2d.items():
        if k not in state_dict_3d.keys():
            print(f"Unknown key {k} from 2d dict")
            continue
        v3d = state_dict_3d[k]
        # Inflate the weight of 2D conv to 3D conv.
        if len(v2d.shape) == 4 and len(v3d.shape) == 5:
            print(
                "Inflate {}: {} -> {}: {}".format(k, v2d.shape, k, v3d.shape)
            )
            # Dimension need to be match.
            assert v2d.shape[-2:] == v3d.shape[-2:]
            assert v2d.shape[:2] == v3d.shape[:2]
            v3d = (
                    v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
            )
        elif v2d.shape == v3d.shape:
            v3d = v2d
        else:
            print(
                "Unexpected {}: {} -|> {}: {}".format(
                    k, v2d.shape, k, v3d.shape
                )
            )
        state_dict_inflated[k] = v3d.clone()
    return state_dict_inflated


def resize_abs_pos(abs_pos, size_new):
    H, W = size_new
    token_num = abs_pos.shape[1]
    size_old = int(math.sqrt(token_num))
    assert size_old ** 2 == token_num

    if token_num != H * W:
        new_abs_pos = F.interpolate(
            rearrange(abs_pos, '1 (h w) c -> 1 c h w', h=size_old),
            size=size_new,
            mode='bicubic',
            align_corners=False
        )
        return rearrange(new_abs_pos, '1 c h w -> 1 (h w) c')
    else:
        return abs_pos


def window_partition(x, size, window_size):
    H, W = size

    x = rearrange(x, 'b (h w) c -> b h w c', h=H)

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h + pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    windows = rearrange(x, 'b (hh win_h) (ww win_w) c -> (b hh ww) (win_h win_w) c',
                        win_h=window_size, win_w=window_size)
    return windows, (Hp, Wp)


def window_reverse(windows, size, window_size, pad_size):
    Hp, Wp = pad_size
    H, W = size
    x = rearrange(windows, '(b hh ww) (win_h win_w) c -> b (hh win_h) (ww win_w) c',
                  hh=Hp//window_size, ww=Wp//window_size, win_h=window_size)

    if Hp * Wp > H * W:
        x = x[:, :H, :W, :].contiguous()
    x = rearrange(x, 'b h w c -> b (h w) c')
    return x


class FPNAdapter(nn.Module):
    def __init__(self,
                 input_channels=768,
                 scale_factor=(2.0, 1.0, 0.5)):

        super(FPNAdapter, self).__init__()

        for idx, scale in enumerate(scale_factor):
            out_dim = input_channels
            out_channel = int(input_channels/scale)
            if scale == 2.0:
                layers = [nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=2, stride=2)]
                out_dim = input_channels // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend([
                nn.Conv2d(out_dim, out_channel, kernel_size=1, bias=False),
                LayerNormChannelFirst(out_channel),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
                LayerNormChannelFirst(out_channel)
            ])
            layers = nn.Sequential(*layers)
            self.add_module(f'fpn_adaptor_{idx}', layers)

    def forward(self, xs):
        xs = [getattr(self, f'fpn_adaptor_{idx}')(x) for idx, x in enumerate(xs)]
        return xs


class LayerNormChannelFirst(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ResBottleneckBlock(nn.Module):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = LayerNormChannelFirst(bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = LayerNormChannelFirst(bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = LayerNormChannelFirst(out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x, size):
        H, W = size
        x = rearrange(x, 'b (h w) c -> b c h w', h=H)
        out = x
        for layer in [self.conv1, self.norm1, self.act1, self.conv2, self.norm2, self.act2, self.conv3, self.norm3]:
            out = layer(out)

        out = x + out
        out = rearrange(out, 'b c h w -> b (h w) c')
        return out


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
