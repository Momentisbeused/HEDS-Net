import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from typing import Callable
from timm.models.layers import DropPath


def _resolve_selective_scan_fn():
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _fn

        return _fn
    except Exception:
        pass
    try:
        from selective_scan import selective_scan_fn as _fn

        return _fn
    except Exception:
        return None


selective_scan_fn = _resolve_selective_scan_fn()


class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, bias=False):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            padding=padding,
            dilation=dilation,
            groups=in_channels, 
            bias=False
        )
        num_groups = 8
        while in_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.act = nn.GELU()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pointwise(x)
        return x


class MultiScaleLocalBranch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.conv_d1 = DepthwiseSeparableConv(dim, dim, kernel_size=3, dilation=1)
        self.conv_d2 = DepthwiseSeparableConv(dim, dim, kernel_size=3, dilation=2)
        self.conv_d4 = DepthwiseSeparableConv(dim, dim, kernel_size=3, dilation=4)

        self.se = SEModule(dim * 3, reduction=4)

        self.fusion = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False)
        nn.init.zeros_(self.fusion.weight)

        num_groups = 8
        while dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim)
    
    def forward(self, x):
        B, H, W, C = x.shape

        x_c = x.permute(0, 3, 1, 2).contiguous()

        feat_d1 = self.conv_d1(x_c)
        feat_d2 = self.conv_d2(x_c)
        feat_d4 = self.conv_d4(x_c)

        multi_scale = torch.cat([feat_d1, feat_d2, feat_d4], dim=1)

        multi_scale = self.se(multi_scale)

        fused = self.fusion(multi_scale)
        fused = self.norm(fused)

        out = fused.permute(0, 2, 3, 1).contiguous()

        return out


class VSSBranch(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False,
                 device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner,
                                bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs)
        self.act = nn.SiLU()

        
        x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(4)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj], dim=0))

        dt_projs = []
        for _ in range(4):
            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            dt_init_std = self.dt_rank**-0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_proj.bias._no_reinit = True
            dt_projs.append(dt_proj)
        
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        
        self.A_logs = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).reshape(1, -1).repeat(self.d_inner * 4, 1).contiguous()))
        self.Ds = nn.Parameter(torch.ones(self.d_inner * 4))
        self.A_logs._no_weight_decay = True
        self.Ds._no_weight_decay = True

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        scan_fn = selective_scan_fn
        if scan_fn is None:
            raise RuntimeError(
                "HVST/VSS requires selective_scan_fn: install `mamba_ssm` (same as VM-UNet/VMamba) "
                "or add the `selective_scan` fallback used by SS2D.forward_corev1 in vmamba.py."
            )
        try:
            out_y = scan_fn(
                xs,
                dts,
                As,
                Bs,
                Cs,
                Ds,
                z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(B, K, -1, L)
        except TypeError:
            out_y = scan_fn(
                xs,
                dts,
                As,
                Bs,
                Cs,
                Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SmoothProgressiveFusion(nn.Module):

    def __init__(self, max_local_weight=0.6):
        super().__init__()
        self.max_local_weight = float(max_local_weight)
        self.register_buffer("local_weight", torch.tensor(0.0))

    def set_progress(self, progress):
        progress = min(max(float(progress), 0.0), 1.0)
        weight = self.max_local_weight * 0.5 * (1.0 - math.cos(math.pi * progress))
        self.local_weight.fill_(weight)

    def forward(self, vss_feat, local_feat):
        w_local = self.local_weight.to(dtype=vss_feat.dtype, device=vss_feat.device)
        return (1.0 - w_local) * vss_feat + w_local * local_feat


class ProgressiveHVSTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        
       
        self.norm1 = norm_layer(hidden_dim)
          # VSSLayer-only kwargs (not passed to VSSBranch)
        kwargs.pop("window_size", None)
        kwargs.pop("num_heads", None)
        self.vss_branch = VSSBranch(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            **kwargs,
        )
        
        self.local_branch = MultiScaleLocalBranch(hidden_dim)

        self.fusion = SmoothProgressiveFusion()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def set_training_progress(self, progress, epoch):

        self.fusion.set_progress(progress)
    
    def forward(self, input: torch.Tensor):
       
        x = self.norm1(input)

        vss_out = self.vss_branch(x)
        local_out = self.local_branch(x)

        fused = self.fusion(vss_out, local_out)

        return input + self.drop_path(fused)
