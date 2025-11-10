"""
Progressive HVST Block v2: 多尺度局部增强 + 平滑渐进融合
核心改进：
1. 空洞卷积替代普通卷积（更大感受野，更少参数）
2. SE通道注意力增强特征质量
3. GroupNorm替代BatchNorm（小batch更稳定）
4. 余弦平滑权重增长（避免阶梯式突变）
5. 分离学习率支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from typing import Callable
from timm.models.layers import DropPath

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    selective_scan_fn = None


class SEModule(nn.Module):
    """Squeeze-and-Excitation通道注意力模块"""
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
    """深度可分离卷积（支持空洞卷积）"""
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
        # 确保num_groups能整除num_channels
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
    """
    多尺度局部增强分支 v2：
    - 使用空洞卷积扩大感受野（dilation: 1, 2, 4）
    - SE通道注意力增强特征
    - GroupNorm提升小batch稳定性
    - 零初始化融合层
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 三个尺度的空洞深度可分离卷积
        # dilation=1: 感受野3×3（细节特征）
        # dilation=2: 感受野5×5（中等结构）
        # dilation=4: 感受野9×9（大尺度边界）
        self.conv_d1 = DepthwiseSeparableConv(dim, dim, kernel_size=3, dilation=1)
        self.conv_d2 = DepthwiseSeparableConv(dim, dim, kernel_size=3, dilation=2)
        self.conv_d4 = DepthwiseSeparableConv(dim, dim, kernel_size=3, dilation=4)
        
        # SE通道注意力（增强多尺度特征质量）
        self.se = SEModule(dim * 3, reduction=4)
        
        # 多尺度融合层（零初始化，确保初期不干扰VSS分支）
        self.fusion = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False)
        nn.init.zeros_(self.fusion.weight)  # ✅ 零初始化
        
        # 使用GroupNorm替代LayerNorm（适配BCHW格式，小batch更稳定）
        # 确保num_groups能整除dim
        num_groups = 8
        while dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim)
    
    def forward(self, x):
        """
        Input: (B, H, W, C) - BHWC格式
        Output: (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # 转换为BCHW格式用于卷积
        x_c = x.permute(0, 3, 1, 2).contiguous()
        
        # 三个尺度的特征提取
        feat_d1 = self.conv_d1(x_c)  # (B, C, H, W)
        feat_d2 = self.conv_d2(x_c)  # (B, C, H, W)
        feat_d4 = self.conv_d4(x_c)  # (B, C, H, W)
        
        # 拼接多尺度特征
        multi_scale = torch.cat([feat_d1, feat_d2, feat_d4], dim=1)  # (B, 3C, H, W)
        
        # SE通道注意力
        multi_scale = self.se(multi_scale)  # (B, 3C, H, W)
        
        # 融合到目标维度
        fused = self.fusion(multi_scale)  # (B, C, H, W)
        fused = self.norm(fused)
        
        # 转回BHWC格式
        out = fused.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        
        return out


class VSSBranch(nn.Module):
    """
    VSS分支：复制自原VSSBlock的self_attention部分
    用于加载预训练权重
    """
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

        # x_proj权重（与SS2D一致）
        x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(4)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj], dim=0))
        
        # dt_projs初始化
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

        if selective_scan_fn is not None:
            out_y = selective_scan_fn(xs, dts, As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False).view(B, K, -1, L)
        else:
            out_y = xs.view(B, K, -1, L)

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
    """
    平滑渐进融合 v2：
    - 使用余弦曲线平滑增长权重（避免阶梯式突变）
    - 保守增长策略：0→0.1(30epoch)→0.5(80epoch)→0.6(100epoch)
    """
    def __init__(self, dim):
        super().__init__()
        # 局部分支的融合权重（可学习参数，初始化为0）
        self.local_weight = nn.Parameter(torch.tensor(0.0))
        # 最大权重限制（通过进度平滑增长）
        self.register_buffer('max_weight', torch.tensor(0.0))
    
    def set_progress(self, progress):
        """
        使用余弦平滑曲线设置最大权重（V3优化版：四阶段平滑增长）
        
        权重增长曲线（基于用户反馈优化）：
        - progress 0.0 (epoch 1):   max_weight = 0.0
        - progress 0.3 (epoch 30):  max_weight = 0.1  ← 初步适应
        - progress 0.5 (epoch 50):  max_weight = 0.3  ← 中间平台
        - progress 0.8 (epoch 80):  max_weight = 0.5  ← 充分融合
        - progress 1.0 (epoch 100): max_weight = 0.6  ← 最终状态
        
        四阶段设计理由：
        1. Epoch 1-30: 让多尺度分支学习基础特征
        2. Epoch 31-50: 逐步参与融合，观察稳定性
        3. Epoch 51-80: 主要融合阶段，VSS+Local协同
        4. Epoch 81-100: 微调最终权重分配
        """
        if progress < 0.3:
            # 阶段1: 0→0.1 (epoch 1-30)
            t = progress / 0.3
            max_w = 0.1 * (1 - math.cos(t * math.pi / 2))
        elif progress < 0.5:
            # 阶段2: 0.1→0.3 (epoch 31-50)
            t = (progress - 0.3) / 0.2
            max_w = 0.1 + 0.2 * (1 - math.cos(t * math.pi / 2))
        elif progress < 0.8:
            # 阶段3: 0.3→0.5 (epoch 51-80)
            t = (progress - 0.5) / 0.3
            max_w = 0.3 + 0.2 * (1 - math.cos(t * math.pi / 2))
        else:
            # 阶段4: 0.5→0.6 (epoch 81-100)
            t = (progress - 0.8) / 0.2
            max_w = 0.5 + 0.1 * (1 - math.cos(t * math.pi / 2))
        
        self.max_weight.fill_(max_w)
    
    def forward(self, vss_feat, local_feat):
        """
        平滑融合VSS和局部特征
        
        Args:
            vss_feat: VSS分支输出 (B, H, W, C)
            local_feat: 多尺度局部分支输出 (B, H, W, C)
        
        Returns:
            fused: 融合后的特征 (B, H, W, C)
        """
        # 限制局部分支权重（sigmoid确保在[0,1]，然后clamp到max_weight）
        w_local = torch.clamp(torch.sigmoid(self.local_weight), max=self.max_weight.item())
        w_vss = 1.0 - w_local
        
        # 简单加权融合
        fused = w_vss * vss_feat + w_local * local_feat
        return fused


class ProgressiveHVSTBlock(nn.Module):
    """
    Progressive HVST Block v2:
    - VSS分支：加载预训练权重，微调（低学习率）
    - 多尺度局部分支：空洞卷积 + SE，快速学习（高学习率）
    - 平滑融合：余弦曲线权重增长，无硬阶段划分
    """
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
        
        # Pre-Norm
        self.norm1 = norm_layer(hidden_dim)
        
        # VSS分支（预训练权重加载）
        self.vss_branch = VSSBranch(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            **kwargs
        )
        
        # 多尺度局部分支（从头训练）
        self.local_branch = MultiScaleLocalBranch(hidden_dim)
        
        # 平滑渐进融合
        self.fusion = SmoothProgressiveFusion(hidden_dim)
        
        # DropPath正则化
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # 不再需要硬冻结/解冻逻辑，所有组件从epoch 1开始训练
        # 通过分离学习率和平滑权重增长来控制训练过程
    
    def set_training_progress(self, progress, epoch):
        """
        设置训练进度（由train.py调用）
        
        Args:
            progress: 训练进度 [0.0, 1.0]
            epoch: 当前epoch
        """
        self.fusion.set_progress(progress)
    
    def forward(self, input: torch.Tensor):
        """
        前向传播
        
        Args:
            input: (B, H, W, C)
        
        Returns:
            output: (B, H, W, C)
        """
        # Pre-Norm
        x = self.norm1(input)
        
        # 双分支并行计算
        vss_out = self.vss_branch(x)       # VSS全局序列建模
        local_out = self.local_branch(x)   # 多尺度局部增强
        
        # 平滑渐进融合
        fused = self.fusion(vss_out, local_out)
        
        # 残差连接 + DropPath
        return input + self.drop_path(fused)
