"""
AXIS-Bridge: context-conditioned mixing plus optional coordinate attention.

Wired into the decoder skip path in ``models/vmunet/vmamba.py`` (``VSSM.forward_features_up``)
when ``use_axis_bridge=True`` via ``self.axis_bridges``; this module is not a dead file.
"""

import torch
import torch.nn as nn


class EnhancedGateAttention(nn.Module):
    """Context-conditioned mixing for decoder/encoder skip features."""

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.reduction = reduction
        reduced_channels = max(in_channels // reduction, 1)

        self.W_g = nn.Conv2d(in_channels, reduced_channels, 1, 1, 0, bias=False)
        self.W_x = nn.Conv2d(in_channels, reduced_channels, 1, 1, 0, bias=False)
        self.W_psi = nn.Conv2d(reduced_channels, in_channels, 1, 1, 0, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, decoder_feat, encoder_feat):
        if decoder_feat.dim() == 4 and decoder_feat.shape[-1] != decoder_feat.shape[1]:
            decoder_feat = decoder_feat.permute(0, 3, 1, 2)
            encoder_feat = encoder_feat.permute(0, 3, 1, 2)
            need_permute_back = True
        else:
            need_permute_back = False

        g = self.W_g(decoder_feat)
        x = self.W_x(encoder_feat)
        psi = self.relu(g + x)
        psi = self.sigmoid(self.W_psi(psi))
        result = decoder_feat + encoder_feat * psi

        if need_permute_back:
            result = result.permute(0, 2, 3, 1)
        return result


class CoordinateAttention(nn.Module):
    """Axis-aware refinement (coordinate attention)."""

    def __init__(self, channels, reduction=32):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, max(channels // reduction, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 1), channels, 1, bias=False),
        )

    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] != x.shape[1]:
            x = x.permute(0, 3, 1, 2)
            need_permute_back = True
        else:
            need_permute_back = False

        identity = x
        x_h = self.avg_pool_h(x)
        x_w = self.avg_pool_w(x)
        x_w = x_w.permute(0, 1, 3, 2)
        concat = torch.cat([x_h, x_w], dim=2)
        out = self.mlp(concat)
        x_h, x_w = torch.split(out, [x_h.size(2), x_w.size(2)], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(x_h)
        a_w = torch.sigmoid(x_w)
        out = identity * a_w * a_h

        if need_permute_back:
            out = out.permute(0, 2, 3, 1)
        return out


class AxisBridge(nn.Module):
    """
    Full AXIS-Bridge: context gate + optional coordinate attention.
    When use_coordinate_attention=False, only the context gate is applied.
    """

    def __init__(self, channels, reduction=4, use_coordinate_attention=True, ca_reduction=32):
        super().__init__()
        self.context_gate = EnhancedGateAttention(channels, reduction=reduction)
        self.axis_refine = (
            CoordinateAttention(channels, reduction=ca_reduction)
            if use_coordinate_attention
            else nn.Identity()
        )

    def forward(self, decoder_feat, encoder_feat):
        x = self.context_gate(decoder_feat, encoder_feat)
        x = self.axis_refine(x)
        return x


# Backward compatibility for old checkpoints / external references
EnhancedSkipConnection = AxisBridge
