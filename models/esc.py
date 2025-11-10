"""
Enhanced Skip Connection (ESC) Module
整合增强跳跃连接的所有相关功能：
1. Enhanced Gate Attention - 增强门控注意力
2. MultiScaleFusion - 多尺度融合
3. CoordinateAttention - 坐标注意力
4. EnhancedSkipConnection - 增强跳跃连接主模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedGateAttention(nn.Module):
    """
    Enhanced Gate Attention for skip connections
    Improved version with better parameter efficiency and stability
    """
    def __init__(self, in_channels, reduction=4):
        """
        Args:
            in_channels: number of input channels (same for both decoder and encoder)
            reduction: channel reduction ratio
        """
        super(EnhancedGateAttention, self).__init__()
        self.reduction = reduction
        reduced_channels = in_channels // reduction
        
        # 降低中间维度，减少计算量
        self.W_g = nn.Conv2d(in_channels, reduced_channels, 1, 1, 0, bias=False)
        self.W_x = nn.Conv2d(in_channels, reduced_channels, 1, 1, 0, bias=False)
        self.W_psi = nn.Conv2d(reduced_channels, in_channels, 1, 1, 0, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，确保训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, decoder_feat, encoder_feat):
        """
        Args:
            decoder_feat: decoder features (B, H, W, C) or (B, C, H, W)
            encoder_feat: encoder features (B, H, W, C) or (B, C, H, W)
        Returns:
            enhanced skip connection (B, H, W, C) or (B, C, H, W)
        """
        # Handle both (B, H, W, C) and (B, C, H, W) formats
        if decoder_feat.dim() == 4 and decoder_feat.shape[-1] != decoder_feat.shape[1]:  # (B, H, W, C) format
            decoder_feat = decoder_feat.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
            encoder_feat = encoder_feat.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
            need_permute_back = True
        else:
            need_permute_back = False
        
        # 解码器特征转换
        g = self.W_g(decoder_feat)  # (B, C//r, H, W)
        # 编码器特征转换
        x = self.W_x(encoder_feat)  # (B, C//r, H, W)
        # 特征融合与非线性激活
        psi = self.relu(g + x)  # 采用相加而非拼接，增强特征交互
        # 注意力权重生成（恢复原通道数）
        psi = self.sigmoid(self.W_psi(psi))  # (B, C, H, W)
        # 编码器特征加权 + 解码器特征直接传递
        result = decoder_feat + encoder_feat * psi
        
        # Convert back to original format if needed
        if need_permute_back:
            result = result.permute(0, 2, 3, 1)  # Convert back to (B, H, W, C)
            
        return result


class MultiScaleFusion(nn.Module):
    """
    Multi-scale feature fusion for skip connections
    Uses 1x1, 3x3, 5x5 convolutions to capture multi-scale features
    """
    def __init__(self, channels):
        super(MultiScaleFusion, self).__init__()
        self.conv1x1 = nn.Conv2d(channels * 2, channels, 1)
        self.conv3x3 = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(channels * 2, channels, 5, padding=2)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        """
        Args:
            x: decoder features (B, H, W, C) or (B, C, H, W)
            skip: encoder features (B, H, W, C) or (B, C, H, W)
        Returns:
            fused features (B, H, W, C) or (B, C, H, W)
        """
        # Handle both (B, H, W, C) and (B, C, H, W) formats
        if x.dim() == 4 and x.shape[-1] != x.shape[1]:  # (B, H, W, C) format
            x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
            skip = skip.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
            need_permute_back = True
        else:
            need_permute_back = False
            
        concat = torch.cat([x, skip], dim=1)
        f1 = self.conv1x1(concat)
        f3 = self.conv3x3(concat)
        f5 = self.conv5x5(concat)
        
        result = self.fusion(torch.cat([f1, f3, f5], dim=1))
        
        # Convert back to original format if needed
        if need_permute_back:
            result = result.permute(0, 2, 3, 1)  # Convert back to (B, H, W, C)
            
        return result


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for Efficient Mobile Network Design
    Captures channel and spatial relationships efficiently
    """
    def __init__(self, channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # 全局平均池化
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # 共享MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
    def forward(self, x):
        """
        Args:
            x: input features (B, H, W, C) or (B, C, H, W)
        Returns:
            enhanced features (B, H, W, C) or (B, C, H, W)
        """
        # Handle both (B, H, W, C) and (B, C, H, W) formats
        if x.dim() == 4 and x.shape[-1] != x.shape[1]:  # (B, H, W, C) format
            x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
            need_permute_back = True
        else:
            need_permute_back = False
            
        identity = x
        
        # 高度方向的全局平均池化
        x_h = self.avg_pool_h(x)  # (B, C, H, 1)
        # 宽度方向的全局平均池化
        x_w = self.avg_pool_w(x)  # (B, C, 1, W)
        
        # 拼接并转置
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, W, 1)
        concat = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        
        # 共享MLP
        out = self.mlp(concat)
        
        # 分离高度和宽度注意力
        x_h, x_w = torch.split(out, [x_h.size(2), x_w.size(2)], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, 1, W)
        
        # 生成注意力权重
        a_h = torch.sigmoid(x_h)
        a_w = torch.sigmoid(x_w)
        
        # 应用注意力
        out = identity * a_w * a_h
        
        # Convert back to original format if needed
        if need_permute_back:
            out = out.permute(0, 2, 3, 1)  # Convert back to (B, H, W, C)
        
        return out


class EnhancedSkipConnection(nn.Module):
    """
    Enhanced skip connection using improved gate attention
    """
    def __init__(self, channels, reduction=4):
        super(EnhancedSkipConnection, self).__init__()
        self.attention_gate = EnhancedGateAttention(channels, reduction=reduction)
        
    def forward(self, decoder_feat, encoder_feat):
        """
        Args:
            decoder_feat: decoder features (B, C, H, W) or (B, H, W, C)
            encoder_feat: encoder features (B, C, H, W) or (B, H, W, C)
        Returns:
            enhanced skip connection (B, C, H, W) or (B, H, W, C)
        """
        # Apply enhanced gate attention
        enhanced_features = self.attention_gate(decoder_feat, encoder_feat)
        
        return enhanced_features


# 工厂函数
def create_enhanced_skip_connection(channels, reduction=4, use_enhanced_skip=True, use_ca_attention=True):
    """
    创建增强跳跃连接模块的工厂函数
    Args:
        channels: 输入通道数
        reduction: 通道缩减比例
        use_enhanced_skip: 是否使用增强跳跃连接
        use_ca_attention: 是否使用坐标注意力
    Returns:
        EnhancedSkipConnection模块
    """
    return EnhancedSkipConnection(channels, reduction, use_enhanced_skip, use_ca_attention)


if __name__ == "__main__":
    # Test the modules
    channels = 96
    x = torch.randn(2, channels, 64, 64)
    skip = torch.randn(2, channels, 64, 64)
    
    # Test Enhanced Gate Attention
    att_gate = EnhancedGateAttention(channels, channels // 4)
    output1 = att_gate(x, skip)
    print(f"Enhanced Gate Attention output shape: {output1.shape}")
    
    # Test Multi-scale Fusion
    ms_fusion = MultiScaleFusion(channels)
    output2 = ms_fusion(x, skip)
    print(f"Multi-scale Fusion output shape: {output2.shape}")
    
    # Test Coordinate Attention
    ca_attention = CoordinateAttention(channels)
    output3 = ca_attention(x)
    print(f"Coordinate Attention output shape: {output3.shape}")
    
    # Test Enhanced Skip Connection
    enhanced_skip = EnhancedSkipConnection(channels)
    output4 = enhanced_skip(x, skip)
    print(f"Enhanced Skip Connection output shape: {output4.shape}")
