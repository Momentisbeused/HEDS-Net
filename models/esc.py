import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedGateAttention(nn.Module):
   
    def __init__(self, in_channels, reduction=4):
       
        super(EnhancedGateAttention, self).__init__()
        self.reduction = reduction
        reduced_channels = in_channels // reduction
        
       
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
        
        # Convert back to original format if needed
        if need_permute_back:
            result = result.permute(0, 2, 3, 1)  
            
        return result


class MultiScaleFusion(nn.Module):
    
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
        
        if x.dim() == 4 and x.shape[-1] != x.shape[1]:  
            x = x.permute(0, 3, 1, 2)  
            skip = skip.permute(0, 3, 1, 2) 
            need_permute_back = True
        else:
            need_permute_back = False
            
        concat = torch.cat([x, skip], dim=1)
        f1 = self.conv1x1(concat)
        f3 = self.conv3x3(concat)
        f5 = self.conv5x5(concat)
        
        result = self.fusion(torch.cat([f1, f3, f5], dim=1))
        
        
        if need_permute_back:
            result = result.permute(0, 2, 3, 1)  
            
        return result


class CoordinateAttention(nn.Module):
   
    def __init__(self, channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
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
            out = out.permute(0, 2, 3, 1)  # Convert back to (B, H, W, C)
        
        return out


class EnhancedSkipConnection(nn.Module):
   
    def __init__(self, channels, reduction=4):
        super(EnhancedSkipConnection, self).__init__()
        self.attention_gate = EnhancedGateAttention(channels, reduction=reduction)
        
    def forward(self, decoder_feat, encoder_feat):
       
        enhanced_features = self.attention_gate(decoder_feat, encoder_feat)
        
        return enhanced_features



def create_enhanced_skip_connection(channels, reduction=4, use_enhanced_skip=True, use_ca_attention=True):
    
    return EnhancedSkipConnection(channels, reduction, use_enhanced_skip, use_ca_attention)


if __name__ == "__main__":
   
    channels = 96
    x = torch.randn(2, channels, 64, 64)
    skip = torch.randn(2, channels, 64, 64)
    
   
    att_gate = EnhancedGateAttention(channels, channels // 4)
    output1 = att_gate(x, skip)
    print(f"Enhanced Gate Attention output shape: {output1.shape}")
    
    
    ms_fusion = MultiScaleFusion(channels)
    output2 = ms_fusion(x, skip)
    print(f"Multi-scale Fusion output shape: {output2.shape}")
    
   
    ca_attention = CoordinateAttention(channels)
    output3 = ca_attention(x)
    print(f"Coordinate Attention output shape: {output3.shape}")
    
   
    enhanced_skip = EnhancedSkipConnection(channels)
    output4 = enhanced_skip(x, skip)
    print(f"Enhanced Skip Connection output shape: {output4.shape}")
