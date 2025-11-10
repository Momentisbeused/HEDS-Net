import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DiceLoss

class DeepSupervisionHead(nn.Module):
    """
    深度监督头，用于在解码器中间层添加辅助损失
    """
    def __init__(self, in_channels, num_classes=1, lightweight=False):
        super(DeepSupervisionHead, self).__init__()
        self.num_classes = num_classes
        self.lightweight = lightweight
        
        if lightweight:
            # 轻量设计：仅1x1卷积+Sigmoid
            self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
            # 权重初始化
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        else:
            # 标准设计
            self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W) 或 (B, H, W, C)
        Returns:
            pred: 预测结果 (B, num_classes, H, W)
        """
        # 输入已经是 (B, C, H, W) 格式，直接使用
        # 不需要格式转换，因为VSSM已经转换过了
        
        pred = self.conv(x)
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
        return pred

class DeepSupervision(nn.Module):
    """
    深度监督模块，管理多个监督头
    """
    def __init__(self, channels_list, num_classes=1, weights=None, lightweight=False):
        super(DeepSupervision, self).__init__()
        self.num_classes = num_classes
        self.channels_list = channels_list
        self.lightweight = lightweight
        
        # 默认权重：越深的层权重越大
        if weights is None:
            self.weights = [0.4, 0.3, 0.2, 0.1][:len(channels_list)]
        else:
            self.weights = weights
            
        # 创建监督头
        self.supervision_heads = nn.ModuleList()
        for channels in channels_list:
            head = DeepSupervisionHead(channels, num_classes, lightweight=lightweight)
            self.supervision_heads.append(head)
    
    def forward(self, features_list, target_size=None):
        """
        Args:
            features_list: 特征列表 [(B, C1, H1, W1), (B, C2, H2, W2), ...]
            target_size: 目标尺寸 (H, W)，如果为None则使用第一个特征的尺寸
        Returns:
            predictions: 预测结果列表
        """
        predictions = []
        
        for features, head in zip(features_list, self.supervision_heads):
            # 获取预测结果
            pred = head(features)
            
            # 如果指定了目标尺寸，进行上采样
            if target_size is not None:
                pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
            
            predictions.append(pred)
            
        return predictions
    
    def compute_loss(self, predictions, target, criterion, epoch=None, total_epochs=None):
        """
        计算深度监督损失
        Args:
            predictions: 预测结果列表
            target: 真实标签 (B, 1, H, W)
            criterion: 损失函数
            epoch: 当前训练轮数
            total_epochs: 总训练轮数
        Returns:
            total_loss: 总损失
            loss_components: 各层损失组件
        """
        total_loss = 0
        loss_components = {}
        
        # 使用DiceLoss作为深度监督损失函数
        dice_criterion = DiceLoss()
        
        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            # 确保预测结果和目标尺寸匹配
            if pred.shape != target.shape:
                # 根据目标数据的维度确定目标尺寸
                if len(target.shape) == 5:  # 3D数据: (B, C, D, H, W)
                    target_size_2d = (target.shape[3], target.shape[4])  # (H, W)
                elif len(target.shape) == 4:  # 2D数据: (B, C, H, W)
                    target_size_2d = target.shape[2:]
                elif len(target.shape) == 3:  # Synapse数据: (B, H, W)
                    target_size_2d = (target.shape[1], target.shape[2])  # (H, W)
                else:
                    print(f"Warning: Unexpected target shape in compute_loss: {target.shape}")
                    target_size_2d = None
                
                if target_size_2d is not None:
                    pred = F.interpolate(pred, size=target_size_2d, mode='bilinear', align_corners=False)
            
            # 确保预测结果和目标数据的维度匹配
            if len(target.shape) == 3:  # Synapse数据: (B, H, W)
                # 为target添加通道维度
                target_4d = target.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
            else:
                target_4d = target
            
            # 确保pred和target_4d的通道数匹配
            if pred.shape[1] != target_4d.shape[1]:
                # 如果通道数不匹配，使用argmax将pred转换为单通道
                pred_single = torch.argmax(pred, dim=1, keepdim=True)  # (B, C, H, W) -> (B, 1, H, W)
                pred = pred_single.float()
            
            # 计算该层的损失 - 使用DiceLoss
            layer_loss = dice_criterion(pred, target_4d)
            
            # 权重衰减：训练后期逐渐降低深度监督权重
            if epoch is not None and total_epochs is not None:
                decay_factor = max(0.1, 1.0 - (epoch / total_epochs) * 0.8)  # 从1.0衰减到0.1
                weight = weight * decay_factor
            
            # 加权累加
            weighted_loss = weight * layer_loss
            total_loss += weighted_loss
            
            # 记录损失组件
            loss_components[f'ds_layer_{i}'] = layer_loss.item()
            loss_components[f'ds_weighted_{i}'] = weighted_loss.item()
        
        return total_loss, loss_components

def create_deep_supervision(channels_list, num_classes=1, weights=None):
    """
    创建深度监督模块的工厂函数
    Args:
        channels_list: 各层通道数列表
        num_classes: 类别数
        weights: 各层权重
    Returns:
        DeepSupervision模块
    """
    return DeepSupervision(channels_list, num_classes, weights)
