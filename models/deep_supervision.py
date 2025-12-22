import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DiceLoss

class DeepSupervisionHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, lightweight=False):
        super(DeepSupervisionHead, self).__init__()
        self.num_classes = num_classes
        self.lightweight = lightweight
        
        if lightweight:
            self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        else:
            self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
       
        
        pred = self.conv(x)
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
        return pred

class DeepSupervision(nn.Module):
   
    def __init__(self, channels_list, num_classes=1, weights=None, lightweight=False):
        super(DeepSupervision, self).__init__()
        self.num_classes = num_classes
        self.channels_list = channels_list
        self.lightweight = lightweight
        
       
        if weights is None:
            self.weights = [0.4, 0.3, 0.2, 0.1][:len(channels_list)]
        else:
            self.weights = weights
            
      
        self.supervision_heads = nn.ModuleList()
        for channels in channels_list:
            head = DeepSupervisionHead(channels, num_classes, lightweight=lightweight)
            self.supervision_heads.append(head)
    
    def forward(self, features_list, target_size=None):
       
        predictions = []
        
        for features, head in zip(features_list, self.supervision_heads):
            # 获取预测结果
            pred = head(features)
            
           
            if target_size is not None:
                pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
            
            predictions.append(pred)
            
        return predictions
    
    def compute_loss(self, predictions, target, criterion, epoch=None, total_epochs=None):
        
        total_loss = 0
        loss_components = {}
        
       
        dice_criterion = DiceLoss()
        
        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
          
            if pred.shape != target.shape:
             
                if len(target.shape) == 5:  
                    target_size_2d = (target.shape[3], target.shape[4])  # (H, W)
                elif len(target.shape) == 4:  
                    target_size_2d = target.shape[2:]
                elif len(target.shape) == 3:  
                    target_size_2d = (target.shape[1], target.shape[2])  # (H, W)
                else:
                    print(f"Warning: Unexpected target shape in compute_loss: {target.shape}")
                    target_size_2d = None
                
                if target_size_2d is not None:
                    pred = F.interpolate(pred, size=target_size_2d, mode='bilinear', align_corners=False)
            
           
            if len(target.shape) == 3:  
                
                target_4d = target.unsqueeze(1) 
            else:
                target_4d = target
            
            
            if pred.shape[1] != target_4d.shape[1]:
                
                pred_single = torch.argmax(pred, dim=1, keepdim=True)  
                pred = pred_single.float()
            
           
            layer_loss = dice_criterion(pred, target_4d)
            
           
            if epoch is not None and total_epochs is not None:
                decay_factor = max(0.1, 1.0 - (epoch / total_epochs) * 0.8)  # 从1.0衰减到0.1
                weight = weight * decay_factor
            
           
            weighted_loss = weight * layer_loss
            total_loss += weighted_loss
            
           
            loss_components[f'ds_layer_{i}'] = layer_loss.item()
            loss_components[f'ds_weighted_{i}'] = weighted_loss.item()
        
        return total_loss, loss_components

def create_deep_supervision(channels_list, num_classes=1, weights=None):
    
    return DeepSupervision(channels_list, num_classes, weights)
