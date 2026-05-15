import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSupervisionHead(nn.Module):

    def __init__(self, in_channels, num_classes=1, lightweight=False):
        super().__init__()
        self.num_classes = num_classes
        bias = not lightweight
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=bias)
        if lightweight:
            nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.conv(x)


class DeepSupervision(nn.Module):
    def __init__(self, channels_list, num_classes=1, weights=None, lightweight=False):
        super().__init__()
        self.num_classes = num_classes
        self.channels_list = channels_list
        self.lightweight = lightweight

        if weights is None:
            self.weights = [0.4, 0.3, 0.2, 0.1][: len(channels_list)]
        else:
            self.weights = weights

        self.supervision_heads = nn.ModuleList()
        for channels in channels_list:
            self.supervision_heads.append(DeepSupervisionHead(channels, num_classes, lightweight=lightweight))

    def forward(self, features_list, target_size=None):
        predictions = []
        for features, head in zip(features_list, self.supervision_heads):
            pred = head(features)
            if target_size is not None:
                pred = F.interpolate(pred, size=target_size, mode="bilinear", align_corners=False)
            predictions.append(pred)
        return predictions

    def compute_loss(self, predictions, target, criterion=None, epoch=None, total_epochs=None):
        if criterion is None:
            from utils import BceDiceLoss, CeDiceLoss

            criterion = (
                BceDiceLoss(wb=1, wd=1) if self.num_classes == 1 else CeDiceLoss(self.num_classes)
            )

        total_loss = predictions[0].new_tensor(0.0)
        loss_components = {}

        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            if self.num_classes == 1:
                target_for_loss = target.float()
                if target_for_loss.dim() == 3:
                    target_for_loss = target_for_loss.unsqueeze(1)
                elif target_for_loss.dim() == 4 and target_for_loss.shape[1] != 1:
                    raise ValueError(f"Binary target should be Bx1xHxW, got {tuple(target_for_loss.shape)}")

                if pred.shape[-2:] != target_for_loss.shape[-2:]:
                    pred = F.interpolate(
                        pred,
                        size=target_for_loss.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
            else:
                if target.dim() == 4 and target.shape[1] == 1:
                    target_for_loss = target.squeeze(1).long()
                elif target.dim() == 3:
                    target_for_loss = target.long()
                else:
                    raise ValueError(f"Multi-class target should be BxHxW or Bx1xHxW, got {tuple(target.shape)}")

                if pred.shape[-2:] != target_for_loss.shape[-2:]:
                    pred = F.interpolate(
                        pred,
                        size=target_for_loss.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if pred.shape[1] != self.num_classes:
                    raise ValueError(
                        f"Multi-class deep supervision: logits C={pred.shape[1]} vs num_classes={self.num_classes}"
                    )

            layer_loss = criterion(pred, target_for_loss)

            w = float(weight)
            if epoch is not None and total_epochs is not None:
                decay = max(0.1, 1.0 - 0.8 * float(epoch) / float(total_epochs))
                w = w * decay

            weighted_loss = w * layer_loss
            total_loss = total_loss + weighted_loss
            loss_components[f"ds_layer_{i}"] = float(layer_loss.detach().cpu())
            loss_components[f"ds_weighted_{i}"] = float(weighted_loss.detach().cpu())

        return total_loss, loss_components


def create_deep_supervision(channels_list, num_classes=1, weights=None):
    return DeepSupervision(channels_list, num_classes, weights)
