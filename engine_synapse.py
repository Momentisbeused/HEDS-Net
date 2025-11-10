import numpy as np
from tqdm import tqdm

from torch.cuda.amp import autocast as autocast
import torch

from sklearn.metrics import confusion_matrix

from scipy.ndimage.morphology import binary_fill_holes, binary_opening

from utils import test_single_volume

import time


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    stime = time.time()
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()

        images, targets = data['image'], data['label']
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()   

        if config.amp:
            with autocast():
                output = model(images)
                # 处理深度监督输出
                if isinstance(output, tuple):
                    out, ds_features = output
                    # 计算主损失
                    main_loss = criterion(out, targets)
                    
                    # 计算深度监督损失
                    if hasattr(model.module, 'vmunet') and hasattr(model.module.vmunet, 'deep_supervision'):
                        # 根据数据维度确定目标尺寸
                        if len(targets.shape) == 5:  # 3D数据: (B, C, D, H, W)
                            target_size_2d = (targets.shape[3], targets.shape[4])  # (H, W)
                        elif len(targets.shape) == 4:  # 2D数据: (B, C, H, W)
                            target_size_2d = (targets.shape[2], targets.shape[3])  # (H, W)
                        elif len(targets.shape) == 3:  # Synapse数据: (B, H, W)
                            target_size_2d = (targets.shape[1], targets.shape[2])  # (H, W)
                        else:
                            print(f"Warning: Unexpected targets shape: {targets.shape}")
                            target_size_2d = None
                        
                        if target_size_2d is not None:
                            ds_predictions = model.module.vmunet.deep_supervision(ds_features, target_size=target_size_2d)
                            ds_loss, ds_components = model.module.vmunet.deep_supervision.compute_loss(
                                ds_predictions, targets, criterion, epoch=epoch, total_epochs=config.epochs
                            )
                            loss = main_loss + config.deep_supervision_weight * ds_loss
                        else:
                            loss = main_loss
                    else:
                        loss = main_loss
                else:
                    out = output
                    loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images)
            # 处理深度监督输出
            if isinstance(output, tuple):
                out, ds_features = output
                # 计算主损失
                main_loss = criterion(out, targets)
                
                # 计算深度监督损失
                if hasattr(model.module, 'vmunet') and hasattr(model.module.vmunet, 'deep_supervision'):
                    # 根据数据维度确定目标尺寸
                    if len(targets.shape) == 5:  # 3D数据: (B, C, D, H, W)
                        target_size_2d = (targets.shape[3], targets.shape[4])  # (H, W)
                    elif len(targets.shape) == 4:  # 2D数据: (B, C, H, W)
                        target_size_2d = (targets.shape[2], targets.shape[3])  # (H, W)
                    elif len(targets.shape) == 3:  # Synapse数据: (B, H, W)
                        target_size_2d = (targets.shape[1], targets.shape[2])  # (H, W)
                    else:
                        print(f"Warning: Unexpected targets shape: {targets.shape}")
                        target_size_2d = None
                    
                    if target_size_2d is not None:
                        ds_predictions = model.module.vmunet.deep_supervision(ds_features, target_size=target_size_2d)
                        ds_loss, ds_components = model.module.vmunet.deep_supervision.compute_loss(
                            ds_predictions, targets, criterion, epoch=epoch, total_epochs=config.epochs
                        )
                        loss = main_loss + config.deep_supervision_weight * ds_loss
                    else:
                        loss = main_loss
                else:
                    loss = main_loss
            else:
                out = output
                loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        mean_loss = np.mean(loss_list)
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {loss.item():.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    if scheduler is not None:
        scheduler.step()
    etime = time.time()
    log_info = f'Finish one epoch train: epoch {epoch}, loss: {mean_loss:.4f}, time(s): {etime-stime:.2f}'
    print(log_info)
    logger.info(log_info)
    return mean_loss





def val_one_epoch(test_datasets,
                    test_loader,
                    model,
                    epoch, 
                    logger,
                    config,
                    test_save_path,
                    val_or_test=False):
    # switch to evaluate mode
    stime = time.time()
    model.eval()
    with torch.no_grad():
        metric_list = 0.0
        i_batch = 0
        for data in tqdm(test_loader):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            metric_i = test_single_volume(img, msk, model, classes=config.num_classes, patch_size=[config.input_size_h, config.input_size_w],
                                    test_save_path=test_save_path, case=case_name, z_spacing=config.z_spacing, val_or_test=val_or_test)
            metric_list += np.array(metric_i)

            logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name,
                        np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            i_batch += 1
        metric_list = metric_list / len(test_datasets)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        for i in range(1, config.num_classes):
            logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {performance}, mean_hd95: {mean_hd95}, time(s): {etime-stime:.2f}'
        print(log_info)
        logger.info(log_info)
    
    return performance, mean_hd95