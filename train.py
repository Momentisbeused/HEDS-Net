import torch
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)





    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
        use_enhanced_skip=config.use_enhanced_skip,  # 使用配置中的开关
        use_deep_supervision=config.use_deep_supervision,  # 深度监督开关
        use_ca_attention=config.use_ca_attention,  # 坐标注意力开关
        use_hvst=config.use_hvst,  # HVST编码器开关
        use_esc=config.use_esc,  # ESC模块开关
    )
    model.load_from()
    model = model.cuda()

    cal_params_flops(model, 256, logger)





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)





    print('#----------Set other params----------#')
    min_loss = 999
    max_miou = 0  # 添加：基于mIoU保存最佳模型（学术规范）
    start_epoch = 1
    min_epoch = 1
    best_epoch = 1  # 记录最佳mIoU的epoch
    
    # 保存指定epoch的权重（用于对比实验）
    save_epochs = [5, 8, 11, 14, 17, 20]  # 需要保存权重的epoch列表（在20个epoch中均匀分布）
    epoch_save_dir = '/root/autodl-tmp/VM-UNet/other'
    os.makedirs(epoch_save_dir, exist_ok=True)
    print(f'将在以下epoch保存权重: {save_epochs}')
    

    if config.only_test_and_save_figs:
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        config.work_dir = config.img_save_path
        if not os.path.exists(config.work_dir + 'outputs/'):
            os.makedirs(config.work_dir + 'outputs/')
        loss = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
            )
        return




    # 恢复训练选项：如果设置了resume_training=False，则从头开始训练
    resume_training = getattr(config, 'resume_training', True)  # 默认允许恢复
    
    if os.path.exists(resume_model) and resume_training:
        print('#----------Resume Model and Other params----------#')
        # 先清理GPU缓存，确保内存充足
        torch.cuda.empty_cache()
        
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        
        # 先提取需要的信息
        saved_epoch = checkpoint['epoch']
        min_loss_val = checkpoint['min_loss']
        min_epoch_val = checkpoint['min_epoch']
        loss_val = checkpoint['loss']
        max_miou = checkpoint.get('max_miou', 0)
        best_epoch = checkpoint.get('best_epoch', min_epoch_val)
        
        # 加载模型、优化器和调度器状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 清理checkpoint占用的内存，并清理GPU缓存
        del checkpoint
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # 设置恢复的训练状态
        start_epoch += saved_epoch
        min_loss = min_loss_val
        min_epoch = min_epoch_val
        loss = loss_val

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, max_miou: {max_miou:.4f}, best_epoch: {best_epoch}'
        logger.info(log_info)
    elif os.path.exists(resume_model) and not resume_training:
        print('#----------Skipping Resume: Starting Fresh Training----------#')
        # 如果不想恢复，可以删除或重命名旧checkpoint
        import shutil
        backup_path = resume_model + '.backup'
        if os.path.exists(resume_model):
            shutil.move(resume_model, backup_path)
            logger.info(f'Moved old checkpoint to: {backup_path}')
    else:
        print('#----------Starting Fresh Training----------#')
        logger.info('Starting fresh training (no checkpoint found)')




    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()
        
        # 学习率预热（前5个epoch）
        if hasattr(config, 'warmup_epochs') and config.warmup_epochs > 0 and epoch <= config.warmup_epochs:
            warmup_factor = epoch / config.warmup_epochs
            for param_group in optimizer.param_groups:
                # 保存原始学习率（第一次预热时）
                if epoch == 1 and 'initial_lr' not in param_group:
                    param_group['initial_lr'] = param_group['lr']
                # 应用预热因子
                param_group['lr'] = param_group['initial_lr'] * warmup_factor
            print(f'Warmup epoch {epoch}/{config.warmup_epochs}: lr_factor={warmup_factor:.4f}')
        elif hasattr(config, 'warmup_epochs') and config.warmup_epochs > 0 and epoch == config.warmup_epochs + 1:
            # 预热结束，恢复原始学习率
            for param_group in optimizer.param_groups:
                if 'initial_lr' in param_group:
                    param_group['lr'] = param_group['initial_lr']
            print(f'Warmup finished, restored to initial learning rates')
        
        # ========== 动态训练策略 ==========
        
        # 1. HVST渐进式训练控制
        if config.use_hvst:
            progress = (epoch - 1) / config.epochs
            for layer in model.vmunet.layers:
                for block in layer.blocks:
                    if hasattr(block, 'set_training_progress'):
                        block.set_training_progress(progress, epoch)
        
        # 2. 动态梯度裁剪（70 epoch后减小，提高精度）
        if hasattr(config, 'grad_clip_norm') and config.grad_clip_norm > 0:
            if epoch <= 70:
                current_grad_clip = config.grad_clip_norm  # 前期：1.0
            else:
                # 后期线性递减到0.5（70→100: 1.0→0.5）
                decay_progress = (epoch - 70) / (config.epochs - 70)
                current_grad_clip = config.grad_clip_norm * (1.0 - 0.5 * decay_progress)
            config.grad_clip_norm = current_grad_clip
        
        # 3. 动态权重衰减（后期增强，防止过拟合）
        if epoch >= 70:
            decay_boost = 1.0 + 0.5 * ((epoch - 70) / (config.epochs - 70))  # 70→100: 1.0→1.5x
            for param_group in optimizer.param_groups:
                if 'initial_weight_decay' not in param_group:
                    param_group['initial_weight_decay'] = param_group['weight_decay']
                param_group['weight_decay'] = param_group['initial_weight_decay'] * decay_boost

        # 预热期间不使用scheduler
        use_scheduler = not (hasattr(config, 'warmup_epochs') and config.warmup_epochs > 0 and epoch <= config.warmup_epochs)
        
        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler if use_scheduler else None,
            epoch,
            step,
            logger,
            config,
            writer
        )

        # 验证并获取loss和mIoU
        loss, miou = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )
        
        # 保存最佳模型（基于mIoU，符合学术规范）
        if miou > max_miou:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            max_miou = miou
            best_epoch = epoch
            logger.info(f'New best model: epoch {epoch}, mIoU: {miou:.6f}')
        
        # 同时记录最小loss（用于对比）
        if loss < min_loss:
            min_loss = loss
            min_epoch = epoch
        
        # 在指定epoch保存权重（用于对比实验）
        if epoch in save_epochs:
            epoch_save_path = os.path.join(epoch_save_dir, f'epoch_{epoch}_{config.datasets}_miou_{miou:.4f}.pth')
            torch.save(model.state_dict(), epoch_save_path)
            logger.info(f'保存epoch {epoch}权重: mIoU: {miou:.4f}, Loss: {loss:.4f} -> {os.path.basename(epoch_save_path)}')

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'max_miou': max_miou,
                'best_epoch': best_epoch,
                'loss': loss,
                'miou': miou,  # 保存当前miou，便于后续恢复
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))
        
        # 在指定epoch保存权重到可视化目录（用于生成对比图）
        visualization_epochs = [60, 70, 80, 90, 95]
        if epoch in visualization_epochs:
            vis_save_dir = '/root/autodl-tmp/VM-UNet/results/visual'
            os.makedirs(vis_save_dir, exist_ok=True)
            vis_save_path = os.path.join(vis_save_dir, f'epoch_{epoch}_miou_{miou:.4f}.pth')
            torch.save(model.state_dict(), vis_save_path)
            logger.info(f'保存可视化用权重: {vis_save_path} (epoch {epoch}, mIoU: {miou:.4f})') 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-miou{max_miou:.4f}.pth')
        )      


if __name__ == '__main__':
    config = setting_config
    main(config)