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
        use_enhanced_skip=config.use_enhanced_skip,  
        use_deep_supervision=config.use_deep_supervision,  
        use_hvst=config.use_hvst,  
        use_esc=config.use_esc,  
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
    max_miou = 0  
    start_epoch = 1
    min_epoch = 1
    best_epoch = 1 

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


    resume_training = getattr(config, 'resume_training', True) 
    
    if os.path.exists(resume_model) and resume_training:
        print('#----------Resume Model and Other params----------#')
        torch.cuda.empty_cache()
        
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        
        saved_epoch = checkpoint['epoch']
        min_loss_val = checkpoint['min_loss']
        min_epoch_val = checkpoint['min_epoch']
        loss_val = checkpoint['loss']
        max_miou = checkpoint.get('max_miou', 0)
        best_epoch = checkpoint.get('best_epoch', min_epoch_val)
    
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
        del checkpoint
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        start_epoch += saved_epoch
        min_loss = min_loss_val
        min_epoch = min_epoch_val
        loss = loss_val

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, max_miou: {max_miou:.4f}, best_epoch: {best_epoch}'
        logger.info(log_info)
    elif os.path.exists(resume_model) and not resume_training:
        print('#----------Skipping Resume: Starting Fresh Training----------#')
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
        
        if hasattr(config, 'warmup_epochs') and config.warmup_epochs > 0 and epoch <= config.warmup_epochs:
            warmup_factor = epoch / config.warmup_epochs
            for param_group in optimizer.param_groups:
                if epoch == 1 and 'initial_lr' not in param_group:
                    param_group['initial_lr'] = param_group['lr']
                param_group['lr'] = param_group['initial_lr'] * warmup_factor
            print(f'Warmup epoch {epoch}/{config.warmup_epochs}: lr_factor={warmup_factor:.4f}')
        elif hasattr(config, 'warmup_epochs') and config.warmup_epochs > 0 and epoch == config.warmup_epochs + 1:
            for param_group in optimizer.param_groups:
                if 'initial_lr' in param_group:
                    param_group['lr'] = param_group['initial_lr']
            print(f'Warmup finished, restored to initial learning rates')
        
    
        if config.use_hvst:
            progress = (epoch - 1) / config.epochs
            for layer in model.vmunet.layers:
                for block in layer.blocks:
                    if hasattr(block, 'set_training_progress'):
                        block.set_training_progress(progress, epoch)
        
        if hasattr(config, 'grad_clip_norm') and config.grad_clip_norm > 0:
            if epoch <= 70:
                current_grad_clip = config.grad_clip_norm  # 前期：1.0
            else:
                decay_progress = (epoch - 70) / (config.epochs - 70)
                current_grad_clip = config.grad_clip_norm * (1.0 - 0.5 * decay_progress)
            config.grad_clip_norm = current_grad_clip
        
        if epoch >= 70:
            decay_boost = 1.0 + 0.5 * ((epoch - 70) / (config.epochs - 70))  # 70→100: 1.0→1.5x
            for param_group in optimizer.param_groups:
                if 'initial_weight_decay' not in param_group:
                    param_group['initial_weight_decay'] = param_group['weight_decay']
                param_group['weight_decay'] = param_group['initial_weight_decay'] * decay_boost

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

        loss, miou = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )
        
        if miou > max_miou:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            max_miou = miou
            best_epoch = epoch
            logger.info(f'New best model: epoch {epoch}, mIoU: {miou:.6f}')
        
        if loss < min_loss:
            min_loss = loss
            min_epoch = epoch
        
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
                'miou': miou, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))
        
        
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
