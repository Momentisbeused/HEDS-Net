import torch
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.vmunet import VMUNet


from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config


def _held_out_test_available(data_path):
    base = os.path.normpath(data_path)
    ti = os.path.join(base, 'test', 'images')
    tm = os.path.join(base, 'test', 'masks')
    if not (os.path.isdir(ti) and os.path.isdir(tm)):
        return False
    try:
        return len(os.listdir(ti)) > 0 and len(os.listdir(tm)) > 0
    except OSError:
        return False


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
                                drop_last=False)

    test_loader = None
    if _held_out_test_available(config.data_path):
        test_dataset = NPY_datasets(config.data_path, config, split='test')
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
            drop_last=False,
        )
        logger.info('Detected held-out test set under data_path (test/images, test/masks). Final eval will use test only.')
    else:
        logger.warning(
            'No held-out test/ split under data_path. Best model is chosen by val mIoU; any post-train eval on val is NOT '
            'an independent test set — do not report it as blind test metrics in the paper.'
        )





    print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
        use_axis_bridge=model_cfg.get('use_axis_bridge', getattr(config, 'use_axis_bridge', True)),
        use_deep_supervision=model_cfg.get('use_deep_supervision', getattr(config, 'use_deep_supervision', True)),
        use_coordinate_attention=model_cfg.get('use_coordinate_attention', getattr(config, 'use_coordinate_attention', True)),
        use_hvst=model_cfg.get('use_hvst', getattr(config, 'use_hvst', True)),
    )
    model.load_from()
    model = model.cuda()





    print('#----------Preparing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)





    print('#----------Set other params----------#')
    min_loss = 999
    max_miou = 0.0
    start_epoch = 1
    min_epoch = 1
    best_epoch = 1

    save_epochs = getattr(config, 'save_intermediate_epochs', [])
    epoch_save_dir = os.path.join(config.work_dir, 'epoch_snapshots')
    if save_epochs:
        os.makedirs(epoch_save_dir, exist_ok=True)
        print(f'Extra checkpoints at epochs (under work_dir): {save_epochs}')
    

    if config.only_test_and_save_figs:
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        config.work_dir = config.img_save_path
        if not os.path.exists(config.work_dir + 'outputs/'):
            os.makedirs(config.work_dir + 'outputs/')
        eval_loader = test_loader if test_loader is not None else val_loader
        eval_split = 'test' if test_loader is not None else 'validation'
        loss = test_one_epoch(
                eval_loader,
                model,
                criterion,
                logger,
                config,
                test_data_name=config.datasets,
                eval_split=eval_split,
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
    last_loss, last_miou = 0.0, 0.0
    val_interval = max(1, int(getattr(config, "val_interval", 1)))
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
            progress = (epoch - 1) / max(config.epochs - 1, 1)
            for layer in model.vmunet.layers:
                for block in layer.blocks:
                    if hasattr(block, 'set_training_progress'):
                        block.set_training_progress(progress, epoch)
        
        current_grad_clip = float(getattr(config, "grad_clip_norm", 0) or 0)

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
            writer,
            grad_clip_norm=current_grad_clip if current_grad_clip > 0 else None,
        )

        run_val = (epoch % val_interval == 0) or (epoch == start_epoch)
        if run_val:
            loss, miou = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config,
            )
            last_loss, last_miou = loss, miou
        else:
            loss, miou = last_loss, last_miou

        if run_val and miou > max_miou:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            max_miou = miou
            best_epoch = epoch
            logger.info(f'New best model: epoch {epoch}, mIoU: {miou:.6f}')
        
        if run_val and loss < min_loss:
            min_loss = loss
            min_epoch = epoch
        
        if save_epochs and epoch in save_epochs and run_val:
            epoch_save_path = os.path.join(epoch_save_dir, f'epoch_{epoch}_{config.datasets}_miou_{miou:.4f}.pth')
            torch.save(model.state_dict(), epoch_save_path)
            logger.info(f'Saved epoch snapshot: {os.path.basename(epoch_save_path)} mIoU={miou:.4f} loss={loss:.4f}')

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
        
        vis_epochs = getattr(config, 'visualization_epochs', [])
        if vis_epochs and epoch in vis_epochs and run_val:
            vis_save_dir = os.path.join(config.work_dir, 'visual_ckpts')
            os.makedirs(vis_save_dir, exist_ok=True)
            vis_save_path = os.path.join(vis_save_dir, f'epoch_{epoch}_miou_{miou:.4f}.pth')
            torch.save(model.state_dict(), vis_save_path)
            logger.info(f'Saved vis checkpoint: {vis_save_path} (epoch {epoch}, mIoU: {miou:.4f})')

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Final evaluation (best checkpoint)----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        if test_loader is not None:
            logger.info('Evaluating best checkpoint on held-out TEST set (not used for model selection).')
            test_one_epoch(
                test_loader,
                model,
                criterion,
                logger,
                config,
                test_data_name=config.datasets,
                eval_split='test',
            )
        else:
            logger.info(
                "Skipping final checkpoint evaluation: no held-out test/images+test/masks. "
                "Add a test split and run again for paper test metrics; do not use val as test."
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-miou{max_miou:.4f}.pth')
        )      


if __name__ == '__main__':
    config = setting_config
    main(config)
