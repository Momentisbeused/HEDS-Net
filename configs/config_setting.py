from torchvision import transforms
from utils import *

from datetime import datetime

class setting_config:
    
    
    use_axis_bridge = True
    use_coordinate_attention = True
    use_deep_supervision = True
    use_hvst = True
    
    
    warmup_epochs = 5
    
    grad_clip_norm = 1.0
    
   
    deep_supervision_weights = [0.3, 0.2, 0.1]
    deep_supervision_weight = 0.1
    model_config = {
        'num_classes': 1, 
        'input_channels': 3, 
        'depths': [2,2,2,2],
        'depths_decoder': [2,2,2,1],
        'drop_path_rate': 0.2,
        'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
        # module switches (mirrored on class attributes above)
        'use_axis_bridge': use_axis_bridge,
        'use_coordinate_attention': use_coordinate_attention,
        'use_deep_supervision': use_deep_supervision,
        'use_hvst': use_hvst,
    }

    datasets = 'isic17'
    if datasets == 'isic18':
        data_path = './data/isic18/'
    elif datasets == 'isic17':
        data_path = './data/isic17/'
    elif datasets ==  'Kvasir-SEG':
        data_path = './data/Kvasir-SEG/'
    else:
        raise ValueError(f"Unknown dataset key: {datasets!r}")

    
    criterion = BceDiceLoss()

    pretrained_path = './pre_trained/'
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 32
    epochs = 300

   
    resume_training = False
    work_dir = 'results/vmunet_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 1
    save_interval = 100
    threshold = 0.5
    only_test_and_save_figs = False
    best_ckpt_path = 'PATH_TO_YOUR_BEST_CKPT'
    img_save_path = 'PATH_TO_SAVE_IMAGES'

    
    train_transformer = transforms.Compose([
        myNormalizeUnit01(),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
    ])
    test_transformer = transforms.Compose([
        myNormalizeUnit01(),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

   
    save_intermediate_epochs = []
    visualization_epochs = []

    opt = "AdamW"
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.05
    amsgrad = False

    sch = "CosineAnnealingLR"
    T_max = epochs
    eta_min = 1e-6
    last_epoch = -1
