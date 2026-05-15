import numpy as np
from tqdm import tqdm
import torch
from sklearn .metrics import confusion_matrix
from utils import save_imgs

def _binary_metrics_from_logits (preds_flat ,gts_flat ,threshold ):
    preds_flat =np .nan_to_num (np .asarray (preds_flat ,dtype =np .float64 ).reshape (-1 ),nan =0.0 )
    gts_flat =np .asarray (gts_flat ,dtype =np .float64 ).reshape (-1 )
    n =preds_flat .size
    if n ==0 :
        cm =np .zeros ((2 ,2 ),dtype =np .int64 )
        return (0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,cm )
    y_pre =np .where (preds_flat >=threshold ,1 ,0 )
    y_true =np .where (gts_flat >=0.5 ,1 ,0 )
    try :
        cm =confusion_matrix (y_true ,y_pre ,labels =[0 ,1 ])
    except ValueError :
        cm =np .zeros ((2 ,2 ),dtype =np .int64 )
    cm =np .asarray (cm ,dtype =np .int64 )
    if cm .size !=4 :
        cm =np .zeros ((2 ,2 ),dtype =np .int64 )
    else :
        cm =cm .reshape (2 ,2 )
    TN ,FP ,FN ,TP =(int (x )for x in cm .ravel ())
    denom =float (np .sum (cm ))
    accuracy =float (TN +TP )/denom if denom >0 else 0.0
    sensitivity =float (TP )/float (TP +FN )if float (TP +FN )>0 else 0.0
    specificity =float (TN )/float (TN +FP )if float (TN +FP )>0 else 0.0
    f1_or_dsc =float (2 *TP )/float (2 *TP +FP +FN )if float (2 *TP +FP +FN )>0 else 0.0
    miou =float (TP )/float (TP +FP +FN )if float (TP +FP +FN )>0 else 0.0
    precision =float (TP )/float (TP +FP )if float (TP +FP )>0 else 0.0
    return (accuracy ,sensitivity ,specificity ,f1_or_dsc ,miou ,precision ,sensitivity ,cm )

def train_one_epoch (train_loader ,model ,criterion ,optimizer ,scheduler ,epoch ,step ,logger ,config ,writer ,grad_clip_norm =None ):
    model .train ()
    loss_list =[]
    for iter ,data in enumerate (train_loader ):
        step +=1
        optimizer .zero_grad ()
        images ,targets =data
        images ,targets =(images .cuda (non_blocking =True ).float (),targets .cuda (non_blocking =True ).float ())
        output =model (images )
        if isinstance (output ,tuple ):
            out ,ds_features =output
            main_loss =criterion (out ,targets )
            if hasattr (model .vmunet ,'deep_supervision'):
                ds_predictions =model .vmunet .deep_supervision (ds_features ,target_size =targets .shape [2 :])
                ds_loss ,ds_components =model .vmunet .deep_supervision .compute_loss (ds_predictions ,targets ,criterion ,epoch =epoch ,total_epochs =config .epochs )
                ds_w =getattr (config ,'deep_supervision_weight',0.1 )
                loss =main_loss +ds_w *ds_loss
            else :
                loss =main_loss
        else :
            out =output
            loss =criterion (out ,targets )
        loss .backward ()
        _clip =grad_clip_norm if grad_clip_norm is not None else getattr (config ,'grad_clip_norm',0 )
        if _clip is not None and float (_clip )>0 :
            torch .nn .utils .clip_grad_norm_ (model .parameters (),max_norm =float (_clip ))
        optimizer .step ()
        loss_list .append (loss .item ())
        now_lr =optimizer .state_dict ()['param_groups'][0 ]['lr']
        writer .add_scalar ('loss',loss ,global_step =step )
        if iter %config .print_interval ==0 :
            log_info =f'train: epoch {epoch }, iter:{iter }, loss: {np .mean (loss_list ):.4f}, lr: {now_lr }'
            print (log_info )
            logger .info (log_info )
    if scheduler is not None :
        scheduler .step ()
    return step

def val_one_epoch (test_loader ,model ,criterion ,epoch ,logger ,config ):
    model .eval ()
    preds =[]
    gts =[]
    loss_list =[]
    with torch .no_grad ():
        for data in tqdm (test_loader ):
            img ,msk =data
            img ,msk =(img .cuda (non_blocking =True ).float (),msk .cuda (non_blocking =True ).float ())
            output =model (img )
            if isinstance (output ,tuple ):
                out =output [0 ]
            else :
                out =output
            loss =criterion (out ,msk )
            loss_list .append (loss .item ())
            gts .append (msk .squeeze (1 ).cpu ().detach ().numpy ())
            if out .shape [1 ]==1 :
                prob =torch .sigmoid (out ).squeeze (1 )
            else :
                prob =torch .softmax (out ,dim =1 )
                prob =prob [:,1 ,:,:]
            preds .append (prob .cpu ().detach ().numpy ())
    preds =np .array (preds ).reshape (-1 )
    gts =np .array (gts ).reshape (-1 )
    accuracy ,sensitivity ,specificity ,f1_or_dsc ,miou ,precision ,recall ,confusion =_binary_metrics_from_logits (preds ,gts ,config .threshold )
    log_info =f'val epoch: {epoch }, loss: {np .mean (loss_list ):.4f}, miou: {miou }, f1_or_dsc: {f1_or_dsc }, accuracy: {accuracy },             specificity: {specificity }, sensitivity: {sensitivity }, confusion_matrix: {confusion }'
    print (log_info )
    logger .info (log_info )
    return (np .mean (loss_list ),miou )

def test_one_epoch (test_loader ,model ,criterion ,logger ,config ,test_data_name =None ,eval_split ='test'):
    model .eval ()
    preds =[]
    gts =[]
    loss_list =[]
    with torch .no_grad ():
        for i ,data in enumerate (tqdm (test_loader )):
            img ,msk =data
            img ,msk =(img .cuda (non_blocking =True ).float (),msk .cuda (non_blocking =True ).float ())
            output =model (img )
            if isinstance (output ,tuple ):
                out =output [0 ]
            else :
                out =output
            loss =criterion (out ,msk )
            loss_list .append (loss .item ())
            msk =msk .squeeze (1 ).cpu ().detach ().numpy ()
            gts .append (msk )
            if out .shape [1 ]==1 :
                prob =torch .sigmoid (out ).squeeze (1 )
            else :
                prob =torch .softmax (out ,dim =1 )[:,1 ,:,:]
            out =prob .cpu ().detach ().numpy ()
            preds .append (out )
            if i %config .save_interval ==0 :
                save_imgs (img ,msk ,out ,i ,config .work_dir +'outputs/',config .datasets ,config .threshold ,test_data_name =test_data_name )
        preds =np .array (preds ).reshape (-1 )
        gts =np .array (gts ).reshape (-1 )
        accuracy ,sensitivity ,specificity ,f1_or_dsc ,miou ,_ ,_ ,confusion =_binary_metrics_from_logits (preds ,gts ,config .threshold )
        if test_data_name is not None :
            log_info =f'dataset_name: {test_data_name }'
            print (log_info )
            logger .info (log_info )
        log_info =f'eval_split={eval_split } (best checkpoint), loss: {np .mean (loss_list ):.4f}, miou: {miou }, f1_or_dsc: {f1_or_dsc }, accuracy: {accuracy }, specificity: {specificity }, sensitivity: {sensitivity }, confusion_matrix: {confusion }'
        print (log_info )
        logger .info (log_info )
    return np .mean (loss_list )
