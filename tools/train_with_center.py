import os
import time
import torch
from utils.meter import AverageMeter
import logging
import torchvision
from torch.cuda import amp
from timm.utils import *
from utils.metrics import R1_mAP_eval

def train_with_center(
        writer,   
        cfg,
        model,
        loader,
        val_loader,
        optimizer,
        loss_fn,
        center_criterion,
        optimizer_center,
        scheduler,
        num_query
    ):

    # prepare logger
    logger = logging.getLogger("volo-vreid")
    logger.info('start training')

    # get logs and eval period args
    log_period = cfg.SOLVER.LOG_PERIOD

    # set device
    if torch.cuda.is_available():
        model_device_ID = cfg.MODEL.DEVICE_ID
        nb_GPU = len(model_device_ID)
        logger.info('Training with a single process on {} GPU(s)'.format(nb_GPU))
        device = 'cuda:' + model_device_ID
        logger.info('Device ID is {}'.format(device))
    else:
        device = 'cpu'
        logger.info('No GPU, training on {}'.format(device))


    # Initialize a meter to record loss, batch time, accuracy
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()   
    acc_m = AverageMeter()

    # initialize evaluator
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=True)     

    # for more info, check https://wandb.ai/wandb_fc/tips/reports/How-to-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5
    scaler = amp.GradScaler() 
    
    # initialize vars to save best .pths
    mAP_prev = 0
    mAP_old_file_n = None

    R1_prev = 0 
    R1_old_file_n = None

    R5_prev = 0 
    R5_old_file_n = None

    R10_prev = 0 
    R10_old_file_n = None

    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS+1):
        start_time = time.time()
        losses_m.reset()
        acc_m.reset()
        evaluator.reset()
        scheduler.step(epoch)	 # All optimizers inherit from a common parent class torch.nn.Optimizer and are updated using the step method implemented for each of them.
        model.train()
    
        last_idx = len(loader) - 1 
        num_updates = epoch * len(loader)

        # Iterate over data and get a batch of inputs
        for batch_idx, (img, vid, target_cam) in enumerate(loader):

            last_batch = batch_idx == last_idx
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                img, target = img.to(device), vid.to(device)
                model.to(device)
            #instead of setting to zero, set the grads to None. This will in general have lower memory footprint, and can modestly improve performance.
            # Clear the gradients
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            #-------- forward --------
            with amp.autocast(enabled=True):
                # Forward Pass
                score, feat = model(img) #score = x_cls
                # Compute Loss
                loss = loss_fn(score, feat, target)
        #--------  backward + optimize --------
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()
            # Update Weights
            # scaler.step() first unscales the gradients of the optimizer's assigned params
            scaler.step(optimizer)
            # then Updates the scale for next iteration
            scaler.update()

            for param in center_criterion.parameters():
                param.grad.data *= (1. / cfg.LOSS.CENTER_LOSS_WEIGHT)
            scaler.step(optimizer_center)
            scaler.update()
                
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()
                
            # Calculate Loss
            losses_m.update(loss.item(), img.size(0))#img.shape[0])
            acc_m.update(acc, 1)

            torch.cuda.synchronize()
            num_updates += 1
            batch_time_m.update(time.time() - start_time)

            torch.cuda.synchronize()
            if last_batch or batch_idx % log_period == 0:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.3e}, Avg Lr: {:.3e}"
                                .format(epoch, (batch_idx + 1), len(loader),
                                        losses_m.avg, acc_m.avg, optimizer.param_groups[0]['lr'], lr))

        writer.add_scalar('LR per epoch',round(optimizer.param_groups[0]['lr'],4), epoch) 
        writer.add_scalars('Loss and Acc / train per epoch', {'Train loss': loss , 'Accuracy': acc}, epoch) 
        
        # Evaluation step
        model.eval()
        for n_iter, (img, vid, camid) in enumerate(val_loader):
            with torch.no_grad():
                img = img.to(device)
                _, feat = model(img)
                evaluator.update((feat, vid, camid))
        
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info("Validation Results - Epoch: {}".format(epoch))
        logger.info("mAP: {:.1%}".format(mAP))

        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

        writer.add_scalars('mAP and CMC per epoch', 
            {'mAP': mAP ,
            'rank-1': cmc[0],
            'rank-5': cmc[4],
            'rank-10': cmc[9],
            }, epoch
        ) 

        torch.cuda.empty_cache()

        # save the best model by comparing with previously saved
        ### >>  can be cleaner...

        if mAP > mAP_prev:
            mAP_prev = mAP
            met  =  str(mAP_prev).split('.')[1][:4]
            file_n = os.path.join(cfg.OUTPUT_DIR, 'mAP_{0}_epoch_{1}.pth'.format(met, epoch))
            if mAP_old_file_n:
                if os.path.exists(mAP_old_file_n):
                    os.remove(mAP_old_file_n)
            torch.save(model.state_dict(),file_n)  
            mAP_old_file_n = file_n

       
        if cmc[0] > R1_prev:
            R1_prev = cmc[0]
            met  =  str(R1_prev).split('.')[1][:4]
            file_n = os.path.join(cfg.OUTPUT_DIR, 'R1_{0}_epoch_{1}.pth'.format(met, epoch))
            torch.save(model.state_dict(),file_n)  
            if R1_old_file_n:
                if os.path.exists(R1_old_file_n):
                    os.remove(R1_old_file_n)
            torch.save(model.state_dict(),file_n)  
            R1_old_file_n = file_n
  

        if cmc[4] > R5_prev:
            R5_prev = cmc[4]
            met  =  str(R5_prev).split('.')[1][:4]
            file_n = os.path.join(cfg.OUTPUT_DIR, 'R5_{0}_epoch_{1}.pth'.format(met, epoch))     
            torch.save(model.state_dict(),file_n)  
            if R5_old_file_n:
                if os.path.exists(R5_old_file_n):
                    os.remove(R5_old_file_n)
            torch.save(model.state_dict(),file_n)  
            R5_old_file_n = file_n


        if cmc[9] > R10_prev:
            R10_prev = cmc[9]
            met  =  str(R10_prev).split('.')[1][:4]
            file_n = os.path.join(cfg.OUTPUT_DIR, 'R10_{0}_epoch_{1}.pth'.format(met, epoch))
            torch.save(model.state_dict(),file_n)  
            if R10_old_file_n:
                if os.path.exists(R10_old_file_n):
                    os.remove(R10_old_file_n)
            torch.save(model.state_dict(),file_n)  
            R10_old_file_n = file_n

