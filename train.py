from config import cfg
from utils.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
from datasets import make_dataloader
from models.volo import *
from timm.models import create_model
from utils import load_pretrained_weights
from loss import make_loss
from solver.scheduler_factory import create_scheduler
from solver import make_optimizer, make_optimizer_with_center
from tools import train_without_center, train_with_center

import numpy as np
import os
import argparse
import torch
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    # parse arguments in command in terminal
    parser = argparse.ArgumentParser(description="VreID using VOLO Training")
    # add argument in terminal
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    ### >>>> you can add more args here

    # parse through args in terminal
    args = parser.parse_args()

    #if mentioned config_file is not empty, merge cfg file with added file
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    #set seed
    set_seed(cfg.SOLVER.SEED)

    #get output directory and create
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("created directory ", output_dir)
    
    #set up a logger 
    logger = setup_logger("volo-vreid", output_dir)
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))
    logger.info("Saving model in the path: {}".format(cfg.OUTPUT_DIR))

    #set up for tensorboard
    out_name = cfg.OUTPUT_DIR
    writer = SummaryWriter(comment=out_name.split("/")[1])

    #define device
    if torch.cuda.is_available():
        model_device_ID = cfg.MODEL.DEVICE_ID
        nb_GPU = len(model_device_ID)
        logger.info('Training with a single process on {} GPU(s)'.format(nb_GPU))
        device = 'cuda:' + model_device_ID
        logger.info('Device ID is {}'.format(device))
    else:
        device = 'cpu'
        logger.info('No GPU, training on {}'.format(device))

    #load datasets
    train_loader, num_classes,val_loader ,num_query = make_dataloader(cfg)

    #model creation
    model = create_model(
        cfg.MODEL.NAME,                             # model name
        num_classes = num_classes,                  # number of classes
        pretrained= cfg.MODEL.PRETRAINED,           # pretained T/F
        img_size = cfg.INPUT.SIZE_TRAIN[0],         # input size
        overlap = cfg.MODEL.OVERLAP,               # overlapping patches T/F
        drop_rate = cfg.MODEL.DROP_OUT,             # drop out rates
        attn_drop_rate = cfg.MODEL.ATT_DROP_RATE,
        drop_path_rate = cfg.MODEL.DROP_PATH ,
        neck = cfg.MODEL.NECK                       # use bnneck, lnneck or none
        )

    # load pretrained model
    if cfg.MODEL.PRETRAIN_PATH:
        load_pretrained_weights(
            model=model,
            checkpoint_path=cfg.MODEL.PRETRAIN_PATH,
            strict = False, 
            num_classes=num_classes)

    #print(model) # >>>>> if you want to see all components in model

    model.to(device) # move model to GPU

    # setup loss function  
    if not cfg.LOSS.CENTER_LOSS :        # without centre loss
        optimizer = make_optimizer(cfg, model)
        loss_func = make_loss(cfg, num_classes)  
        scheduler = create_scheduler(cfg, optimizer)

        train_without_center(writer,    # tensorboard writer 
            cfg,            # config file
            model,          # model
            train_loader,   # training dataset
            val_loader,     # query + gallery
            optimizer,      # optimizer
            loss_func,      # training loss 
            scheduler,      # LR scheduler
            num_query,      # number of queries
        )

    else:     # with center loss
        loss_func, center_criterion = make_loss(cfg, num_classes)  
        optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)
        scheduler = create_scheduler(cfg, optimizer)
    
        train_with_center(writer,    
            cfg,                
            model,            
            train_loader,    
            val_loader,       
            optimizer,    
            loss_func,        
            center_criterion,   
            optimizer_center,
            scheduler,         
            num_query,    
        )


if __name__ == '__main__':
    main()