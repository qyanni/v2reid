# encoding: utf-8
"""
From: https://github.com/damo-cv/TransReID/blob/main/solver/make_optimizer.py

differnence between state_dict and named_parameters():
https://stackoverflow.com/questions/54746829/pytorch-whats-the-difference-between-state-dict-and-parameters

"""

import torch



def make_optimizer(cfg, model):

    """ Creates an optimizer.
    Args:
        model (nn.Module): model containing parameters to optimize
    Returns:
        Optimizer
    """

    params = []
    for key, value in model.named_parameters():

        # requires_grad is a flag that controls whether a tensor requires a gradient or not.
        if not value.requires_grad:
            continue

        lr = cfg.SOLVER.BASE_LR                      # initial learning rate
        weight_decay = cfg.SOLVER.WEIGHT_DECAY       # weight decay to apply in optimizer
        optimizer_name = cfg.SOLVER.OPTIMIZER_NAME   # name of optimizer to create
        momentum = cfg.SOLVER.MOMENTUM               # momentum for momentum based optimizers (others may use betas via kwargs)
        bias_lr_factor = cfg.SOLVER.BIAS_LR_FACTOR   # multiplier of lr for bias parameters

        if "bias" in key:
            lr = lr * bias_lr_factor
            weight_decay = weight_decay
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = lr * 2
                print('Using two times learning rate for fc ')
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if optimizer_name == 'SGD': 
        optimizer = getattr(torch.optim, optimizer_name)(params, momentum=momentum)
    else:
        optimizer = getattr(torch.optim, optimizer_name)(params)
    return optimizer


def make_optimizer_with_center(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr = cfg.SOLVER.BASE_LR                      # initial learning rate
        weight_decay = cfg.SOLVER.WEIGHT_DECAY       # weight decay to apply in optimizer
        optimizer_name = cfg.SOLVER.OPTIMIZER_NAME   # name of optimizer to create
        momentum = cfg.SOLVER.MOMENTUM               # momentum for momentum based optimizers (others may use betas via kwargs)
        bias_lr_factor = cfg.SOLVER.BIAS_LR_FACTOR   # multiplier of lr for bias parameters

        if "bias" in key:
            lr = lr * bias_lr_factor
            weight_decay = weight_decay
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = lr * 2
                print('Using two times learning rate for fc ')
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if optimizer_name == 'SGD': 
        optimizer = getattr(torch.optim, optimizer_name)(params, momentum=momentum)
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    else:
        optimizer = getattr(torch.optim, optimizer_name)(params)
        optimizer_center = getattr(torch.optim, optimizer_name)(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center