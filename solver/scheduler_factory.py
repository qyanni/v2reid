""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman

Cosine: https://timm.fast.ai/SGDR
Step :  https://timm.fast.ai/stepLR#decay_rate

"""
from .cosine_lr import CosineLRScheduler
from .step_lr import StepLRScheduler

def create_scheduler(cfg, optimizer):

    # type 1
    # lr_min = 0.01 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR

    # type 2 -> kept from TransReID
    lr_min = 0.002 * cfg.SOLVER.BASE_LR
    warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

    #print('wcfg WARMUP_LR ', cfg.SOLVER.WARMUP_LR)
    # type 3
    # lr_min = 0.001 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

    decay_t = cfg.SOLVER.DECAY_T 
    decay_r = cfg.SOLVER.DECAY_RATE
    warmup_t = cfg.SOLVER.WARMUP_EPOCHS
    k_decay = cfg.SOLVER.K_DECAY
    t_mul = cfg.SOLVER.CYCLE_MUL
    cycle_lim = cfg.SOLVER.CYCLE_LIM
    warmup_prefix = cfg.SOLVER.WARMUP_PRE


    noise_range = None

    lr_scheduler = None

    if cfg.SOLVER.SCHED == 'cosine':
        lr_scheduler = CosineLRScheduler(
                optimizer,                      # optimizer used for the training process, from create_optimizer
                t_initial=decay_t,              # number of epochs for scheduler
                lr_min=lr_min,                  # the minimum LR used during the scheduling, LR does not ever go below this value
                cycle_mul= t_mul,               # the number of iterations (epochs) in the i-th decay cycle
                cycle_decay=decay_r,            # at every restart the learning rate is decayed by new learning rate which equals lr * decay_rate
                warmup_lr_init=warmup_lr_init,  #  initial learning rate during warmup
                warmup_t=warmup_t,              # number of warmup epochs
                cycle_limit= cycle_lim,         # the number of maximum restarts in SGDR
                noise_range_t=noise_range,
                warmup_prefix=warmup_prefix,    # Defaults to False. If set to True, then every new epoch number equals epoch = epoch - warmup_t
                noise_pct= 0.67,
                noise_std= 1.,
                noise_seed=42,
                k_decay= k_decay
            )


    elif cfg.SOLVER.SCHED == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_t,
            decay_rate=decay_r,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )
    return lr_scheduler
