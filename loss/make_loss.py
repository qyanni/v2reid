import torch.nn.functional as F
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):
   # sampler = cfg.DATALOADER.SAMPLER
    model_name =  cfg.MODEL.NAME

    feat_dim = 0

    # can be automated
    if 'd1' in model_name:
        feat_dim = 384
    elif 'd2'in model_name:
        feat_dim = 512
    elif 'd3'in model_name:
        feat_dim = 512
    elif 'd4'in model_name:
        feat_dim = 768
    elif 'd5'in model_name:
        feat_dim = 768
    elif 'yan'in model_name:
        feat_dim = 768
    else:
        print('expected model name to be d1, d2, d3, d4 or d5'
                'but got {}'.format(model_name))
   
    
    device = 'cuda:' + cfg.MODEL.DEVICE_ID

    make_loss.update_iter_interval = 500
    make_loss.id_loss_history = []
    make_loss.metric_loss_history = []
    make_loss.ID_LOSS_WEIGHT = cfg.LOSS.ID_LOSS_WEIGHT
    make_loss.TRIPLET_LOSS_WEIGHT = cfg.LOSS.TRIPLET_LOSS_WEIGHT
    

    # define center loss, and condition: if no center loss then set weight as 0
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True, device=device)  # center loss
    if  cfg.LOSS.CENTER_LOSS:
        make_loss.CENTER_LOSS_WEIGHT = cfg.LOSS.CENTER_LOSS_WEIGHT 
    else:
        make_loss.CENTER_LOSS_WEIGHT = 0

    id_loss_func = F.cross_entropy

    # if softmax sampler then normal dataloader and normal softmax/CE los    
    if not cfg.LOSS.TRIPLET_LOSS:
        print('Training with ID loss')
        def loss_func(score, feat, target):
            return make_loss.ID_LOSS_WEIGHT *id_loss_func(score, target) +\
                make_loss.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

    else:
        # define metric loss function -> here we combine them directly: CE + metric
        if cfg.LOSS.TRIPLET_LOSS:
            metric_loss_func = TripletLoss(cfg.LOSS.TRIPLET_LOSS_MARGIN) 


        if  cfg.LOSS.CENTER_LOSS:
            print('Training with ID loss, triplet loss and center loss')    
        else:
            print('Training with ID loss and triplet loss')    

        def loss_func(score, feat, target):
            return make_loss.ID_LOSS_WEIGHT * id_loss_func(score, target) + \
                    make_loss.TRIPLET_LOSS_WEIGHT * metric_loss_func(feat, target) +\
                    make_loss.CENTER_LOSS_WEIGHT * center_criterion(feat, target)


    if cfg.LOSS.CENTER_LOSS:
        return loss_func, center_criterion
    else:
        return loss_func



