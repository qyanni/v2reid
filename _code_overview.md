## Overview

This is an overview of the codes for V2ReID. The codes find their inspiration from [VOLO](https://github.com/sail-sg/volo) and [TransReID](https://github.com/damo-cv/TransReID).
Each _.py_ file contains further descriptions. Please do not hesitate to contact us if have any questions or suggestions! :blush:

[datasets directory](https://github.com/qyanni/v2reid/tree/main/datasets) contains modules related to reading and sampling the input datasets. 
Please refer to the configs under `DATALOADER` and `INPUT` to choose the the desired parameters:

- [bases.py](https://github.com/qyanni/v2reid/blob/main/datasets/bases.py) retrieves the information of the ReID dataset (train, query and gallery), such as number of vehicle IDs, number of images, number of camera IDs (not used in training), and the image paths for data loading.
- [make_dataloader.py](https://github.com/qyanni/v2reid/blob/main/datasets/make_dataloader.py) loads the train, query and gallery data and applies data augmentation such as resizing, padding, flips, etc. 
- [sampler.py](https://github.com/qyanni/v2reid/blob/main/datasets/sampler.py) implements the random identity sampler, that's used when training with the triplet loss. The data needs to be sampled in a specific way, such that we have _k_ instances for each _N_ vehicle identity per batch.
- [veri.py](https://github.com/qyanni/v2reid/blob/main/datasets/veri.py) specifically reads VeRi-776. For other datasets, e.g. VehicleID, you'll need to modify it.


[loss directory](https://github.com/qyanni/v2reid/tree/main/loss) contains modules related to the different losses (ID, triplet and center). Please refer to the configs under `LOSS` to choose the desired parameters. 
- [center_loss.py](https://github.com/qyanni/v2reid/blob/main/loss/center_loss.py) implements the centre loss.
- [make_loss.py](https://github.com/qyanni/v2reid/blob/main/loss/make_loss.py) builds the loss.
- [triplet_loss.py](https://github.com/qyanni/v2reid/blob/main/loss/triplet_loss.py) implements the triplet loss.

The total loss is formulated as

$$ \text{total loss} = \lambda_{ID} \times \mathcal{L_{\text{ID}}} + \lambda_{tri} \times \mathcal{L_{\text{tri}}} + \lambda_{cen} \times \mathcal{L_{\text{cen}}} $$

where $\mathcal{L_{\text{(.)}}}$ is the loss and $\lambda_{(.)}$ the associated weight.
Use the following configs if you want to train:
- using the ID loss only: set `TIRPLET_LOSS` and `CENTER_LOSS` to `False`
- using the ID and triplet loss: set `TIRPLET_LOSS` to `True` and `CENTER_LOSS` to `False`
- using the three losses: set `TIRPLET_LOSS` and `CENTER_LOSS` to `True`
The weights can be adjusted with `ID_LOSS_WEIGHT`, `TRIPLET_LOSS_WEIGHT` and `CENTER_LOSS_WEIGHT`.


[models directory](https://github.com/qyanni/v2reid/tree/main/models) contains modules related to the backbone architecture (VOLO). Please refer to the configs under `MODEL`.

-	[volo.py](https://github.com/qyanni/v2reid/blob/main/models/volo.py): implementation of VOLO. We added a snippet for overlapping patches, the batch normalization/layer normalization neck and how to create your own model with your own settings.


[solver directory](https://github.com/qyanni/v2reid/tree/main/solver) contains modules related to the optimization and optimization scheduler (i.e. learning rate scheduler). Please refer to the configs under `SOLVER`.

-	[cosine_lr.py](https://github.com/qyanni/v2reid/blob/main/solver/cosine_lr.py) implementats the cosine LR scheduler
-	[make_optimizer.py](https://github.com/qyanni/v2reid/blob/main/solver/make_optimizer.py) creates the optimizer
-	[scheduler_factory.py](https://github.com/qyanni/v2reid/blob/main/solver/scheduler.py) creates the LR scheduler (cosine or step)
-	[scheduler.py](https://github.com/qyanni/v2reid/blob/main/solver/scheduler_factory.py) base class used to schedule optimizer parameter groups
-	[step_lr.py](https://github.com/qyanni/v2reid/blob/main/solver/step_lr.py) implements the step LR scheduler

[tools directory](https://github.com/qyanni/v2reid/tree/main/tools) contains the training options (with and without center loss)
-	[train_with_center.py](https://github.com/qyanni/v2reid/blob/main/tools/train_with_center.py): training with center loss
-	[train_without_center.py](https://github.com/qyanni/v2reid/blob/main/tools/train_without_center.py): training without center loss


[utils directory](https://github.com/qyanni/v2reid/tree/main/utils) contains modules related to the evaluation and logging the process.
-	[logger.py](https://github.com/qyanni/v2reid/blob/main/utils/logger.py): set up a logger
-	[meter.py](https://github.com/qyanni/v2reid/blob/main/utils/meter.py): computes and stores a meter
-	[metrics.py](https://github.com/qyanni/v2reid/blob/main/utils/metrics.py): computes the mAP and CMC
-	[utils.py](https://github.com/qyanni/v2reid/blob/main/utils/utils.py): for loading the pretrained weights

