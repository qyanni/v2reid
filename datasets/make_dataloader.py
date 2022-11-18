
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .sampler import RandomIdentitySampler
from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .veri import VeRi


"""
Load the training, query and gallery data;
Apply data transformation
"""

def train_collate_fn(batch):
    imgs, vids, camids, _  = zip(*batch)
    vids = torch.tensor(vids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), vids, camids

def val_collate_fn(batch):  
    imgs, vids, camids, _ = zip(*batch)
    vids = torch.tensor(vids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), vids, camids

def make_dataloader(cfg):
    if cfg.INPUT.RESIZECROP == True:
        randomcrop = T.RandomResizedCrop(cfg.INPUT.SIZE_TRAIN,scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3)
    else:
        randomcrop = T.RandomCrop(cfg.INPUT.SIZE_TRAIN)
    # train transforms
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),    # resize                                   
        T.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB),      # horizontal flip
        T.RandomPerspective(distortion_scale=0.1, p = cfg.INPUT.DISTORTION_PROB),   # distortion
        T.Pad(cfg.INPUT.PADDING, padding_mode='constant'),  # padding
        randomcrop,                                         # random crop
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)], p=cfg.INPUT.COLORJIT_PROB),  # change contrast, hue, etc
        T.ToTensor(),                                       # to tensor
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),    # normalize
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')   # random erasing
    ])
    # validation transforms
    val_test_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3), # resize        
        T.ToTensor(),                                   # to tensor
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)  # normalize
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = VeRi(root=cfg.INPUT.ROOT_DIR)

    # get train set use different sampler depending on cfg/loss
    train_dataset = ImageDataset(dataset.train, train_transforms)
    num_classes_train = dataset.num_train_vids

    if cfg.LOSS.TRIPLET_LOSS:
        print('Datasampling using triplet sampler')
        train_loader = DataLoader(
            train_dataset, 
            batch_size = cfg.DATALOADER.IMS_PER_BATCH, 
            sampler = RandomIdentitySampler(dataset.train, cfg.DATALOADER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers = num_workers, 
            collate_fn = train_collate_fn
        )
    else:
        print('Datasampling using softmax sampler')
        train_loader = DataLoader(
            train_dataset, 
            batch_size = cfg.DATALOADER.IMS_PER_BATCH, 
            shuffle = True,
            num_workers = num_workers,
            collate_fn = train_collate_fn
        )

    # get validation dataset -> query + gallery with val transforms
    val_set = ImageDataset(dataset.query + dataset.gallery, val_test_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.DATALOADER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, num_classes_train, val_loader, len(dataset.query)



