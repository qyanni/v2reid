from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os.path as osp

ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    Here we read the vehicle id (vid) and camera id (camid)
    """
    def get_imagedata_info(self, data):
        vids, cams = [], []
        for _, vid, camid in data:
            vids += [vid]
            cams += [camid]
        vids = set(vids)
        cams = set(cams)
        num_vids = len(vids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_vids, num_imgs, num_cams  #number of vehicle IDs, number of images, number of cameras

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Print datasets stats for train, query and gallery
    Number of vehicle IDs, number of training images and number of cameras
    """
    def print_dataset_statistics(self, train, query, gallery):
        num_train_vids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_vids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_vids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_vids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_vids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_vids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, vid, camid  = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, vid, camid ,img_path.split('/')[-1]