import os
import torch.utils.data as udata
import torchvision.transforms as transforms

from torchvision import datasets as ds
from torchvision.transforms import transforms


class DataLoader:
    def __init__(self,
                 data_root_dir,
                 batch_size=1,
                 num_workers=4,
                 shuffle=False,
                 is_resize=False,
                 resize=(64, 64),
                 is_gray=False,
                 mean=0.5,
                 std=0.5):
        transform_list = []
        keys = ['train', 'val']
        dataset_name = os.path.basename(data_root_dir)
        if is_resize:
            transform_list.append(transforms.RandomResizedCrop(resize))
        if is_gray:
            transform_list.append(transforms.Grayscale())
        transform_list.append(transforms.ToTensor())
        data_transforms = {
            'train': transforms.Compose(transform_list),
            'val': transforms.Compose(transform_list)
        }

        image_datasets = {
            key: ds.ImageFolder(os.path.join(data_root_dir, key), data_transforms[key]) for key in keys
        }

        for key in image_datasets:
            image_datasets[key].__class__.__name__ = dataset_name

        self.data_loaders = {
            key: udata.DataLoader(image_datasets[key], batch_size=batch_size,
                                  num_workers=num_workers, shuffle=shuffle) for key in keys
        }

        self.data_size = {x: len(image_datasets[x]) for x in keys}

    def __getitem__(self, item):
        return self.data_loaders[item]
