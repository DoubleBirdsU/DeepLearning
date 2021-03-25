import os
import torch

from torch.utils import data as tud
from base.datasets import LoadImageAndLabels


def dataLoader(
        data_path,
        img_size=416,
        batch_size=16,
        augment=True,
        hyper=None,
        rect=False,
        cache_images=False,
        single_cls=False,
        num_workers=16,
        pin_memory=True
):
    r"""dataLoader

    Args:
        data_path:
        img_size: (=416)
        batch_size: (=16)
        augment: (=False)
        hyper: (=None)
        rect: (=False)
        cache_images: (=False)
        single_cls: (=False)
        num_workers: (=16)
        pin_memory: (=False)

    Returns:
        None
    """

    class TransData(LoadImageAndLabels):
        def __init__(self, *argv, **kwargs):
            super(TransData, self).__init__(*argv, **kwargs)

        @staticmethod
        def collate_fn(batch):
            imgs, labels, path, shapes, index = zip(*batch)  # transposed
            for i, label in enumerate(labels):
                label[:, 0] = i  # add target image index for build_targets()
            return torch.stack(imgs, 0).type(torch.FloatTensor), torch.cat(labels, 0)

    dataset = TransData(
        data_path, img_size, batch_size, augment=augment,
        hyp=hyper, rect=rect, cache_images=cache_images, single_cls=single_cls)

    data_loader = tud.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=not rect, pin_memory=pin_memory, collate_fn=dataset.collate_fn)

    return data_loader


def get_img_size(dataset_name):
    """get_img_size

    Args:
        dataset_name (str):

    Returns:
        size (num, num_col, num_row)
    """
    image_size = (1, 28, 28)
    dataset_name = dataset_name.lower()
    if 'mnist' == dataset_name:
        image_size = (1, 28, 28)
    elif 'cifar10' == dataset_name:
        image_size = (3, 32, 32)
    return image_size


# 导入模型
def load_model(net, net_name=None):
    if net_name is None:
        net_name = net.__class__.__name__
    model_save_path = net_name  # __generateModelName(net_name)
    is_exist = os.path.exists(model_save_path + '.index')
    if is_exist:
        net.load(model_save_path)
    return is_exist
