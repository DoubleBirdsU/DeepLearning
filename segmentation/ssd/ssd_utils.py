import os
import pickle
import torch
import torch.nn as nn

from torch.utils import data as tud
from base.datasets import LoadImageAndLabels
from base.model import ModelCheckpoint as MCP


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


# 导入参数
def load_weights(net, file_weight, mode='weight'):
    """load_weights

    Args:
        net (nn.Module):
        file_weight (str):
        mode:
    """
    is_exist = os.path.exists(file_weight)
    if is_exist:
        if 'weight' == mode:
            net.load_state_dict(torch.load(file_weight))
    return is_exist


# 断点续训
def load_breakpoint(net, data_name=None, weights_root='./weights', save_weights_only=True,
                    save_best_only=True, check_ckpt=False, pickle_module=pickle):
    """load_breakpoint

    Args:
        net:
        data_name (str):
        weights_root:
        save_weights_only:
        save_best_only:
        check_ckpt:
        pickle_module:

    Returns:
        None
    """
    if data_name is None:
        data_name = 'dataset'

    if not os.path.exists(weights_root):
        os.mkdir(weights_root)
    param_path = os.path.join(weights_root, data_name.upper())
    if not os.path.exists(param_path):
        os.mkdir(param_path)

    mode = 'weight' if save_weights_only else 'model'
    filename = os.path.join(param_path, f'{data_name}_{net.__class__.__name__}_{mode}.pt')
    ckpt = MCP.ckpt_read(param_path)
    weight_filename = filename
    if ckpt is not None:
        if check_ckpt and not os.path.exists(weight_filename) and 'filename' in ckpt:
            weight_filename = ckpt['filename']
        weight_filename += ckpt['suffix_best'] if 'suffix_best' in ckpt else '.best'
    if load_weights(net, weight_filename, mode):
        print('----------------load the weight----------------')

    return MCP(filename, save_weights_only, save_best_only, pickle_module=pickle_module)
