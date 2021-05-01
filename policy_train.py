import os
import pickle
import dill
import numpy as np

import torch
import torch.nn as nn
import yaml

from base.model import ModelCheckpoint as MCp, NNet
from base.utils import check_dirs
from chess_manual.manualloader import DataLoader
from segmentation.yolo.yolo_blocks import create_yolo_blocks


def load_breakpoint(net, data_name='',
                    class_name='',
                    weights_root='./weights',
                    save_weights_only=True,
                    save_best_only=True,
                    check_ckpt=False,
                    map_location=None,
                    pickle_module=pickle,
                    rm_saved_net=False):
    """
    Args:
        net:
        data_name (str):
        class_name (str):
        weights_root:
        save_weights_only:
        save_best_only:
        check_ckpt:
        map_location:
        pickle_module:
        rm_saved_net:
    """
    if data_name is None:
        data_name = 'dataset'

    mode = 'weight' if save_weights_only else 'model'
    dir_path = check_dirs([data_name, class_name, mode], dir_root=os.path.abspath(weights_root))

    filename = os.path.join(dir_path, f'{data_name}_{net.__class__.__name__}_{mode}.pt')
    ckpt = MCp.ckpt_read(dir_path, ckpt_name=f'ckpt.yaml')
    ckpt_file_name = filename
    if check_ckpt and not os.path.exists(ckpt_file_name) and 'filename' in ckpt:
        ckpt_file_name = ckpt['filename']
    ckpt_file = ckpt_file_name + (ckpt['suffix_best'] if 'suffix_best' in ckpt else '')
    if net.load_weights(ckpt_file, mode=mode, map_location=map_location, pickle_module=pickle_module):
        print('----------------load the weight----------------')

    return MCp(filename, save_weights_only, save_best_only, pickle_module=pickle_module)


def target_fn(targets):
    device = targets.device
    winners_z = torch.ones_like(targets)
    targets = targets - 225
    winners_z[targets < 0] = -1.0
    winners_z[targets > 0] = 1.0
    winners_z.unsqueeze_(-1)
    targets = torch.abs_(targets) - 1
    return [targets.to(device=device), winners_z.to(dtype=torch.float,device=device)]


class PolicyNetLoss(nn.Module):
    """PolicyNetLoss

        PolicyNet 的损失函数

    Example:

            loss = PolicyNetLoss()
            pnl_loss = loss([act_pred, state_pred], y_true)
    """

    def __init__(self, weight=(1.0, 1.0), target_transform=None):
        super(PolicyNetLoss, self).__init__()
        if not isinstance(weight, np.ndarray):
            weight = np.array(weight)
        self.weight = weight / weight.prod()
        self.target_transform = target_transform
        self.act_loss = nn.NLLLoss()
        self.state_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        y_true[1] = y_true[1].unsqueeze_(-1).to(y_pred[1].dtype)
        return self.act_loss(y_pred[0], y_true[0]) + self.state_loss(y_pred[1], y_true[1])


def net_train(epochs=600, data_root_dir='~/.dataset/data_paper/'):
    layers_fn_dict = {
        'create_yolo_blocks': create_yolo_blocks,
    }

    batch_size = 128
    device = torch.device('cuda')

    # 导入数据
    dataLoader = DataLoader(data_root_dir, batch_size=batch_size, shuffle=True)
    train_loader = dataLoader['train']
    val_loader = dataLoader['val']

    # 创建网络
    with open('cfg/policy_net.yaml') as f:
        net_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        layers_fn = layers_fn_dict[net_cfg['layers_fn']] if 'layers_fn' in net_cfg else None
        net = NNet(net_cfg, layers_fn=layers_fn).to(device)

    # 断点续训
    dataset_name = train_loader.dataset.__class__.__name__
    class_name = net.__class__.__name__
    cp_callback = load_breakpoint(net, dataset_name, class_name, save_weights_only=True,
                                  map_location=device, pickle_module=dill)

    # 损失函数
    loss = PolicyNetLoss(target_transform=target_fn)
    net.compile(optimizer='adam',
                loss=loss,
                device=device,
                metrics=['acc'])

    # 训练
    net.fit_generator(train_loader,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=val_loader,
                      callbacks=[cp_callback])


if __name__ == '__main__':
    net_train(100, data_root_dir='./data/chess_manual/wuziqi')
