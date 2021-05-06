# coding=utf-8
import os
import pickle
import dill
import numpy as np

import torch
import torch.nn as nn
import yaml

from base.utils import check_dirs
from base.model import ModelCheckpoint as MCp, NNet
from chess_manual.chess_board import ChessBoard, ChessGame
from chess_manual.manualloader import DataLoader, PolicyNet
from chess_manual.mcts_alpha_zero import MCPlayer


def load_breakpoint(net, data_name='',
                    class_name='',
                    weights_root='./weights',
                    save_weights_only=True,
                    save_best_only=True,
                    check_ckpt=False,
                    map_location=None,
                    pickle_module=pickle,
                    rm_saved_net=False):
    """load_breakpoint
        断点续训

    Args:
        net: 网络
        data_name (str): 数据集模型
        class_name (str): 网络模型名称
        weights_root: 权重文件目录
        save_weights_only: 是否仅保存权重
        save_best_only: 是否仅保存最优权重
        check_ckpt: 是否检测配置
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


class PolicyNetLoss(nn.Module):
    """PolicyNetLoss

        PolicyNet 的损失函数

    Example:

            loss = PolicyNetLoss()
            pnl_loss = loss([act_pred, state_pred], y_true)
    """

    def __init__(self, weight=(1.0, 1.0)):
        super(PolicyNetLoss, self).__init__()
        if not isinstance(weight, np.ndarray):
            weight = np.array(weight)
        self.weight = weight / weight.prod()
        self.act_loss = nn.NLLLoss()
        self.state_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        y_true[1] = y_true[1].to(y_pred[1].dtype)
        return self.act_loss(y_pred[0], y_true[0]) + self.state_loss(y_pred[1], y_true[1])


def train(net, epochs=600, batch_size=256, data_root_dir='~/.dataset/data_paper/'):
    device = torch.device('cuda')
    net = net.to(device)

    # 导入数据
    dataLoader = DataLoader(data_root_dir, batch_size=batch_size, shuffle=True)
    train_loader = dataLoader['train']
    val_loader = dataLoader['val']

    # 断点续训
    dataset_name = 'WZQManuals'
    class_name = net.__class__.__name__
    cp_callback = load_breakpoint(net, dataset_name, class_name, save_weights_only=True,
                                  map_location=device, pickle_module=dill)

    # 损失函数
    loss = PolicyNetLoss()
    net.compile(optimizer='adam', loss=loss, metrics=['acc'], device=device, targets_fn=dataLoader.target_fn,
                data_fn=dataLoader.data_fn)

    # 训练
    net.fit_generator(train_loader,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=val_loader,
                      callbacks=[cp_callback])
    pass


def auto_play():
    num_roll_out = 256
    while True:
        cfg_net_one = 'cfg/policy_net_big.yaml'
        cfg_net_two = 'cfg/policy_net_big.yaml'
        policy_trainer, policy_player = PolicyNet(cfg_net_two, is_training=True), PolicyNet(cfg_net_one)
        mcts_trainer = MCPlayer(policy_trainer.policy_value_fn, num_roll_out=num_roll_out)
        mcts_player = MCPlayer(policy_player.policy_value_fn, num_roll_out=num_roll_out)
        trainer_game = ChessGame(ChessBoard(), mcts_trainer, mcts_player, buffer_size=100000)
        trainer_game.auto_play(13)
    pass


def train_net(net_cfg_file='cfg/policy_net_res.yaml', data_root_dir='~/.dataset/wuziqi/images'):
    # 创建网络
    with open(net_cfg_file) as f:
        net_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        net_trainer = NNet(net_cfg)

    train(net_trainer, epochs=100, batch_size=256, data_root_dir=data_root_dir)
    pass


if __name__ == '__main__':
    auto_play()
    # train_net()
    pass
