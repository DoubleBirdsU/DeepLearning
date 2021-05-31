# coding=utf-8
import os
import pickle
import random

import dill
import torch

import cv2
import numpy as np
import torch.utils.data as udata
import torchvision.transforms as transforms
import yaml
from torch import nn

from torchvision import datasets as ds
from torchvision.transforms import transforms

from base.model import NNet, ModelCheckpoint as MCp
from base.utils import check_dirs
from chess_manual.chess_utils import rot90board, node2flatten
from chess_manual.chessboard_define import cbd


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def create_state(chess_manual, value=255):
    img = np.zeros([15, 15, 4], dtype=np.uint8)
    for i, str_node in enumerate(chess_manual):
        node = [ord(str_node[0]) - ord('a'), 15 - int(str_node[1:])]
        img[node[0], node[1], i % 2 + 1] = value
        pass
    img[:, :, 3] = value
    return img


def create_player(idx):
    return ManualSectionCreator.BlackChess if idx % 2 == 0 else ManualSectionCreator.WhiteCHess


class DataLoader:
    def __init__(self,
                 data_root,
                 batch_size=128,
                 num_workers=4,
                 shuffle=True):
        batch_size = batch_size // 16
        transform_list = []
        keys = ['train', 'val']
        dataset_name = os.path.basename(data_root)
        transform_list.extend([
            transforms.ToTensor()
        ])
        data_transforms = {
            'train': transforms.Compose(transform_list),
            'val': transforms.Compose(transform_list)
        }

        image_datasets = {
            key: ds.ImageFolder(os.path.join(data_root, key), data_transforms[key],
                                loader=self.loader_fn) for key in keys
        }

        for key in image_datasets:
            image_datasets[key].__class__.__name__ = dataset_name

        self.data_loaders = {
            key: udata.DataLoader(image_datasets[key], batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers) for key in keys
        }

        self.data_size = {x: len(image_datasets[x]) for x in keys}
        pass

    def __getitem__(self, item):
        return self.data_loaders[item]

    @staticmethod
    def target_fn(targets, shape):
        count_targets = (shape[-1] // 15) * (shape[-2] // 15)
        device = targets.device
        winners_z = torch.ones_like(targets, dtype=torch.float)
        targets = targets - 225
        winners_z[targets < 0] = -1.0
        winners_z[targets > 0] = 1.0
        winners_z.unsqueeze_(-1)
        targets = torch.abs_(targets) - 1
        winners_z = torch.cat([winners_z for _ in range(count_targets)], dim=0)
        targets = torch.cat([targets for _ in range(count_targets)], dim=0)
        return [targets.to(device=device), winners_z.to(device=device)]

    @staticmethod
    def data_fn(data):
        shape = data.shape
        data_list = list()
        for i in range(shape[-1] // 15):
            row_s, row_e = i * 15, (i + 1) * 15
            row_imgs = data[:, :, :, row_s:row_e]
            data_list += [row_imgs[:, :, j * 15:(j + 1) * 15, :] for j in range(shape[-2] // 15)]
        return torch.cat(data_list, dim=0)

    @staticmethod
    def loader_fn(x):
        return cv2.imread(x, cv2.IMREAD_UNCHANGED)


class ManualSectionCreator:
    """ManualSectionCreator

        棋谱切片生成器
    """
    NoChess = 0
    BlackChess = 1
    WhiteCHess = -1

    def __init__(self, data_root,
                 obj_root,
                 batch_size=128,
                 board_size=(15, 15),
                 nlc_min=5,
                 buffer_size=2 ** 10,
                 is_extend=False,
                 is_random_cut=False,
                 mode='valid',
                 manual_step_max=None,
                 shuffle=True):
        self.data_root = data_root
        self.obj_root = obj_root
        self.batch_size = batch_size
        self.batch_section_size = batch_size
        self.board_size = board_size
        self.count_chess = board_size[0] * board_size[1]
        self.nlc_min = nlc_min
        self.buffer_size = buffer_size
        self.is_extend = is_extend
        self.is_random_cut = is_random_cut
        self.mode = mode
        self.manual_step_max = manual_step_max
        self.shuffle = shuffle
        self.boards = list()
        self.init_boards()

        self.data_buffer = list()
        self.file_list = self.walkFile(data_root, '.txt')
        self.idx_current_manual = 0
        self.chess_manuals = list()
        self.idx_current_manuals = 0
        self.idx_iter = 0

        self.load_chess_manuals(self.file_list)
        self.count_chess_manual = len(self.chess_manuals)
        self.imgs_dict = dict()
        for i in range(225):
            idx = f'000{i}'[-3:]
            self.imgs_dict[f'w_{idx}'] = list()
            self.imgs_dict[f'f_{idx}'] = list()
        pass

    def __iter__(self):
        self.idx_iter = 0
        return self

    def __next__(self):
        if self.idx_iter <= self.idx_current_manual:
            self.idx_iter = self.idx_current_manual
        else:
            self.idx_current_manual = 0
            raise StopIteration
        if 'valid' == self.mode:
            return self.__load_data()
        elif 'create' == self.mode:
            return None

    def __load_data(self):
        count_data = 0
        manual_list = list()
        for k in range(self.batch_section_size // 8 + 1):
            if self.idx_current_manual >= len(self.chess_manuals):
                self.idx_current_manual = 0
                break

            chess_manual = self.chess_manuals[self.idx_current_manual]
            if len(self.data_buffer) > self.batch_section_size:
                break

            count_data += len(chess_manual)
            self.idx_current_manual += 1
            winner = self.WhiteCHess if len(chess_manual) % 2 == 0 else self.BlackChess
            manuals = self.revert_manual_step(chess_manual, winner)
            manual_list += manuals
        return manual_list
        # return random.sample(self.data_buffer, self.batch_size)

    def revert_manual_step(self, chess_manual, winner, min_step=28):
        if len(chess_manual) < min_step:
            return list()

        manuals = list()
        last_move = -1
        for i, step in enumerate(chess_manual):
            player_idx = self.BlackChess if i % 2 == 0 else self.WhiteCHess
            move = node2flatten(step, self.board_size)
            loc_state = self.get_board(player_idx, last_move, is_copy=True)
            winner_z = 1.0 if (winner == player_idx) or i < min_step // 2 else -1.0
            self.filter_save_state(i, loc_state, winner_z, move, chess_manual)
            manuals.append([loc_state, [move, winner_z]])
            self.update_board(move, player_idx)
            last_move = move
            pass
        self.init_boards()
        return list()

    def filter_save_state(self, idx, loc_state, winner_z, move, manual):
        len_manual = len(manual)
        self.save_state(loc_state, winner_z, move)
        # if (len_manual > 50 and not (len_manual // 2 < idx < len_manual - 6) or
        #         (len_manual < 15 and idx > len_manual - 3) or
        #         (15 <= len_manual <= 50 and idx > len_manual - 5)):
        #     self.save_state(loc_state, winner_z, move)
        pass

    def save_state(self, state, winner_z, move):
        if winner_z == 1.0:
            cls_dir_prefix = 'w'
        else:
            cls_dir_prefix = 'f'
            move = 224 - move
        idx_move = f'000{move}'[-3:]
        key = f'{cls_dir_prefix}_{idx_move}'
        filename = f'section_{self.idx_current_manual}_{idx_move}_{winner_z + 1}.png'
        fp = os.path.join(self.obj_root, key, filename)
        img = np.transpose(np.array(state, dtype=np.uint8), axes=[1, 2, 0]) * 255
        self.imgs_dict[key].append(img)
        num_row = 4
        if len(self.imgs_dict[key]) == num_row * num_row:
            imgs_list = self.imgs_dict[key]
            imgs = np.concatenate(
                [np.concatenate([imgs_list[i * num_row + j] for j in range(num_row)],
                                axis=1) for i in range(num_row)], axis=0)
            cv2.imwrite(fp, imgs)
            self.imgs_dict[key] = list()
        pass

    def update_board(self, move, player_idx):
        (width, _) = self.board_size
        wi, hi = move // width, move % width
        self.boards[player_idx][0][wi, hi] = 1.0
        self.boards[-player_idx][1][wi, hi] = 1.0
        pass

    def get_board(self, player_idx, last_move=-1, is_copy=False):
        curr_board = np.copy(self.boards[player_idx]) if is_copy else self.boards[player_idx]
        if is_copy and last_move >= 0:
            wi, hi = last_move // self.board_size[0], last_move % self.board_size[1]
            curr_board[1:3, wi, hi] = 1.0
            return curr_board[:, ::-1, :]
        return curr_board

    def load_chess_manuals(self, file_list):
        chess_manuals = list()
        for file in file_list:
            with open(file, 'r') as f:
                lines = f.readlines()
                for str_line in lines:
                    if str_line[-1] == '\n':
                        str_line = str_line[:-1]
                    chess_manual = str_line.split(' ')
                    chess_manual = self.check_manual(chess_manual)
                    if len(chess_manual) > 8:
                        chess_manuals.append(chess_manual)

        if self.is_extend:
            for manual in chess_manuals:
                rotates = [rot90board(manual, i) for i in range(4)]
                flips = [np.fliplr(rot) for rot in rotates]
                self.chess_manuals += rotates + flips
        else:
            self.chess_manuals = chess_manuals
        random.shuffle(self.chess_manuals)

    def init_boards(self):
        self.boards = list([np.zeros((4, self.board_size[0], self.board_size[1])) for _ in range(3)])
        self.boards[self.BlackChess][3][:, :] = 1.0
        pass

    @staticmethod
    def walkFile(dir_root, suffix='.ui'):
        files_list = list()
        for root, _, fs in os.walk(os.path.abspath(dir_root)):
            for f in fs:
                if f[-len(suffix):] == suffix:
                    files_list.append(os.path.join(root, f))
                pass
            pass
        return files_list

    def check_manual(self, chess_manual):
        if len(chess_manual) <= 8:
            return list()

        node_list = list()
        for i, step in enumerate(chess_manual):
            if not (2 <= len(step) <= 3 and ord('a') <= ord(step[0]) <= ord('o')):
                return list()

            node_step = self.sn2node(step)
            if not (0 <= node_step[0] < 15 and 0 <= node_step[1] < 15):
                return list()

            node_list.append(node_step)
        return np.array(node_list, dtype=np.uint8)

    @staticmethod
    def sn2node(str_node):
        try:
            return [ord(str_node[0]) - ord('a'), 15 - int(str_node[1:])]
        except IndexError as e:
            print(f'{str_node}')
            raise e
        pass


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, y_pred, y_true):
        prob = self.softmax(y_pred.view(-1, self.class_num))
        prob = prob.clamp(min=0.0001, max=1.0)

        target_ = y_true.cuda()

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(
                1 - prob, self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1 - prob, self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss
    pass


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
        self.act_loss = FocalLoss(225)
        self.state_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        y_true[1] = y_true[1].to(y_pred[1].dtype)
        return self.act_loss(y_pred[0], y_true[0]) + self.state_loss(y_pred[1], y_true[1])


class PolicyNet(object):
    """PolicyNet

        该类主要用于强化训练, 直接将已经训练过的模型结果用于自动下棋.
        然后胜利后, 下棋过程产生的棋谱直接用于训练.

    """

    def __init__(self, net, device=torch.device('cuda'), is_training=False, cls_name='PolicyPlayer'):
        self.device = device
        if isinstance(net, nn.Module):
            self.net = net.to(device)
        else:
            with open(net) as f:
                net_cfg = yaml.load(f, Loader=yaml.SafeLoader)
                self.net = NNet(net_cfg, cls_name=cls_name).to(device)

        self.player_id = cbd.NoWinner
        self.init_train_params(is_training)
        pass

    def get_action(self, board, **kwargs):
        available_moves = np.array(board.available_moves, dtype=np.int)
        board_section = board.get_current_section()
        act_prob = self.net(torch.Tensor(board_section[np.newaxis, :, :, :]).to(self.device))
        action = act_prob[0].cpu().detach().numpy().flatten()
        probs = act_prob[1].cpu().detach().numpy()
        return np.random.choice(available_moves, p=softmax(action[available_moves])), probs.item()

    def policy_value_fn(self, board):
        chess_type = board.get_current_player()
        board_section = board.get_board_section(chess_type)
        act_prob = self.net(torch.Tensor(board_section[np.newaxis, :, :, :]).to(self.device))
        action = act_prob[0].cpu().detach().numpy().flatten()
        probs = act_prob[1].cpu().detach().numpy()
        move_softmax = softmax(action[board.available_moves])
        return zip(board.available_moves, move_softmax), probs.item()

    def set_player_id(self, idx=0):
        self.player_id = idx
        pass

    def init_train_params(self, is_training, device=torch.device('cuda'), metrics=None):
        dataset_name = 'WZQManuals'
        class_name = self.net.__class__.__name__
        cp_callback = load_breakpoint(self.net, dataset_name, class_name, save_weights_only=True,
                                      map_location=device, pickle_module=dill)

        if is_training:
            # 损失函数
            loss = PolicyNetLoss()
            self.net.compile(optimizer='adam', loss=loss, metrics=metrics, callbacks=[cp_callback], device=device,
                             data_fn=DataLoader.data_fn)
        pass


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
