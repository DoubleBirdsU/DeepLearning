import os
import random
import torch

import cv2
import numpy as np
import torch.utils.data as udata
import torchvision.transforms as transforms

from torchvision import datasets as ds
from torchvision.transforms import transforms


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


def rot90board(chess_manual, axis=0, board_size=(15, 15)):
    """rot90board

    Args:
        axis = 0     1          2            3
             w, h  h, wi-w  hi-h,wi-w    hi-h,w
    """
    if isinstance(chess_manual, tuple) or isinstance(chess_manual, list):
        chess_manual = np.array(chess_manual)
    wi, hi = board_size[0] - 1, board_size[1] - 1
    ret_manual = np.copy(chess_manual)
    if axis % 4 == 1:
        ret_manual[:, 0] = chess_manual[:, 1]
        ret_manual[:, 1] = wi - chess_manual[:, 0]
    elif axis % 4 == 2:
        ret_manual[:, 0] = hi - chess_manual[:, 1]
        ret_manual[:, 1] = wi - chess_manual[:, 0]
    elif axis % 4 == 3:
        ret_manual[:, 0] = hi - chess_manual[:, 1]
        ret_manual[:, 1] = chess_manual[:, 0]
    return ret_manual


class DataLoader:
    def __init__(self,
                 data_root,
                 batch_size=128,
                 num_workers=4,
                 shuffle=True):
        batch_size = batch_size // 4
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
    def target_fn(targets):
        device = targets.device
        winners_z = torch.ones_like(targets, dtype=torch.float)
        targets = targets - 225
        winners_z[targets < 0] = -1.0
        winners_z[targets > 0] = 1.0
        winners_z.unsqueeze_(-1)
        targets = torch.abs_(targets) - 1
        winners_z = torch.cat([winners_z for _ in range(4)], dim=0)
        targets = torch.cat([targets for _ in range(4)], dim=0)
        return [targets.to(device=device), winners_z.to(device=device)]

    @staticmethod
    def data_fn(data):
        data_list = [data[:, :, :15, :15], data[:, :, :15, 15:],
                     data[:, :, 15:, 15:], data[:, :, 15:, :15]]
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

    def revert_manual_step(self, chess_manual, winner):
        manuals = list()
        last_move = -1
        for i, step in enumerate(chess_manual):
            player_idx = self.BlackChess if i % 2 == 0 else self.WhiteCHess
            move = self.node2flatten(step, False)
            loc_state = self.get_board(player_idx, last_move, is_copy=True)
            winner_z = 1.0 if winner == player_idx else -1.0
            self.filter_save_state(i, loc_state, winner_z, move, chess_manual)
            manuals.append([loc_state, [move, winner_z]])
            self.update_board(move, player_idx)
            last_move = move
            pass
        self.init_boards()
        return list()

    def filter_save_state(self, idx, loc_state, winner_z, move, manual):
        len_manual = len(manual)
        if (len_manual > 50 and not (len_manual // 2 < idx < len_manual - 6) or
                (len_manual < 15 and idx > len_manual - 3) or
                (15 <= len_manual <= 50 and idx > len_manual - 5)):
            self.save_state(loc_state, winner_z, move)
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
        if len(self.imgs_dict[key]) == 4:
            imgs_list = self.imgs_dict[key]
            imgs = np.concatenate([np.concatenate([imgs_list[i * 2 + j] for j in range(2)], axis=1) for i in range(2)],
                                  axis=0)
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

    def flatten2node(self, flatten, is_flip=True):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h, w = flatten % self.board_size[0], flatten // self.board_size[0]
        return [w, h] if is_flip else [h, w]

    def node2flatten(self, node, is_flip=True):
        if len(node) != 2:
            return -1
        [h, w] = node if is_flip else [node[1], node[0]]
        flatten = h * self.board_size[0] + w
        return flatten if 0 <= flatten < self.count_chess else -1

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
