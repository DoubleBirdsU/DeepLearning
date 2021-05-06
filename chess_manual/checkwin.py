import numpy as np
import torch


class CheckWinNet(object):
    NoWinner = 0
    BlackPlayer = 1
    WhitePlayer = -1

    def __init__(self, board_size=(19, 19), nlc_min=5):
        self.board_size = board_size
        self._section_size = (1, 1, board_size[0], board_size[1])
        self.nlc_min = nlc_min
        self.chess_section = torch.zeros(self._section_size)
        self.win_conv = torch.nn.Conv2d(1, 4, kernel_size=5, padding=2, bias=False)
        self.init_weight(nlc_min)

        # 棋局状态
        self.section_list = list()

    def __call__(self, node):
        winner = self.NoWinner
        current_player = self.get_board_chess(node)
        features = self.win_conv(self.chess_section).data.int().numpy()[0]
        for val in [np.min(features), np.max(features)]:
            if val * current_player <= -4.0:
                nodes = np.argwhere(features == val)
                for node in nodes:
                    if self.is_win_pos(node):
                        return self.BlackPlayer if val > 0 else self.WhitePlayer
        return winner

    def init(self):
        self.section_list = list()
        self.chess_section = torch.zeros(self._section_size)

    def update_board_chess(self, node, value):
        self.chess_section[0, 0, node[0], node[1]] = value

    def get_board_chess(self, node):
        return int(self.chess_section[0, 0, node[0], node[1]])

    def init_weight(self, nlc_min=5):
        conv_weight = np.zeros([4, 1, nlc_min, nlc_min])
        conv_weight[0, 0, 2, :] = 1
        conv_weight[1, 0, :, 2] = 1
        for i in range(nlc_min):
            conv_weight[2, 0, i, i] = 1
            conv_weight[3, 0, i, nlc_min - i - 1] = 1
        self.win_conv.weight.data = torch.Tensor(conv_weight)
        pass

    def is_win_pos(self, pos):
        w_win_pos = 2 <= pos[1] <= self.board_size[0] - 3
        h_win_pos = 2 <= pos[2] <= self.board_size[1] - 3
        if pos[0] == 0:
            return h_win_pos
        elif pos[0] == 1:
            return w_win_pos
        elif pos[0] >= 2:
            return w_win_pos and h_win_pos
        return False

    @staticmethod
    def index2pos(index, shape):
        shape = np.array(shape)
        pos = np.zeros_like(shape)
        pos[0] = index
        for i in range(1, len(shape)):
            pos[i] = pos[i - 1] % shape[i:].prod()
            pos[i - 1] = pos[i - 1] // shape[i:].prod()
        return tuple(pos)
