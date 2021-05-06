import dill
import numpy as np
import torch

from chess_manual.checkwin import CheckWinNet
from chess_manual.chessboard_define import cbd
from chess_manual.manual_api import MCPlayer
from chess_manual.manualloader import DataLoader, PolicyNetLoss, load_breakpoint


def rot90board(chess_manual, axes=0, out_type='node', board_size=(15, 15)):
    """rot90board
        逆时针旋转棋盘对局 90 度

    Args:
        chess_manual: 棋谱
        axes: 旋转次数, 0: (w, h); 1: (h, wi-w); 2: (hi-h, wi-w); 3: (hi-h, w)
        out_type: 'node', 'move'
        board_size: 棋盘尺寸
    """
    if isinstance(chess_manual, tuple) or isinstance(chess_manual, list):
        chess_manual = np.array(chess_manual)
    wi, hi = board_size[0] - 1, board_size[1] - 1
    ret_manual = np.copy(chess_manual)
    if axes % 4 == 1:
        ret_manual[:, 0] = chess_manual[:, 1]
        ret_manual[:, 1] = wi - chess_manual[:, 0]
    elif axes % 4 == 2:
        ret_manual[:, 0] = hi - chess_manual[:, 1]
        ret_manual[:, 1] = wi - chess_manual[:, 0]
    elif axes % 4 == 3:
        ret_manual[:, 0] = hi - chess_manual[:, 1]
        ret_manual[:, 1] = chess_manual[:, 0]
    if 'node' == out_type:
        return ret_manual
    else:
        return ret_manual[:, 0] + ret_manual[:, 1] * board_size[0]


def rot90moves(moves_manual, axes=0, board_size=(15, 15)):
    node_manual = np.zeros([len(moves_manual), 2], dtype=np.int)
    node_manual[:, 0] = moves_manual // board_size[0]
    node_manual[:, 1] = moves_manual % board_size[0]
    node_manual = rot90board(node_manual, axes=axes, board_size=board_size)
    return node_manual[:, 0] * board_size[0] + node_manual[:, 1]


def fliplr90moves(moves_manual, board_size=(15, 15)):
    node_manual = np.zeros([len(moves_manual), 2], dtype=np.int)
    node_manual[:, 0] = moves_manual // board_size[0]
    node_manual[:, 1] = moves_manual % board_size[0]
    node_manual[:, 0] = board_size[0] - node_manual[:, 0] - 1
    return node_manual[:, 0] * board_size[0] + node_manual[:, 1]


class ChessBoard:
    """ChessBoard
        棋盘
    """
    def __init__(self, board_size=(15, 15), nlc_min=5):
        self.board_size = board_size
        self.count_chess = board_size[0] * board_size[1]

        self.states = dict()
        self.boards = list()
        self.buffer = list()
        self.move_buffer = list()
        self.winners_buffer = list()

        # 状态
        self.winner = cbd.NoWinner
        self.available_moves = [move for move in range(self.count_chess)]

        self.check_win = CheckWinNet(board_size, nlc_min)
        self.init_boards()
        pass

    def do_move(self, move):
        player_id = cbd.BlackChess if len(self.states) % 2 == 0 else cbd.WhiteChess
        self.update_board(move, player_id)
        pass

    def get_current_player(self):
        return cbd.BlackChess if len(self.states) % 2 else cbd.WhiteChess

    def game_state(self):
        """game_state
            游戏状态

        Returns:
            Tuple([is_end, winner]), (棋局状态, 获胜者)
        """
        return (len(self.available_moves) == 0 or self.winner != cbd.NoWinner), self.winner

    def init_boards(self):
        """黑子先手"""
        self.check_win.init()
        self.boards = list([None] + [np.zeros((4, self.board_size[0], self.board_size[1])) for _ in range(2)])
        self.boards[cbd.BlackChess][3][:, :] = 1.0
        pass

    def update_board(self, move, player_id):
        self.states[move] = player_id
        self.available_moves.remove(move)

        node = self.flatten2node(move, is_flip=False)
        self.check_win.update_board_chess(node, player_id)
        (width, _) = self.board_size
        wi, hi = move // width, move % width
        self.boards[player_id][0][wi, hi] = 1.0
        self.boards[-player_id][1][wi, hi] = 1.0
        pass

    def get_board(self, player_id, last_move=-1, is_copy=False):
        curr_board = np.copy(self.boards[player_id]) if is_copy else self.boards[player_id]
        if is_copy and last_move >= 0:
            wi, hi = last_move // self.board_size[0], last_move % self.board_size[1]
            curr_board[1:3, wi, hi] = 1.0
            return np.copy(curr_board[:, ::-1, :])
        return curr_board

    def get_board_section(self, chess_type, move=0):
        if 0 <= move < self.count_chess:
            board_section = np.copy(self.boards[chess_type])
            wi, hi = move // self.board_size[0], move % self.board_size[1]
            board_section[1:3, wi, hi] = 1.0
            return np.copy(board_section[:, ::-1, :])
        return None

    def flatten2node(self, flatten, is_flip=True):
        """flatten2node
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


class ChessGame(object):
    def __init__(self, chess_board: ChessBoard, trainer=None, player=None, buffer_size=10000):
        self.chess_board = chess_board
        self.trainer = trainer  # 训练器
        self.player = player    # 对手器
        self.buffer_size = buffer_size
        self.manuals_buffer = list()
        pass

    def auto_play(self, epochs_exp=20, batch_size=256):
        idx_first_play = 1
        epochs = 2 ** epochs_exp
        policies = [None, self.trainer, self.player]
        for epoch in range(epochs):
            # 进行一局对战
            winner, manual_states = self.play_one_game(policies[idx_first_play], policies[-idx_first_play])

            # 存储数据, 进行训练
            idx_first_play = -idx_first_play
            self.manuals_buffer.append([winner, manual_states])
            if len(self.manuals_buffer) > batch_size:
                self.train_once(self.trainer.net, self.manuals_buffer, epoch)
        pass

    def self_play(self, epochs_exp=20, batch_size=256):
        idx_first_play = 1
        epochs = 2 ** epochs_exp
        policies = [None, self.trainer, self.trainer]
        for epoch in range(epochs):
            # 进行一局对战
            winner, manual_states = self.play_one_game(policies[idx_first_play], policies[-idx_first_play])

            # 存储数据, 进行训练
            idx_first_play = -idx_first_play
            self.manuals_buffer.append([winner, manual_states])
            if len(self.manuals_buffer) > batch_size:
                self.train_once(self.trainer.net, self.manuals_buffer, epoch)
        pass

    def play_one_game(self, player_one: MCPlayer, player_two: MCPlayer):
        """play_one_game
            进行一场对局
        """
        self.reset_game()  # 重置游戏数据

        moves = list()
        probs = list()

        curr_id = 1  # 开始一号(黑子)先手
        players = [None, player_one, player_two]
        while True:
            curr_player = players[curr_id]
            move, move_prob = curr_player.get_action(self.chess_board, temp=1.0, return_prob=True)

            # 收集走子信息
            moves.append(move)
            probs.append(move_prob)

            # 更新棋盘状态
            self.chess_board.update_board(move, curr_id)
            self.chess_board.check_win(self.chess_board.flatten2node(move, is_flip=False))
            is_end, winner = self.chess_board.game_state()
            if is_end:
                break
            curr_id = -curr_id
        return winner, zip(moves, probs)

    @staticmethod
    def train_once(net, epochs=600, batch_size=256, data_root_dir='~/.dataset/data_paper/', **kwargs):
        """train_once
            对网络 net 进行一轮训练, 并更新模型参数.
        """
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

    def reset_game(self):
        self.chess_board.init_boards()
        self.trainer.reset_player()
        self.player.reset_player()
