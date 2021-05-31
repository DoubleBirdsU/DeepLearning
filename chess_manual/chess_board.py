import time
import random

import dill
import numpy as np
import torch

from chess_manual.checkwin import CheckWinNet
from chess_manual.chess_utils import rot90moves, fliplrmoves, rot90probs, fliplrprobs, flatten2node
from chess_manual.chessboard_define import cbd
from chess_manual.manual_api import MCPlayer
from chess_manual.manualloader import PolicyNetLoss, load_breakpoint, softmax


def get_extend_manual(winner, chess_manual, move_prob, board_sections, move_scale=1.0, board_size=(15, 15)):
    num_step = len(chess_manual)
    up_scale, down_scale = 1.01, 0.999

    winners = -np.ones(num_step) * winner
    winners[np.arange((num_step + 1) // 2) * 2] = winner

    log_step = np.log(0.5) / (num_step - 1)
    for i in range(num_step):
        win = (1 - i % 2 * 2) * winner
        move_prob[i][chess_manual[i]] *= up_scale if win > 0 else down_scale
        move_prob[i] = move_prob[i] / move_prob[i].sum()

        # winners[i] *= np.exp((num_step - 1 - i) * log_step)

    train_data = create_extend(chess_manual, np.array(move_prob), board_sections, move_scale, board_size)
    return [[manual, winners, prob, sect] for manual, prob, sect in train_data]


def create_extend(chess_manual, move_prob, board_sections, move_scale=1.0, board_size=(15, 15)):
    manuals = [rot90moves(chess_manual, i, board_size) for i in range(4)]
    manuals += [fliplrmoves(manual) for manual in manuals]

    probs = [rot90probs(move_prob, i, board_size) for i in range(4)]
    probs += [fliplrprobs(prob, board_size) for prob in probs]

    sects = [np.rot90(board_sections, i, axes=(-2, -1)) for i in range(4)]
    sects += [sect[..., ::-1, :] for sect in sects]

    prob_1 = np.zeros_like(probs[0][0, ...])
    for prob in probs:
        prob_1 += prob[1, ...]
    prob_1 = prob_1 / prob_1.sum()
    for prob in probs:
        prob[1, ...] = prob_1
    return zip(manuals, probs, sects)


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
        self.available_moves = list()

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
        self.states = dict()
        self.boards = list()
        self.buffer = list()
        self.move_buffer = list()
        self.winners_buffer = list()

        # 其他重置
        self.winner = cbd.NoWinner
        self.available_moves = [move for move in range(self.count_chess)]
        self.boards = list([None] + [np.zeros((4, self.board_size[0], self.board_size[1])) for _ in range(2)])
        self.boards[cbd.BlackChess][3][:, :] = 1.0
        pass

    def update_board(self, move, player_id):
        self.states[move] = player_id
        self.available_moves.remove(move)
        self.move_buffer.append(move)

        node = flatten2node(move)
        self.check_win.update_board_chess(node, player_id)
        self.winner = self.check_win(node)

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

    def get_board_section(self, chess_type, move=-1):
        board_section = np.copy(self.boards[chess_type])
        if 0 <= move < self.count_chess:
            wi, hi = move // self.board_size[0], move % self.board_size[1]
            board_section[1:3, wi, hi] = 1.0
        return np.copy(board_section[:, ::-1, :])

    def get_current_section(self):
        return self.get_board_section(
            -self.get_current_player(),
            self.move_buffer[-1] if len(self.move_buffer) > 0 else -1)


def targets_fn(x, *args, **kwargs):
    return x


class ChessGame(object):
    def __init__(self, chess_board: ChessBoard, trainer=None, player=None, buffer_size=10000, is_self_play=False):
        self.chess_board = chess_board
        self.trainer = trainer  # 训练器
        self.player = player    # 对手器
        self.buffer_size = buffer_size
        self.is_self_play = is_self_play
        self.manuals_buffer = list()

        self.trainer.is_self_play = is_self_play
        pass

    def auto_play(self, epochs_exp=20, epoch_size=1024):
        idx_first_play = 1
        epochs = 2 ** epochs_exp
        policies = [None, self.trainer, self.trainer if self.is_self_play else self.player]

        device = torch.device('cuda')
        net = self.trainer.net.to(device)

        for epoch in range(epochs):
            # 进行一局对战
            winner, manual_state = self.play_one_game(policies[idx_first_play], policies[-idx_first_play])

            # 存储数据, 进行训练
            idx_first_play = -idx_first_play

            self.manuals_buffer.extend(manual_state)
            if (epoch + 1) % epoch_size == 0:
                random.shuffle(self.manuals_buffer)
                self.train_step(net, self.manuals_buffer, 1024)
                self.manuals_buffer.clear()
                print(f'\n\n========= Self-Play ==========')
        pass

    @staticmethod
    def train_step(net, manuals_buffer, batch_size=256, eopchs=64):
        targets = list()
        winners_z = list()
        data = list()
        for _, winners, target, state in manuals_buffer:
            winners_z.append(winners)
            targets.append(target)
            data.append(state)
        targets = np.concatenate(targets, axis=0)
        winners_z = np.concatenate(winners_z, axis=0)
        data = np.concatenate(data, axis=0)

        num_data = data.shape[0] // batch_size * batch_size
        mask = np.arange(num_data)
        np.random.shuffle(mask)

        targets = targets[mask, ...]
        winners_z = winners_z[mask, ...]
        data = data[mask, ...]

        states, trues = list(), list()
        for epoch in range(eopchs):
            print(f'\nEpoch: {epoch + 1}')
            count_batch = num_data // batch_size
            loss_mean = 0.0
            for i in range(count_batch):
                if epoch == 0:
                    idx_s, idx_e = i * batch_size, (i + 1) * batch_size
                    data_state = torch.Tensor(data[idx_s:idx_e, ...])
                    y_true = [torch.Tensor(targets[idx_s:idx_e, ...]),
                              torch.Tensor(winners_z[idx_s:idx_e, ...]).unsqueeze_(-1)]
                    states.append(data_state)
                    trues.append(y_true)
                else:
                    data_state = states[i]
                    y_true = trues[i]
                loss, correct = net.train_step(data_state, y_true, cur_batch=i + 1, count_batch=count_batch)

                loss_mean += (loss - loss_mean) / (i + 1)
            net.checkpoint_fn(net, accuracy=0, loss=loss_mean)
        pass

    def play_one_game(self, player_one: MCPlayer, player_two: MCPlayer):
        """play_one_game
            进行一场对局
        """
        self.reset_game()  # 重置游戏数据

        moves = list()
        probs = list()
        sects = list()

        curr_id = 1  # 开始一号(黑子)先手
        players = [None, player_one, player_two]
        move_version = 'v1'
        is_first = True
        time_start = time.time()
        while True:
            curr_player = players[curr_id]
            if is_first:
                board_size = self.chess_board.board_size
                count_chess = board_size[0] * board_size[1]
                move, move_prob = count_chess // 2, np.zeros(count_chess)
                move_prob[move] = 1.0
                is_first = False
            else:
                move, move_prob = curr_player.get_action(
                    self.chess_board, temp=1.0, return_prob=True, move_version=move_version)

            # 收集走子信息
            moves.append(move)
            probs.append(move_prob)
            sects.append(self.chess_board.get_current_section())

            # 更新棋盘状态
            self.chess_board.update_board(move, curr_id)
            is_end, winner = self.chess_board.game_state()
            if is_end:
                break
            curr_id = -curr_id
        print(f'cast time: {time.time() - time_start}')
        # 输出棋局
        with open('./chess_manual/chess_cn/chess.txt', 'a+') as f:
            f.write(f'{moves}\n')
        return winner, get_extend_manual(winner, moves, probs, sects, move_scale=1.0)

    @staticmethod
    def train_once(net, epochs=600, batch_size=256, **kwargs):
        """train_once
            对网络 net 进行一轮训练, 并更新模型参数.
        """
        device = torch.device('cuda')
        net = net.to(device)

        # 断点续训
        dataset_name = 'WZQManuals'
        class_name = net.__class__.__name__
        cp_callback = load_breakpoint(net, dataset_name, class_name, save_weights_only=True,
                                      map_location=device, pickle_module=dill)

        # 损失函数
        loss = PolicyNetLoss()
        net.compile(optimizer='adam', loss=loss, metrics=['acc'], device=device, callbacks=[cp_callback])

        # 训练
        # net.train_step(train_loader,
        #                   batch_size=batch_size,
        #                   epochs=epochs,
        #                   validation_data=val_loader,
        #                   callbacks=[cp_callback])
        pass

    def reset_game(self):
        if self.chess_board is not None:
            self.chess_board.init_boards()
        if self.trainer is not None and isinstance(self.trainer, MCPlayer):
            self.trainer.reset_player()
        if self.player is not None and isinstance(self.trainer, MCPlayer):
            self.player.reset_player()
        pass
