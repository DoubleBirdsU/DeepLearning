# coding=utf-8
import copy
import numpy as np

from chess_manual.chess_board import ChessBoard


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class Node(object):
    """A str_node in the MCTS tree.

        Each str_node keeps track of its own value Q, prior probability P, and
        its visit-count-adjusted prior score u.
    """

    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}
        self.num_visited = 0  # 访问次数
        self._self_value = 0  # 自身价值
        self._prior_pro = prior_p  # 先验概率
        self._adjusted_pro = 0  # 后验概率

    def expand(self, action_priors):
        """expand
            通过创建子节点进行树的扩展

        Args:
            action_priors: 动作先验, 由策略网络给出(评估当前局面)

        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = Node(self, prob)
        pass

    def search(self, c_puct):
        """search

        Returns:
            Tuple[(action, next_node)]
        """
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self.num_visited += 1
        self._self_value += 1.0 * (leaf_value - self._self_value) / self.num_visited
        pass

    def update_ancestors(self, leaf_value):
        """update_ancestors
            递归更新自身和祖先节点的价值

        Args:
            leaf_value: 节点的价值

        """
        if self.parent:
            self.parent.update_ancestors(-leaf_value)  # 祖先节点减少
            pass
        self.update(leaf_value)  # 更新当前节点
        pass

    def get_value(self, c_puct):
        """get_value

        Args:
            c_puct: (0, inf)
        """
        self._adjusted_pro = (c_puct * self._self_value * np.sqrt(self.parent.num_visited) / (self.num_visited + 1))
        return self._self_value + self._adjusted_pro

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None


class MCTreeSearch(object):
    """MCTreeSearch
        霥特卡罗搜索树, An implementation of Monte Carlo Tree Search.
    """

    def __init__(self, policy_value_fn, c_puct=5, num_roll_out=10000):
        """MCTreeSearch

        Args:
            policy_value_fn: a function that takes in a board state and outputs
                a list of (action, probability) tuples and also a score in [-1, 1]
                (i.e. the expected value of the end game score from the current
                player's perspective) for the current player.
            c_puct: a number in (0, inf) that controls how quickly exploration
                converges to the maximum-value policy. A higher value means
                relying on the prior more.
            num_roll_out:
        """
        self.root = Node(None, 1.0)
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.num_roll_out = num_roll_out
        pass

    def roll_out(self, board: ChessBoard):
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.search(self.c_puct)
            board.do_move(action)
            pass

        action_probs, leaf_value = self.policy_value_fn(board)
        is_end, winner = board.game_state()
        if not is_end:
            node.expand(action_probs)
        elif winner == 0:  # tie, 平局
            leaf_value = 0.0
        else:
            leaf_value = 1.0 if winner == board.get_current_player() else -1.0
            pass
        node.update_ancestors(-leaf_value)
        pass

    def get_move_probs(self, board: ChessBoard, temp=1.0e-3):
        """get_move_probs
            根据访问次数生成概率

        Args:
            board: 当前棋局状态
            temp:
        """
        for i in range(self.num_roll_out):
            board_copy = copy.deepcopy(board)
            self.roll_out(board_copy)
            pass

        act_visits = [(act, node.num_visited) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1.0e-10))
        return acts, act_probs

    def update_with_move(self, move):
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)
        pass

    def __str__(self):
        return 'MCTS'

    pass


class MCPlayer(object):
    """MCPlayer
        AI player based on MCTS.
    """

    def __init__(self, policy_value_fn, c_puct=5, num_roll_out=2000, is_self_play=False):
        self.mcts = MCTreeSearch(policy_value_fn, c_puct, num_roll_out)
        self.is_self_play = is_self_play
        self.player_id = 0
        pass

    def reset_player(self):
        self.mcts.update_with_move(-1)
        pass

    def set_id(self, player_id):
        self.player_id = player_id
        pass

    def get_action(self, board: ChessBoard, temp=1e-3, return_prob=False):
        """get_action
            获取下一次动作, 根据 MCTS 及 模型评估网络, 进行按概率获取下一次动作.

        Args:
            board: 当前棋局状态
            temp:
            return_prob: 是否返回 MCTS 概率分布
        """
        sensible_moves = board.available_moves
        move_probs = np.zeros(board.count_chess)
        if len(sensible_moves) == 0:
            print("WARNING: the board is full!")
            return (-1, move_probs) if return_prob else -1

        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        # 按照指定的概率选取下一次动作, 并更新 MCTS
        if self.is_self_play:  # 自我对奕
            move = np.random.choice(
                acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
            self.mcts.update_with_move(move)
        else:  # 与其他棋手对奕, 获取动作后重置 MCTS
            move = np.random.choice(acts, p=probs)
            self.reset_player()
            pass
        return (move, move_probs) if return_prob else move

    def __str__(self):
        return f'MCPlayer {self.player_id}'

    pass
