# -*- coding: utf-8 -*-
"""
@author: wnyl
"""


class MCTreeSearch(object):
    """MCTreeSearch: 霥特卡罗搜索树

        An implementation of Monte Carlo Tree Search.
    """
    def __init__(self, policy_value_fn, c_puct=5, num_roll_out=10000, **kwargs):
        pass

    def roll_out(self, board, **kwargs):
        pass

    def get_move_probs(self, board, temp=1.0e-3, **kwargs):
        pass

    def update_with_move(self, move, **kwargs):
        pass
    pass


class MCPlayer(object):
    """MCPlayer

        AI player based on MCTS.
    """
    def __init__(self, policy_value_fn, c_puct=5, num_roll_out=2000, is_self_play=False, **kwargs):
        self.mcts = None
        self.is_self_play = is_self_play
        self.player_id = 0
        pass

    def reset_player(self):
        pass

    def set_id(self, player_id, **kwargs):
        pass

    def get_action(self, board, temp=1e-3, return_prob=False, **kwargs):
        pass
