import numpy as np
from PyQt5.QtCore import QPoint


def pos2node(pos, zero_pos=(32, 32), grid_wh=(45, 45), board_size=(15, 15)):
    zero_pos, grid_wh = np.array(zero_pos), np.array(grid_wh)
    wi, hi = board_size
    if isinstance(pos, QPoint):
        pos = np.array([pos.x(), pos.y()])
    elif not isinstance(pos, np.ndarray):
        pos = np.array(pos)

    node = (pos - zero_pos + grid_wh // 2) // grid_wh
    node[0] = node[0] if 0 <= node[0] < wi else -1
    node[1] = node[1] if 0 <= node[1] < hi else -1
    return node


def node2pos(node, zero_pos=(32, 32), grid_wh=(45, 45), is_center=False, is_point=False):
    zero_pos, grid_wh = np.array(zero_pos), np.array(grid_wh)
    if isinstance(node, QPoint):
        node = np.array([node.x(), node.y()])
    elif not isinstance(node, np.ndarray):
        node = np.array(node)
    pos = node * grid_wh + zero_pos
    pos = pos if is_center else pos - grid_wh // 2
    return QPoint(pos[0], pos[1]) if is_point else pos


def node2flatten(node, board_size=(15, 15)):
    if isinstance(node, QPoint):
        node = np.array([node.x(), node.y()])
    elif not isinstance(node, np.ndarray):
        node = np.array(node)
    flatten = np.array(node[..., 1] * board_size[0] + node[..., 0]).astype(dtype=np.int)
    if node.ndim <= 1 or node.shape[0] <= 1:
        flatten = flatten[np.newaxis, ...]
    flatten[np.where(flatten >= board_size[0] * board_size[1])] = -1
    flatten = flatten.clip(min=-1)
    if node.ndim <= 1 or node.shape[0] <= 1:
        flatten = flatten.item()
    return flatten


def flatten2node(flatten, board_size=(15, 15)):
    return np.array([flatten % board_size[0], flatten // board_size[0]])


def rot90moves(moves_manual, k=0, board_size=(15, 15)):
    node_manual = flatten2node(np.array(moves_manual), board_size).T
    node_manual = rot90board(node_manual, k=k, board_size=board_size)
    return node2flatten(node_manual, board_size)


def fliplrmoves(moves_manual, board_size=(15, 15)):
    node_manual = flatten2node(np.array(moves_manual), board_size).T
    node_manual[:, 0] = (board_size[0] - 1) - node_manual[:, 0]
    return node2flatten(node_manual, board_size)


def rot90board(chess_manual, k=0, out_type='node', board_size=(15, 15)):
    """rot90board
        逆时针旋转棋盘对局 90 度

    Args:
        chess_manual: 棋谱
        k: 旋转次数, 0: (w, h); 1: (h, wi-w); 2: (wi-w, hi-h); 3: (hi-h, w)
        out_type: 'node', 'move'
        board_size: 棋盘尺寸
    """
    if isinstance(chess_manual, tuple) or isinstance(chess_manual, list):
        chess_manual = np.array(chess_manual)

    wi, hi = board_size[0] - 1, board_size[1] - 1
    ret_manual = np.copy(chess_manual)
    if k % 4 == 1:
        ret_manual[:, 0] = chess_manual[:, 1]
        ret_manual[:, 1] = wi - chess_manual[:, 0]
    elif k % 4 == 2:
        ret_manual[:, 0] = wi - chess_manual[:, 0]
        ret_manual[:, 1] = hi - chess_manual[:, 1]
    elif k % 4 == 3:
        ret_manual[:, 0] = hi - chess_manual[:, 1]
        ret_manual[:, 1] = chess_manual[:, 0]

    if 'node' == out_type:
        return ret_manual
    else:
        return ret_manual[:, 0] + ret_manual[:, 1] * board_size[0]


def rot90probs(action_probs, k=0, board_size=(15, 15)):
    probs = np.reshape(action_probs, (action_probs.shape[0], *board_size))
    return np.reshape(np.rot90(probs, k=k, axes=(-2, -1)), action_probs.shape)


def fliplrprobs(action_probs, board_size=(15, 15)):
    probs = np.reshape(action_probs, (action_probs.shape[0], *board_size))
    return np.reshape(np.fliplr(probs), action_probs.shape)
