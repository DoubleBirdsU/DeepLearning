import os
import time

import dill
import torch
import yaml

DEFAULT_PROTOCOL = 2


def make_callbacks(net, callbacks):
    """make_callbacks
        checkpoint_fn, checkdata_fn

    Args:
        net:
        callbacks:

    """
    if callbacks is None:
        return

    for fn in callbacks:
        if isinstance(fn, ModelCheckpoint):
            net.checkpoint_fn = fn
        elif isinstance(fn, DatasetCheck):
            net.checkdata_fn = fn


class ModelCheckpoint(object):
    Weight = 'weights'
    Model = 'model'
    All = 'all'
    Last = 'last'
    Best = 'best'
    Index = 'index'
    ckpt_filename = 'ckpt.yaml'

    def __init__(
            self, filepath,
            save_weights_only=False,
            save_best_only=False,
            pickle_module=dill,
            pickle_protocol=DEFAULT_PROTOCOL,
            _use_new_zipfile_serialization=True
    ):
        self.filepath = filepath
        self.root_path = os.path.dirname(filepath)
        self.save_weights_only = save_weights_only
        self.save_best_only = save_best_only
        self.pickle_module = pickle_module
        self.pickle_protocol = pickle_protocol
        self._use_new_zipfile_serialization = _use_new_zipfile_serialization

        # ckpt
        self.ckpt = self.check_ckpt_complete(self.ckpt_read(self.root_path, self.ckpt_filename))
        self.accuracy_best = self.ckpt['accuracy'] if 'accuracy' in self.ckpt else 0
        self.loss_best = self.ckpt['loss'] if 'loss' in self.ckpt else float('inf')
        self.ckpt['filename'] = os.path.abspath(self.filepath)
        self.ckpt['save_best_only'] = self.save_best_only

    def __call__(self, net, **kwargs):
        """
            accuracy:

        Args:
            net (NNet):
        """
        acc_cur = kwargs.pop('accuracy') if 'accuracy' in kwargs else 0.  # 取出 accuracy 参数
        loss = kwargs.pop('loss') if 'loss' in kwargs else float('inf')
        if self.save_weights_only:  # parameters
            f = net.state_dict()
            self.ckpt['save_mode'] = self.Weight
        else:  # model and parameters
            f = net
            self.ckpt['save_mode'] = self.Model

        index_last = self.Last
        index_best = self.Best
        mode = None
        if not self.save_best_only:
            mode = '{}-of-{}_{:.4f}'.format(net.epoch, net.epochs, acc_cur)
            index_last = self.Index

        # Save last 存在问题: 如果新训练的覆盖了旧的概率模型
        suffix_last = self.get_suffix(index=index_last, mode=mode)
        file_last = self.filepath + suffix_last
        param_change = (self.accuracy_best < acc_cur or loss < self.loss_best or
                        (self.save_best_only and not os.path.exists(file_last)))
        torch.save(f, self.filepath + suffix_last, self.pickle_module, self.pickle_protocol,
                   self._use_new_zipfile_serialization)

        # Accuracy and Loss
        suffix_best = self.ckpt['suffix_best'] if 'suffix_best' in self.ckpt else suffix_last
        if param_change:
            suffix_best = suffix_last
            self.accuracy_best = acc_cur
            self.loss_best = float(loss.cpu().detach().numpy()) if isinstance(loss, torch.Tensor) else loss

        # Save best
        if self.save_best_only:
            suffix_best = self.get_suffix(index=index_best)
            file_best = self.filepath + suffix_best
            if param_change or not os.path.exists(file_best):
                self.ckpt['best_save_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                torch.save(f, file_best, self.pickle_module, self.pickle_protocol,
                           self._use_new_zipfile_serialization)

        # Dump ckpt
        self.ckpt['accuracy'] = self.accuracy_best
        self.ckpt['loss'] = self.loss_best
        self.ckpt['suffix_best'] = suffix_best
        self.ckpt['suffix_last'] = suffix_last
        self.ckpt_dump(self.ckpt, os.path.join(self.root_path, self.ckpt_filename))

    @staticmethod
    def get_suffix(index='index', mode=None):
        """
        Args:
            index: 'last', 'best', 'index'
            mode:

        Returns:
            str
        """
        return (f'.{index}' if index else '') + (f'-{mode}' if mode else '')

    @staticmethod
    def ckpt_read(ckpt_path, ckpt_name='ckpt.yaml'):
        ckpt_file = os.path.join(ckpt_path, ckpt_name)

        ckpt = dict()
        if os.path.exists(ckpt_file):
            with open(ckpt_file, 'r', encoding='utf-8') as f:
                ckpt = yaml.load(f.read(), Loader=yaml.SafeLoader)

        return ckpt

    @staticmethod
    def ckpt_dump(ckpt, ckpt_path):
        with open(ckpt_path, 'w', encoding='utf-8') as f:
            yaml.dump(ckpt, f, Dumper=yaml.SafeDumper)

    @staticmethod
    def check_ckpt_complete(ckpt):
        if ckpt is not None and 'filename' in ckpt and 'suffix_best' in ckpt and 'suffix_last' in ckpt:
            file_best = ckpt['filename'] + ckpt['suffix_best']
            file_last = ckpt['filename'] + ckpt['suffix_last']
            if os.path.exists(file_best) and os.path.exists(file_last):
                return ckpt
        return dict()


class DatasetCheck(torch.nn.Module):
    """DatasetCheck

        处理 loss 输入前 y_pred, y_true 数据

    """

    def __init__(self, trans_fn=None):
        super(DatasetCheck, self).__init__()
        self.trans_fn = trans_fn

    def __call__(self, y_pred, y_true):
        return self.trans_fn(y_pred, y_true) if self.trans_fn is not None else (y_pred, y_true)
