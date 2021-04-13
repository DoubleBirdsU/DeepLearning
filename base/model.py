import os
import torch
import pickle
import torch.optim as optim

from torch import nn
from base.base_model import base_model, Module
from base.message import Message
from base.share.callbacks import ModelCheckpoint, make_callbacks
from base.share.net_parser import NetParser

ONNX_EXPORT = False
DEFAULT_PROTOCOL = 2


class Model(base_model):
    def __init__(self, cls_name=None):
        super(Model, self).__init__()
        self.set_class_name(cls_name)

    def set_class_name(self, cls_name):
        if cls_name is not None:
            self.__class__.__name__ = cls_name


class NNet(Module):
    def __init__(self, cfg=None, layers_fn=None, include_top=True, cls_name=None):
        super(NNet, self).__init__(cls_name=cls_name)
        self.cfg = cfg
        self.layers_fn = layers_fn
        self.include_top = include_top
        self.cls_name = cls_name
        self.channels = None
        self.optimizer = None
        self.loss = None
        self.device = None
        self.metrics = None
        self.checkpoint_fn = None
        self.checkdata_fn = None
        self.epoch = 0
        self.epochs = 0
        self.msg = Message()
        self._callbacks = None
        self._save_weights_only = False,
        self._save_best_only = False,
        self._params_dict = dict()
        if cfg is not None:
            if cls_name is None:
                self.set_class_name(cfg['net_name'])
            self.cfg_module_list = nn.ModuleList()
            self.cfg_from_list = []
            self.cfg_value_list = []
            self.base = nn.ModuleList()
            self.classifier = nn.ModuleList()
            self.parser = NetParser(self, self.cfg)
            self.parser.create_net(include_top)

    def forward(self, *args, **kwargs):
        if self.cfg is not None:
            return self.forward_cfg(args[0])
        else:
            return super(NNet, self).forward(*args, **kwargs)

    def forward_cfg(self, x):
        self.cfg_value_list = [x]
        for from_idx, module in zip(self.cfg_from_list, self.cfg_module_list):
            self.cfg_value_list.append(
                module(self.cfg_value_list[from_idx] if isinstance(from_idx, int) else
                       [self.cfg_value_list[i] for i in from_idx]))
        return self.cfg_value_list[-1]

    def compile(self, optimizer=None, loss=None, call_params=None, metrics=None, device=torch.device('cpu')):
        """
        Args:
            optimizer:
            loss:
            call_params:
            device:
            metrics:

        Returns:
            None
        """
        self.optimizer = self.get_optimizer(optimizer, call_params)
        self.loss = loss
        self.metrics = metrics
        self.device = device
        self.msg.set_metrics(metrics)

    def fit_generator(
            self,
            train_data=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
    ):
        """fit_generator

        Args:
            train_data (DataLoader):
            batch_size ():
            epochs ():
            verbose ():
            callbacks ():
            validation_split ():
            validation_data (DataLoader):
            shuffle (bool):
            class_weight ():
            sample_weight ():
            initial_epoch ():
            steps_per_epoch ():
            validation_steps ():
            validation_batch_size ():
            validation_freq ():
            max_queue_size ():
            workers ():
            use_multiprocessing (bool):

        Returns:
            None
        """
        self.epochs = epochs
        self._callbacks = callbacks
        make_callbacks(self, callbacks)

        # Train
        print(f'Train on {len(train_data.dataset)} samples, validate on {len(validation_data.dataset)} samples')
        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}/{epochs}:')
            self.epoch = epoch
            self.train_once(train_data,
                            batch_size,
                            validation_data=validation_data,
                            validation_steps=validation_steps,
                            validation_batch_size=validation_batch_size)
            if epoch % validation_freq == 0 or epoch == epochs:
                self.valid_once(validation_data)

    def evaluate(self, imgs, dim=1):
        """evaluate

        Args:
            imgs:
            dim:

        Returns:
            Tensor
        """
        self.eval()
        target_pred = []
        with torch.no_grad():
            for img in imgs:
                y_pred = self(img.to(self.device))
                target_pred.append(torch.argmax(y_pred, dim=dim))
        return torch.cat(target_pred, dim=0)

    def evaluate_generator(self, test_loader, batch_size):
        """evaluate_generator

        Args:
            test_loader:
            batch_size:

        Returns:
            None
        """
        self.eval()
        correct = 0
        num_data = 0
        loss_mean = 0.0
        count_data = len(test_loader.dataset)
        count_batch = (count_data + batch_size - 1) // batch_size
        with torch.no_grad():
            for batch_idx, (data, y_true) in enumerate(test_loader):
                data, y_true = data.to(self.device), y_true.to(self.device)
                if data.shape[0] > 1:
                    y_pred = self(data)
                    loss = self.loss(y_pred, y_true).clone()

                    # 输出信息
                    num_data += len(data)
                    loss_mean = (loss_mean * batch_idx + loss) / (batch_idx + 1)
                    correct += self.get_correct(y_pred, y_true) if self.metrics else 0.

                self.msg.msg_out(loss_mean, 1.0 * correct / num_data, cur_batch=batch_idx + 1, count_batch=count_batch)
        print('')

    def train_once(
            self,
            train_loader,
            batch_size,
            validation_bool=None,
            validation_data=None,
            validation_steps=None,
            validation_batch_size=None,
    ):
        """train_once

        Args:
            train_loader:
            batch_size:
            validation_bool:
            validation_data:
            validation_steps:
            validation_batch_size:

        Returns:
            None
        """
        self.train()
        correct = 0
        num_data = 0
        loss_mean = 0.0
        count_data = len(train_loader.dataset)
        count_batch = (count_data + batch_size - 1) // batch_size
        for batch_idx, (data, y_true) in enumerate(train_loader):
            data, y_true = data.to(self.device), y_true.to(self.device)
            if data.shape[0] > 1:
                self.optimizer.zero_grad()
                y_pred = self(data)
                loss = self.loss(y_pred, y_true)
                loss.backward()
                self.optimizer.step()

                # 输出信息
                num_data += len(data)
                loss_mean = (loss_mean * batch_idx + loss) / (batch_idx + 1)
                correct += self.get_correct(y_pred, y_true) if self.metrics else 0.

            self.msg.msg_out(loss_mean, 1.0 * correct / num_data, cur_batch=batch_idx + 1, count_batch=count_batch)

            # 测试
            if validation_batch_size is not None and 0 == (batch_idx + 1) % validation_batch_size:
                self.valid_once(validation_data)

    def valid_once(self, valid_loader):
        """valid_once

        Args:
            valid_loader:

        Returns:
            None
        """
        self.eval()
        correct = 0
        loss_mean = 0.
        with torch.no_grad():
            for data, y_true in valid_loader:
                y_true = y_true.to(self.device)
                y_pred = self(data.to(self.device))
                loss_mean += self.loss(y_pred, y_true)
                correct += self.get_correct(y_pred, y_true) if self.metrics else 0.

        # 输出信息
        loss_mean /= len(valid_loader.dataset)
        val_acc = 1. * correct / len(valid_loader.dataset)
        self.msg.msg_out(loss_mean, val_acc, is_wrap=True, prefix='val_')

        # checkpoint
        if self.checkpoint_fn is not None:
            self.checkpoint_fn(self, accuracy=val_acc, loss=loss_mean)

        return loss_mean, val_acc

    def get_optimizer(self, optimizer='adam', call_params=None):
        if isinstance(optimizer, optim.Optimizer):
            return optimizer

        optimizer = optimizer.lower()
        params_list = dict()
        if 'adam' == optimizer:  # adjust beta1 to momentum
            call_params = self.check_call_params(call_params, lr_default=1e-3)
            params_list = self.get_parameters()
            self.optimizer = optim.Adam(self.parameters(), **call_params)
        elif 'adamw' == optimizer:
            call_params = self.check_call_params(call_params, lr_default=1e-3)
            params_list = self.get_parameters()
            self.optimizer = optim.AdamW(self.parameters(), **call_params)
        elif 'sgd' == optimizer:
            call_params = self.check_call_params(call_params, lr_default=1e-2)
            self.optimizer = optim.SGD(self.parameters(), **call_params)
        else:
            call_params = self.check_call_params(call_params, lr_default=1e-2)
            self.optimizer = optim.ASGD(self.parameters(), **call_params)

        self.del_parameters(params_list)
        return self.optimizer

    def get_parameters(self):
        opg_bn, opg_weight, opg_bias = [], [], []  # optimizer parameter groups
        for layer in self.get_modules():
            if hasattr(layer, 'bias') and isinstance(layer.bias, nn.Parameter):
                opg_bias.append(layer.bias)  # biases
            if isinstance(layer, nn.BatchNorm2d):
                opg_bn.append(layer.weight)  # no decay
            elif hasattr(layer, 'weight') and isinstance(layer.weight, nn.Parameter):
                opg_weight.append(layer.weight)  # apply decay
        return [{'params': opg_bn}, {'params': opg_weight}, {'params': opg_bias}]

    @staticmethod
    def del_parameters(params_list):
        for params in params_list:
            del params['params']

    @staticmethod
    def check_call_params(call_params=None, lr_default=1e-2):
        if call_params is None:
            call_params = {'lr': lr_default}
        elif 'lr' not in call_params:
            call_params['lr'] = lr_default
        return call_params

    def save(
            self, filepath,
            save_weights_only=False,
            save_best_only=False,
            pickle_module=pickle,
            pickle_protocol=DEFAULT_PROTOCOL,
            _use_new_zipfile_serialization=True,
            **kwargs
    ):
        """
            checkpoint.yaml
            model_name_weights.pt.last
            model_name_weights.pt.best
        """
        # Parameters
        self._save_weights_only = save_weights_only
        self._save_best_only = save_best_only

        # Save last, best
        if save_weights_only:  # parameters
            f = self.state_dict()
            mode = ModelCheckpoint.Weight
        else:  # model and parameters
            f = self
            mode = ModelCheckpoint.Model

        if not save_best_only:
            mode += f'_{self.epoch}-of-{self.epochs}'
        else:
            mode += f'_{ModelCheckpoint.Best}'

        best = kwargs['best'] if 'best' in kwargs else False
        if save_best_only and best:
            filename_best = ModelCheckpoint.get_suffix(filepath, mode=mode)
            torch.save(f, filename_best, pickle_module, pickle_protocol, _use_new_zipfile_serialization)

    @staticmethod
    def get_channels(channels, in_shape, down_size=32, channels_bias=0, shape_bias=0):
        """get_channels

        Args:
            channels:
            in_shape:
            down_size (Union[int, tuple]):
            channels_bias (Union[int, tuple]):
            shape_bias (Union[int, tuple]):

        Returns:
            channels
        """
        return channels * ((torch.Tensor(in_shape).int() + down_size - 1 - shape_bias) //
                           down_size - channels_bias).prod()

    @staticmethod
    def get_roi_size(roi_size, in_shape, down_size=32, channels_bias=0, shape_bias=0):
        output_size: torch.Tensor = (torch.Tensor(in_shape).int() + down_size - 1 - shape_bias) // \
                                    down_size - channels_bias
        if roi_size is None:
            roi_size = output_size
        else:
            roi_size = torch.Tensor(roi_size).int()
        return output_size.minimum(roi_size)

    def load_weights(self, file_weight, mode='weight', map_location=None, pickle_module=pickle):
        """load_weights

        Args:
            file_weight (str):
            mode:
            pickle_module:
            map_location:
        """
        is_exist = os.path.exists(file_weight)
        if is_exist:
            if 'weight' == mode:
                state_dict = torch.load(file_weight, map_location=map_location, pickle_module=pickle_module)
                self.load_state_dict(state_dict)
            elif 'model' == mode:
                state_dict = torch.load(file_weight, map_location=map_location, pickle_module=pickle_module)
                state_dict = state_dict.state_dict()
                self.load_state_dict(state_dict)
        return is_exist

    @staticmethod
    def get_correct(y_pred, y_true):
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        return pred.eq(y_true.view_as(pred)).sum().item()
