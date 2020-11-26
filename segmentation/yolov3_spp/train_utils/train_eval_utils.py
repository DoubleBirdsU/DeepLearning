import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

import train_utils.distributed_utils as utils
from train_utils.coco_eval import CocoEvaluator
from train_utils.coco_utils import get_coco_api_from_dataset
from utils.utils import non_max_suppression, scale_coordinates


class Trainer(object):
    def __init__(self, model, optimizer, loss, lr_lambda, last_epoch, loss_param_fun=None, callbacks=None):
        """Trainer
            模型训练器

        Args:
            model: Module
            optimizer: 优化器, Adam, SGD,...
            loss: 损失函数
            loss_param_fun: 损失函数第三参数生成函数
            callbacks: 无参函数集

        Returns:
            None
        """
        self.log_dir = './logs'
        self.weights_dir = './weights'
        self.results_file = os.path.join(self.weights_dir, 'results.txt')
        self._check_dir([self.log_dir, self.weights_dir],
                        [self.results_file])

        # model
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.loss_param_fun = loss_param_fun
        self.callbacks = callbacks
        self.lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        self.lr_scheduler.last_epoch = last_epoch
        self.tb_writer = SummaryWriter(comment=os.path.join(self.log_dir, 'summary.txt'))

    def fit_generate(self,
                     train_loader,
                     batch_size=0,
                     epochs=0,
                     test_loader=None,
                     callbacks=None,
                     print_freq=1,
                     save_best=False,
                     multi_scale=False,
                     img_size=(512, 512),
                     grid_min=None,
                     grid_max=None,
                     grid_size=32,
                     random_size=64,
                     device=torch.device('cuda'),
                     warmup=False):
        r"""fit_generate: 训练模型

        Args:
            train_loader:
            batch_size:
            epochs:
            test_loader:
            callbacks:
            print_freq:
            save_best:
            multi_scale:
            img_size:
            grid_min:
            grid_max:
            grid_size:
            random_size:
            device:
            warmup:

        Returns:
            None
        """
        opt_dict = self.optimizer.state_dict
        best_map = 0.0
        for epoch in range(epochs):
            loss_mean, lr = loss_mean, lr_now = self._train_one_epoch(train_loader, batch_size, epoch, print_freq,
                                                                      multi_scale, img_size, grid_min, grid_max,
                                                                      grid_size, random_size, device, warmup)

            # update lr_scheduler
            self.lr_scheduler.step()

            if test_loader and epoch == epochs - 1:
                # evaluate on the test dataset
                result_info = self.evaluate(self.model, test_loader, device=device)

                coco_mAP = result_info[0]
                voc_mAP = result_info[1]
                coco_mAR = result_info[8]

                # write into tensorboard
                if self.tb_writer:
                    tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
                            "mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]

                    for x, tag in zip(loss_mean.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
                        self.tb_writer.add_scalar(tag, x, epoch)

                # write into txt
                with open(self.results_file, "a") as f:
                    result_info = [str(round(i, 4)) for i in result_info]
                    txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                    f.write(txt + "\n")

                # update best mAP(IoU=0.50:0.95)
                if coco_mAP > best_map:
                    best_map = coco_mAP

                if save_best is False:
                    # save weights every epoch
                    with open(self.results_file, 'r') as f:
                        save_files = {
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'training_results': f.read(),
                            'epoch': epoch,
                            'best_map': best_map}
                        torch.save(save_files, "./weights/yolov3spp-{}.pt".format(epoch))
                else:
                    # only save best weights
                    best_file_name = os.path.join(self.weights_dir, 'model_weights_best.pt')
                    if best_map == coco_mAP:
                        with open(self.results_file, 'r') as f:
                            save_files = {
                                'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'training_results': f.read(),
                                'epoch': epoch,
                                'best_map': best_map}
                            torch.save(save_files, best_file_name.format(epoch))

            if test_loader:
                self.evaluate(self.model, test_loader)

        return self

    @torch.no_grad()
    def evaluate(self, data_loader, coco=None, device=None):
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        if not device:
            device = torch.device("cpu")
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test: "

        if coco is None:
            coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = _get_iou_types(self.model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

        log_every = metric_logger.log_every(data_loader, 100, header)
        for images, targets, paths, shapes, img_index in log_every:
            images = images.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

            # 当使用CPU时，跳过GPU相关指令
            if device != torch.device("cpu"):
                torch.cuda.synchronize(device)

            model_time = time.time()
            pred = self.model(images)[0]  # only get inference result
            pred = non_max_suppression(pred, conf_thres=0.001, iou_thres=0.6, multi_label=False)
            outputs = []
            for index, pred_i in enumerate(pred):
                if pred_i is None:
                    pred_i = torch.empty((0, 6), device=device)
                    boxes = torch.empty((0, 4), device=device)
                else:
                    boxes = pred_i[:, :4]  # l, t, r, b
                    # shapes: (h0, w0), ((h / h0, w / w0), pad)
                    # 将boxes信息还原回原图尺度，这样计算的mAP才是准确的
                    boxes = scale_coordinates(boxes, images[index].shape[1:], shapes[index]).round()

                image = images[index]
                self.img_show(image, boxes)
                # 注意这里传入的boxes格式必须是 l_abs, t_abs, r_abs, b_abs，且为绝对坐标
                info = {"boxes": boxes.to(device),
                        "labels": pred_i[:, 5].to(device=device, dtype=torch.int64),
                        "scores": pred_i[:, 4].to(device)}
                outputs.append(info)
            model_time = time.time() - model_time

            res = {img_id: output for img_id, output in zip(img_index, outputs)}

            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)

        result_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

        return result_info

    @staticmethod
    def random_size(images, img_size, rand_size=False, grid_min=32, grid_max=64, grid_size=32,
                    interpolate_mode='bilinear'):
        if rand_size:  #  adjust img_size (67% - 150%) every 1 batch
            # 在给定最大最小输入尺寸范围内随机选取一个size(size 为 grid_size 的整数倍)
            img_size = random.randrange(grid_min, grid_max + 1) * grid_size
        scale_factor = img_size / max(images.shape[2:])  # scale factor

        # 如果图片最大边长不等于img_size, 则缩放图片，并将长和宽调整到32的整数倍
        if scale_factor != 1:
            # new shape (stretched to 32-multiple)
            new_shape = [math.ceil(x * scale_factor / grid_size) * grid_size for x in images.shape[2:]]
            images = F.interpolate(images, size=new_shape, mode=interpolate_mode, align_corners=False)
        return images, img_size

    @staticmethod
    def img_show(img, boxes=None, channel='first_channel'):
        img = img.cpu().numpy() if isinstance(img, torch.Tensor) else img.clone()
        if 'first_channel' == channel:
            img = np.transpose(img, (1, 2, 0))

        if boxes is not None:
            for box in boxes:  # l, t, r, b
                Trainer.add_box(img, box, channel='last_channel')

        plt.imshow(img)
        plt.show()
        pass

    @staticmethod
    def add_box(img, box, color=(1., 0., 0.), channel='first_channel'):
        # box=(l, t, r, b)
        l, t, r, b = box.int()
        for i in range(3):
            img[t:b, l, i] = color[i]  # l
            img[t, l:r, i] = color[i]  # t
            img[t:b, r, i] = color[i]  # r
            img[b, l:r, i] = color[i]  # b

    def _train_one_epoch(
            self,
            train_loader,
            batch_size=0,
            epoch=0,
            print_freq=1,
            multi_scale=False,
            img_size=(512, 512),
            grid_min=None,
            grid_max=None,
            grid_size=32,
            random_size=64,
            device=torch.device('cuda'),
            warmup=False
    ):
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        lr_scheduler = None
        if epoch == 0 and warmup:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(train_loader) - 1)

            lr_scheduler = utils.warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)
            random_size = 1

        enable_amp = 'cuda' in device.type
        scale = amp.GradScaler(enabled=enable_amp)

        lr_now = 0.
        loss_mean = torch.zeros(4).to(device)  # mean losses
        batch_size = len(train_loader)  # number of batches
        for i, (images, targets, paths, _, _) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            # count_batch 统计从 epoch0 开始的所有 batch 数
            count_batch = i + batch_size * epoch  # number integrated batches (since train start)
            images = images.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Multi-Scale
            # 由于label已转为相对坐标，故缩放图片不影响label的值
            # 每训练64张图片，就随机修改一次输入图片大小
            if multi_scale:
                images, img_size = self.random_size(images, img_size, count_batch % random_size == 0,
                                                    grid_min, grid_max, grid_size)

            # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
            with amp.autocast(enabled=enable_amp):
                # loss: compute_loss
                loss_dict = self.loss(self.model(images), targets)

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purpose
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                loss_items = torch.cat((loss_dict_reduced["box_loss"],
                                        loss_dict_reduced["obj_loss"],
                                        loss_dict_reduced["class_loss"],
                                        losses_reduced)).detach()
                loss_mean = (loss_mean * i + loss_items) / (i + 1)  # update mean losses

                if not torch.isfinite(losses_reduced):
                    print('WARNING: non-finite loss, ending training ', loss_dict_reduced)
                    print("training image path: {}".format(",".join(paths)))
                    sys.exit(1)

                losses *= 1. / random_size  # scale loss

            # backward
            scale.scale(losses).backward()
            # optimize
            # 每训练64张图片更新一次权重
            if count_batch % random_size == 0:
                scale.step(self.optimizer)
                scale.update()
                self.optimizer.zero_grad()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            lr_now = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr_now)

            if count_batch % random_size == 0 and lr_scheduler is not None:  # 第一轮使用warmup训练方式
                self.optimizer.step()
                lr_scheduler.step()

        return loss_mean, lr_now

    @staticmethod
    def _check_dir(dir_list, file_list):
        for d in dir_list:
            if not os.path.exists(d):
                os.mkdir(d)
        for f in file_list:
            d = f[:f.rfind('/')]
            if not os.path.exists(d):
                os.mkdir(d)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
