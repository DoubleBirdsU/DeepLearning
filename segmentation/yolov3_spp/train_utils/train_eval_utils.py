import math
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

import train_utils.distributed_utils as utils
from train_utils.coco_eval import CocoEvaluator
from train_utils.coco_utils import get_coco_api_from_dataset
from utils.utils import compute_loss, non_max_suppression, scale_coords


class Trainer(object):
    def __init__(self, model, optimizer, loss, loss_param_fun=None, callbacks=None):
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
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.loss_param_fun = loss_param_fun
        self.callbacks = callbacks

    def train_once(self, data_loader, device, epoch, print_freq,
                   multi_scale=False, img_size=(512, 512),
                   grid_min=8, grid_max=32, grid_size=32,
                   random_size=64, warmup=False):
        r"""train_once: 训练模型

        Args:
            data_loader:
            device:
            epoch:
            print_freq:
            multi_scale:
            img_size:
            grid_min:
            grid_max:
            grid_size:
            random_size:
            warmup:

        Returns:
            None
        """
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        lr_scheduler = None
        if epoch == 0 and warmup:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = utils.warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)
            random_size = 1

        enable_amp = 'cuda' in device.type
        scale = amp.GradScaler(enabled=enable_amp)

        lr_now = 0.
        loss_mean = torch.zeros(4).to(device)  # mean losses
        batch_size = len(data_loader)  # number of batches
        for i, (images, targets, paths, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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
                pred = self.model(images)

                # loss: compute_loss
                loss_dict = self.loss(pred, targets)

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

    def evaluate(self, data_loader, coco=None, device=None):
        return None

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


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq, accumulate, img_size,
                    grid_min, grid_max, gs,
                    multi_scale=False, warmup=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        accumulate = 1

    enable_amp = True if "cuda" in device.type else False
    scaler = amp.GradScaler(enabled=enable_amp)

    mloss = torch.zeros(4).to(device)  # mean losses
    now_lr = 0.
    nb = len(data_loader)  # number of batches
    # imgs: [batch_size, 3, img_size, img_size]
    # targets: [num_obj, 6] , that number 6 means -> (img_index, obj_index, x, y, w, h)
    # paths: list of img path
    for i, (imgs, targets, paths, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # ni 统计从epoch0开始的所有batch数
        ni = i + nb * epoch  # number integrated batches (since train start)
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)

        # Multi-Scale
        if multi_scale:
            # 每训练64张图片，就随机修改一次输入图片大小，
            # 由于label已转为相对坐标，故缩放图片不影响label的值
            if ni % accumulate == 0:  #  adjust img_size (67% - 150%) every 1 batch
                # 在给定最大最小输入尺寸范围内随机选取一个size(size为32的整数倍)
                img_size = random.randrange(grid_min, grid_max + 1) * gs
            sf = img_size / max(imgs.shape[2:])  # scale factor

            # 如果图片最大边长不等于img_size, 则缩放图片，并将长和宽调整到32的整数倍
            if sf != 1:
                # gs: (pixels) grid size
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with amp.autocast(enabled=enable_amp):
            pred = model(imgs)

            # loss
            loss_dict = compute_loss(pred, targets, model)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purpose
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_items = torch.cat((loss_dict_reduced["box_loss"],
                                    loss_dict_reduced["obj_loss"],
                                    loss_dict_reduced["class_loss"],
                                    losses_reduced)).detach()
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses

            if not torch.isfinite(losses_reduced):
                print('WARNING: non-finite loss, ending training ', loss_dict_reduced)
                print("training image path: {}".format(",".join(paths)))
                sys.exit(1)

            losses *= 1. / accumulate  # scale loss

        # backward
        scaler.scale(losses).backward()
        # optimize
        # 每训练64张图片更新一次权重
        if ni % accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

        if ni % accumulate == 0 and lr_scheduler is not None:  # 第一轮使用warmup训练方式
            optimizer.step()
            lr_scheduler.step()

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, coco=None, device=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    if coco is None:
        coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for imgs, targets, paths, shapes, img_index in metric_logger.log_every(data_loader, 100, header):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        # targets = targets.to(device)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        pred = model(imgs)[0]  # only get inference result
        pred = non_max_suppression(pred, conf_thres=0.001, iou_thres=0.6, multi_label=False)
        outputs = []
        for index, p in enumerate(pred):
            if p is None:
                p = torch.empty((0, 6), device=cpu_device)
                boxes = torch.empty((0, 4), device=cpu_device)
            else:
                # xmin, ymin, xmax, ymax
                boxes = p[:, :4]
                # shapes: (h0, w0), ((h / h0, w / w0), pad)
                # 将boxes信息还原回原图尺度，这样计算的mAP才是准确的
                boxes = scale_coords(imgs[index].shape[1:], boxes, shapes[index][0]).round()

            # 注意这里传入的boxes格式必须是xmin, ymin, xmax, ymax，且为绝对坐标
            info = {"boxes": boxes.to(cpu_device),
                    "labels": p[:, 5].to(device=cpu_device, dtype=torch.int64),
                    "scores": p[:, 4].to(cpu_device)}
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


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
