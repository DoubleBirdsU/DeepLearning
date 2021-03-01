import glob
import math
import os
import random
import shutil
import subprocess
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision

from tqdm import tqdm
from copy import copy
from pathlib import Path
from sys import platform
from base import torch_utils  # , google_utils
from base.base_model import base_model


# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def check_git_status():
    if platform in ['linux', 'darwin']:
        # Suggest 'git pull' if repo is out of date
        s = subprocess.check_output('if [ -d .git ]; then git fetch && git status -uno; fi', shell=True).decode('utf-8')
        if 'Your branch is behind' in s:
            print(s[s.find('Your branch is behind'):s.find('\n\n')] + '\n')


def check_file(file):
    # Searches for file if not found locally
    if os.path.isfile(file):
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]  # return first file if multiple found


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def ltrb2xywh(ltrb, dim=-1):
    r"""ltrb2xywh
        left-top Coordinate System

    Args:
        ltrb: [left, top, right, bottom], e.g. left <= right, top < bottom
        dim:

    Returns:
        xywh: [center_x, center_y, weight, high]
    """
    if dim < 0:
        dim += ltrb.ndim
    if 0 < dim < ltrb.ndim:
        dims = [dim] + [i for i in range(ltrb.ndim) if i != dim]
        ltrb = ltrb.permute(dims) if isinstance(ltrb, torch.Tensor) else ltrb.transpose(dims)
    xywh = torch.zeros_like(ltrb) if isinstance(ltrb, torch.Tensor) else np.zeros_like(ltrb)
    xywh[:2] = (ltrb[2:] + ltrb[:2]) / 2  # center [x, y]
    xywh[2:] = ltrb[2:] - ltrb[:2]  # [width, height]
    if 0 < dim < xywh.ndim:
        dims = [i for i in range(1, xywh.ndim)]
        dims.insert(dim, 0)
        xywh = xywh.permute(dims) if isinstance(xywh, torch.Tensor) else xywh.transpose(dims)
    return xywh


def xywh2ltrb(xywh, dim=-1):
    r"""xywh2ltrb
        left-top Coordinate System

    Args:
        xywh: [center_x, center_y, weight, high]
        dim:

    Returns:
        ltrb: [left, top, right, bottom]
    """
    if dim < 0:
        dim += xywh.ndim
    if 0 < dim < xywh.ndim:
        dims = [dim] + [i for i in range(xywh.ndim) if i != dim]
        xywh = xywh.permute(dims) if isinstance(xywh, torch.Tensor) else xywh.transpose(dims)
    ltrb = torch.zeros_like(xywh) if isinstance(xywh, torch.Tensor) else np.zeros_like(xywh)
    ltrb[:2] = xywh[:2] - xywh[2:] / 2.  # left top
    ltrb[2:] = xywh[:2] + xywh[2:] / 2.  # right bottom
    if 0 < dim < ltrb.ndim:
        dims = [i for i in range(1, ltrb.ndim)]
        dims.insert(dim, 0)
        ltrb = ltrb.permute(dims) if isinstance(ltrb, torch.Tensor) else ltrb.transpose(dims)

    return ltrb


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将预测的坐标信息转换回原图尺度
    :param img1_shape: 缩放后的图像尺度
    :param coords: 预测的box信息
    :param img0_shape: 缩放前的图像尺度
    :param ratio_pad: 缩放过程中的缩放比例以及pad
    :return:
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def bbox_iou(box1, box2, ltrb=True, iou_type='IoU', offset_iou='valid', delta=1e-9):
    """bbox_iou
        left-top Coordinate System.

        xywh = [center_x, center_y, weight, height]

        ltrb = [left, top, right, bottom], where left <= right, top <= bottom.

    Args:
        box1: box1 position description, xywh or ltrb.
        box2: box2 position description, as above.
        ltrb: bool, box position data is type of 'ltrb', default True.
        iou_type: 'IoU', 'GIoU', 'DIoU', 'CIoU', it isn't sensitive to upper or lower, default 'IoU'.
        offset_iou: when iou_type = 'IoU' offset_iou is valid. 'valid', 'first', 'last'
        delta: precision or disturbance, default 1e-16

    Returns:
        iou
    """
    # Get the coordinates of bounding boxes
    lt1, rb1 = init_boxes(box1 if ltrb else xywh2ltrb(box1))  # lt point, rb point
    lt2, rb2 = init_boxes(box2 if ltrb else xywh2ltrb(box2))  # lt point, rb point
    wh1, wh2 = rb1 - lt1, rb2 - lt2  # weight, height

    # Intersection area
    inter = ((torch.minimum(rb1, rb2) - torch.maximum(lt1, lt2)).clamp(0)).prod(0)

    # Union Area, union = (s1 + s2 - inter) + delta, delta = 1e-16
    union = wh1.prod(0) + wh2.prod(0) - inter + delta

    iou = inter / union  # iou
    iou_type = iou_type.upper()  # upper
    if iou_type != 'IOU':
        # convex width, height (smallest enclosing box)
        convex_wh = torch.max(rb1, rb2) - torch.min(lt1, lt2)
        if iou_type == 'GIOU':  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            convex_area = convex_wh.prod(0) + delta  # convex area
            iou -= (convex_area - union) / convex_area  # GIoU
        elif iou_type in ['DIOU', 'CIOU']:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            convex = (convex_wh ** 2).sum(0) + delta
            # center point distance squared
            rho = (((lt2 + rb2) - (lt1 + rb1)) ** 2 / 4).sum(0)
            if iou_type == 'DIOU':
                iou -= rho / convex  # DIoU
            elif iou_type == 'CIOU':
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(wh2[0] / wh2[1]) -
                                                   torch.atan(wh1[0] / wh1[1]), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                iou -= (rho / convex + v * alpha)  # CIoU
    elif offset_iou != 'valid':
        iou = inter / (wh1.prod(0) if offset_iou == 'first' else wh2.prod(0))

    return iou


def init_boxes(box):
    if not isinstance(box, torch.Tensor):
        box = torch.Tensor(box)
        if len(box.shape) == 1:
            box = box.reshape([-1, 1])
    return [box[:2], box[2:]]


def box_iou(box_lhs, box_rhs):
    """box_iou
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (l, t, r, b) format.

    Arguments:
        box_lhs (Tensor[N, 4])
        box_rhs (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    def box_area(box):  # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area_l = box_area(box_lhs.T)
    area_r = box_area(box_rhs.T)

    inter = (torch.min(box_lhs[:, None, 2:], box_rhs[:, 2:]) -
             torch.max(box_lhs[:, None, :2], box_rhs[:, :2])).clamp(0).prod(2)
    return inter / (area_l[:, None] + area_r - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh_lhs, wh_rhs):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh_lhs = wh_lhs[:, None]  # [N,1,2]
    wh_rhs = wh_rhs[None]  # [1,M,2]
    inter = torch.min(wh_lhs, wh_rhs).prod(2)  # [N,M]
    return inter / (wh_lhs.prod(2) + wh_rhs.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression_v2(prediction, conf_threshold=0.1, iou_threshold=0.6,
                           classes=None, agnostic=False, labels=(), multi_label=True, max_det=300):
    r"""Performs  Non-Maximum Suppression on inference results

    Args:
         prediction: [batch, num_anchors, num_y, num_x, (4 + 1 + num_cls)]
         conf_threshold: confidence threshold
         iou_threshold:
         classes:
         agnostic:
         labels:
         multi_label:
         max_det:

    Returns:
        detections with shape:

        nx6={l, t, r, b, conf, cls}
    """

    # Settings
    min_wh, max_wh = 2, 4096  # 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    merge = False  # merge for best mAP

    # parameters
    num_cls = prediction[0].shape[1] - 5  # number of classes
    mask_cls = prediction[..., 4] > conf_threshold  # confidence
    multi_label &= num_cls > 1  # multiple labels per box

    time_before = time.time()
    output = [None] * prediction.shape[0]
    for img_idx, img in enumerate(prediction):  # image index, image inference 遍历每张图片
        # Apply constraints
        img = img[mask_cls[img_idx]]  # confidence 根据obj confidence虑除背景目标
        # img = img[((img[..., 2:4] > min_wh) & (img[..., 2:4] < max_wh)).all(1)]  # width-height 虑除小目标

        # Cat apriori labels if auto-labelling
        if labels and len(labels[img_idx]):
            label = labels[img_idx]
            info_vec = torch.zeros((len(label), num_cls + 5), device=img.device)
            info_vec[..., :4] = label[..., 1:5]  # box
            info_vec[..., 4] = 1.0  # conf
            info_vec[range(len(label)), label[..., 0].long() + 5] = 1.0  # cls
            img = torch.cat((img, info_vec), dim=0)

        # If none remain process next image
        if not img.shape[0]:
            continue

        # Compute conf
        img[..., 5:] *= img[..., 4:5]  # conf = cls_conf * obj_conf

        # Box (x_center, y_center, width, height) to (l, t, r, b)
        box = xywh2ltrb(img[..., :4], dim=-1)

        # Detections matrix nx6 (ltrb, conf, cls)
        if multi_label:  # 针对每个类别执行非极大值抑制
            idx, jdx = (img[..., 5:] > conf_threshold).nonzero(as_tuple=False).T
            img = torch.cat((box[idx], img[idx, jdx + 5].unsqueeze(1), jdx.float().unsqueeze(1)), 1)
        else:  # best class only  直接针对每个类别中概率最大的类别进行非极大值抑制处理
            conf, jdx = img[..., 5:].max(1, keepdim=True)
            img = torch.cat((box, conf, jdx.float()), 1)[conf.view(-1) > conf_threshold]

        # Filter by class
        if classes:
            img = img[(img[..., 5:6] == torch.tensor(classes, device=img.device)).any(1)]

        # If none remain process next image
        num_boxes = img.shape[0]  # number of boxes
        if not num_boxes:
            continue

        # Batched NMS
        box_cls = img[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = img[..., :4].clone() + box_cls, img[..., 4]  # boxes (offset by class), scores
        idx = torchvision.ops.boxes.nms(boxes, scores, iou_threshold)
        idx = idx[:max_det]  # 最多只保留前max_num个目标信息
        if merge and (1 < num_boxes < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(idx,4) = weights(idx,num_boxes) * boxes(num_boxes,4)
                iou = box_iou(boxes[idx], boxes) > iou_threshold  # iou matrix
                weights = iou * scores[None]  # box weights
                img[idx, :4] = torch.mm(weights, img[..., :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    idx = idx[iou.sum(1) > 1]  # require redundancy
            except Exception:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(img, idx, img.shape, idx.shape)
                pass

        output[img_idx] = img[idx]
        if (time.time() - time_before) > time_limit:
            break  # time limit exceeded

    return output


class FocalLoss(base_model):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6,
                        multi_label=True, classes=None, agnostic=False, max_num=100):
    """
    Performs  Non-Maximum Suppression on inference results

    param: prediction[batch, num_anchors, (num_classes+1+4) x num_anchors]
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    merge = False  # merge for best mAP
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference 遍历每张图片
        # Apply constraints
        x = x[x[:, 4] > conf_thres]  # confidence 根据obj confidence虑除背景目标
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height 虑除小目标

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2ltrb(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:  # 针对每个类别执行非极大值抑制
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only  直接针对每个类别中概率最大的类别进行非极大值抑制处理
            conf, j = x[:, 5:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]

        # Filter by class
        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        i = i[:max_num]  # 最多只保留前max_num个目标信息
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                # i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def print_model_biases(model):
    # prints the bias neurons preceding each yolo layer
    print('\nModel Bias Summary: %8s%18s%18s%18s' % ('layer', 'regression', 'objectness', 'classification'))
    try:
        multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
        for layer in model.yolo_layers:  # print pretrained biases
            if multi_gpu:
                na = model.module.module_list[layer].na  # number of anchors
                b = model.module.module_list[layer - 1][0].bias.view(na, -1)  # bias 3x85
            else:
                na = model.module_list[layer].na
                b = model.module_list[layer - 1][0].bias.view(na, -1)  # bias 3x85
            print(' ' * 20 + '%8g %18s%18s%18s' % (layer, '%5.2f+/-%-5.2f' % (b[:, :4].mean(), b[:, :4].std()),
                                                   '%5.2f+/-%-5.2f' % (b[:, 4].mean(), b[:, 4].std()),
                                                   '%5.2f+/-%-5.2f' % (b[:, 5:].mean(), b[:, 5:].std())))
    except:
        pass


def strip_optimizer(f='weights/best.pt'):  # from utils.utils import *; strip_optimizer()
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    print('Optimizer stripped from %s' % f)
    torch.save(x, f)


def create_backbone(f='weights/best.pt'):  # from utils.utils import *; create_backbone()
    # create a backbone from a *.pt file
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    for p in x['model'].parameters():
        p.requires_grad = True
    s = 'weights/backbone.pt'
    print('%s saved as %s' % (f, s))
    torch.save(x, s)


def coco_class_count(path='../coco/labels/train2014/'):
    # Histogram of occurrences per class
    nc = 80  # number classes
    x = np.zeros(nc, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nc)
        print(i, len(files))


def coco_only_people(path='../coco/labels/train2017/'):  # from utils.utils import *; coco_only_people()
    # Find images with only people
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def crop_images_random(path='../images/', scale=0.50):  # from utils.utils import *; crop_images_random()
    # crops images into random squares up to scale fraction
    # WARNING: overwrites images!
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        img = cv2.imread(file)  # BGR
        if img is not None:
            height, width = img.shape[:2]

            # create random mask
            min_size = 30  # minimum size (pixels)
            mask_height = random.randint(min_size, int(max(min_size, height * scale)))  # mask height
            mask_width = mask_height  # mask width

            # box
            box_left = max(0, random.randint(0, width) - mask_width // 2)
            box_top = max(0, random.randint(0, height) - mask_height // 2)
            box_right = min(width, box_left + mask_width)
            box_bottom = min(height, box_top + mask_height)

            # apply random color mask
            cv2.imwrite(file, img[box_top:box_bottom, box_left:box_right])


def coco_single_class_labels(path='../coco/labels/train2014/', label_class=43):
    # Makes single-class coco datasets. from utils.utils import *; coco_single_class_labels()
    if os.path.exists('new/'):
        shutil.rmtree('new/')  # delete output folder
    os.makedirs('new/')  # make new output folder
    os.makedirs('new/labels/')
    os.makedirs('new/images/')
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        with open(file, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
        i = labels[:, 0] == label_class
        if any(i):
            img_file = file.replace('labels', 'images').replace('txt', 'jpg')
            labels[:, 0] = 0  # reset class to 0
            with open('new/images.txt', 'a') as f:  # add image to dataset list
                f.write(img_file + '\n')
            with open('new/labels/' + Path(file).name, 'a') as f:  # write label
                for l in labels[i]:
                    f.write('%g %.6f %.6f %.6f %.6f\n' % tuple(l))
            shutil.copyfile(src=img_file, dst='new/images/' + Path(file).name.replace('txt', 'jpg'))  # copy images


def kmean_anchors(path='./data/coco64.txt', n=9, img_size=(640, 640), thr=0.20, gen=1000):
    """ Creates kmeans anchors for use in *.cfg files: from utils.utils import *; _ = kmean_anchors()

    Param:
        n: number of anchors
        img_size: (min, max) image size used for multi-scale training (can be same values)
        thr: IoU threshold hyper parameter used for training (0.0 - 1.0)
        gen: generations to evolve anchors using genetic algorithm

    Returns:

    """
    from utils.datasets import LoadImageAndLabels

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        iou = wh_iou(wh, torch.Tensor(k))
        max_iou = iou.max(1)[0]
        bpr, aat = (max_iou > thr).float().mean(), (iou > thr).float().mean() * n  # best possible recall, anch > thr
        print('%.2f iou_thr: %.3f best possible recall, %.2f anchors > thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, IoU_all=%.3f/%.3f-mean/best, IoU>thr=%.3f-mean: ' %
              (n, img_size, iou.mean(), max_iou.mean(), iou[iou > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    def fit_ness(k):  # mutation fitness
        iou = wh_iou(wh, torch.Tensor(k))  # iou
        max_iou = iou.max(1)[0]
        return (max_iou * (max_iou > thr).float()).mean()  # product

    # Get label wh
    wh = []
    dataset = LoadImageAndLabels(path, augment=True, rect=True)
    num_rep = 1 if img_size[0] == img_size[1] else 10  # number augmentation repetitions
    for sigma, l in zip(dataset.shapes, dataset.labels):
        wh.append(l[:, 3:5] * (sigma / sigma.max()))  # image normalized to letterbox normalized wh
    wh = np.concatenate(wh, 0).repeat(num_rep, axis=0)  # augment 10x
    wh *= np.random.uniform(img_size[0], img_size[1], size=(wh.shape[0], 1))  # normalized to pixels (multi-scale)
    wh = wh[(wh > 2.0).all(1)]  # remove below threshold boxes (< 2 pixels wh)

    # Kmeans calculation
    from scipy.cluster.vq import kmeans
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    sigma = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / sigma, n, iter=30)  # points, mean distance
    k *= sigma
    wh = torch.Tensor(wh)
    k = print_results(k)

    # Evolve
    npr = np.random
    f, shape, mp, sigma = fit_ness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    for _ in tqdm(range(gen), desc='Evolving anchors'):
        v = np.ones(shape)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(shape) < mp) * npr.random() * npr.randn(*shape) * sigma + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fit_ness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            print_results(k)
    k = print_results(k)

    return k


def print_mutation(hyp, results, bucket=''):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        os.system('gsutil cp gs://%s/evolve.txt .' % bucket)  # download evolve.txt

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    np.savetxt('evolve.txt', x[np.argsort(-fitness(x))], '%10.3g')  # save sort by fitness

    if bucket:
        os.system('gsutil cp evolve.txt gs://%s' % bucket)  # upload evolve.txt


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = ltrb2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2ltrb(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    w = [0.0, 0.01, 0.99, 0.00]  # weights for [P, R, mAP, F1]@0.5 or [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def output_to_target(output, width, height):
    """
    Convert a YOLO model output to target format
    [batch_id, class_id, x, y, w, h, conf]
    """
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                box = pred[:4]
                w = (box[2] - box[0]) / width
                h = (box[3] - box[1]) / height
                x = box[0] / width + w / 2
                y = box[1] / height + h / 2
                conf = pred[4]
                cls = int(pred[5])

                targets.append([i, cls, x, y, w, h, conf])

    return np.array(targets)


# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_wh_methods():  # from utils.utils import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x, ya, '.-', label='yolo method')
    plt.plot(x, yb ** 2, '.-', label='^2 power method')
    plt.plot(x, yb ** 2.5, '.-', label='^2.5 power method')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.legend()
    fig.tight_layout()
    fig.savefig('comparison.png', dpi=200)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    """plot_images 绘制边框

    [extended_summary]

    Args:
        images (None): [description]
        targets (None): [description]
        paths (None, optional): [description]. Defaults to None.
        fname (str, optional): [description]. Defaults to 'images.jpg'.
        names (None, optional): [description]. Defaults to None.
        max_size (int, optional): [description]. Defaults to 640.
        max_subplots (int, optional): [description]. Defaults to 16.

    Returns:
        None: [description]
    """
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    if os.path.isfile(fname):  # do not overwrite
        return None

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2ltrb(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            gt = image_targets.shape[1] == 6  # ground truth if no conf column
            conf = None if gt else image_targets[:, 6]  # check for confidence presence (gt vs pred)

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname is not None:
        mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic


def plot_lr_scheduler(optimizer, scheduler, epochs=300):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.tight_layout()
    plt.savefig('LR.png', dpi=200)


def plot_test_txt():  # from utils.utils import *; plot_test()
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = ltrb2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.utils import *; plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_labels(labels):
    # plot dataset labels
    cls, boxes = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes

    def hist2d(x, y, n=100):
        x_edges, y_edges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
        hist, x_edges, y_edges = np.histogram2d(x, y, (x_edges, y_edges))
        x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, hist.shape[0] - 1)
        y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, hist.shape[1] - 1)
        return hist[x_idx, y_idx]

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    ax[0].hist(cls, bins=int(cls.max() + 1))
    ax[0].set_xlabel('classes')
    ax[1].scatter(boxes[0], boxes[1], c=hist2d(boxes[0], boxes[1], 90), cmap='jet')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[2].scatter(boxes[2], boxes[3], c=hist2d(boxes[2], boxes[3], 90), cmap='jet')
    ax[2].set_xlabel('width')
    ax[2].set_ylabel('height')
    plt.savefig('labels.png', dpi=200)


def plot_evolution_results(hyp):  # from utils.utils import *; plot_evolution_results(hyp)
    # Plot hyperparameter evolution results in evolve.txt
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    # weights = (f - f.min()) ** 2  # for weighted results
    fig = plt.figure(figsize=(12, 10), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(4, 5, i + 1)
        plt.plot(mu, f.max(), 'o', markersize=10)
        plt.plot(y, f, '.')
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)


def plot_results_overlay(start=0, stop=0):  # from utils.utils import *; plot_results_overlay()
    # Plot training results files 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val', 'val', 'Recall', 'F1']  # legends
    t = ['GIoU', 'Objectness', 'Classification', 'P-R', 'mAP-F1']  # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                if i in [0, 1, 2]:
                    y[y == 0] = np.nan  # dont show zero loss values
                ax[i].plot(x, y, marker='.', label=s[j])
            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_results(start=0, stop=0, bucket='', id=()):  # from utils.utils import *; plot_results()
    # Plot training 'results*.txt' as seen in https://github.com/ultralytics/yolov3#training
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'F1']
    if bucket:
        os.system('rm -rf storage.googleapis.com')
        files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
    else:
        files = glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')
    for f in sorted(files):
        try:
            results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
            num_row = results.shape[1]  # number of rows
            x = range(start, min(stop, num_row) if stop else num_row)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # dont show zero loss values
                    # y /= y[0]  # normalize
                ax[i].plot(x, y, marker='.', label=Path(f).stem, linewidth=2, markersize=8)
                ax[i].set_title(s[i])
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except:
            print('Warning: Plotting error for %s, skipping file' % f)

    ax[1].legend()
    fig.savefig('results.png', dpi=200)


def print_cover(*arg, **kwargs):
    if 'flush' in kwargs:
        kwargs.pop('flush')
    if 'end' in kwargs:
        kwargs.pop('end')
    print('\r', *arg, **kwargs, end='', flush=True)


class ModelPath(object):
    logs_path = 'logs'
    for i in range(7):
        if os.path.exists(logs_path):
            logs_path = logs_path + os.sep
            break
        logs_path = '..' + os.sep + logs_path

    models_path = logs_path + 'models' + os.sep
    fit_path = logs_path + 'fit' + os.sep
    model_type = '.ckpt'

    def __init__(self, root=None):
        self.root = root

    # 生成路径
    @staticmethod
    def generateFolderPath(folder, path):
        """
        generateFolderPath 生成文件夹路径

        [extended_summary]

        Args:
            folder (str): 文件夹名
            path (str): 路径

        Returns:
            str: 文件夹路径
        """
        return path + os.sep + folder + os.sep

    # 生成文件名
    @staticmethod
    def generateName(file_name, filetype, path=None):
        """
        generateName 生成名称

        `[extended_summary]`

        Args:
            file_name (str): 文件名
            filetype (str): 文件类型
            path (str, optional): 文件路径. Defaults to None.

        Returns:
            str: 文件路径
        """
        return path + file_name + filetype

    # 生成模型路径
    def generateModelName(self, model_name, filetype=model_type):
        """
        generateModelName 生成模型路径

        [extended_summary]

        Args:
            model_name (str): 模型名成
            filetype (str, optional): 模型类型. Defaults to model_type.

        Returns:
            str: 模型路径
        """
        return self.generateName(model_name + '_model', filetype, self.generateFolderPath(model_name, ))

    def generateWeightName(self, model_name, filetype='.txt'):
        """
        generateWeightName 生成权重路径名

        [extended_summary]

        Args:
            model_name (str): 模型名称
            filetype (str, optional): 文件类型. Defaults to '.txt'.

        Returns:
            str: 权重路径
        """
        return self.generateName(model_name + '_weights', filetype, self.generateFolderPath(model_name, ))
