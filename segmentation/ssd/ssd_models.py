from base.utils import xywh2ltrb, bbox_iou
from math import sqrt

import torch
import torch.nn as nn

from base.base_model import Module
from base.blocks import ResConvBlock, ResBlockB
from base.layers import FeatureExtractor
from base.model import ResNet, NNet

# ********************** SSD hyper-parameters begin ********************** #
g_num_anchors = [4, 6, 6, 6, 4, 4]
g_maps_size = [38, 19, 10, 5, 3, 1]
g_ratios = [1, 2, 1 / 2, 3, 1 / 3]
# *********************** SSD hyper-parameters end *********************** #


class SSDClassifier(NNet):
    def __init__(self, in_ch, num_cls, num_anchor):
        super(SSDClassifier, self).__init__()
        self.anchor_vector_len = num_cls + 4
        self.classifier = FeatureExtractor(in_ch, num_anchor * (num_cls + 4), 3, padding='same')
        self.addLayers(self.classifier)

    def forward(self, x):
        anchors = self.classifier(x)
        return anchors.view([x.shape[0], self.anchor_vector_len, -1])


class SSDBlock(NNet):
    def __init__(self, in_ch, out_ch, mid_ch, out_stride, num_cls, num_anchor=4, kernel_sizes=(1, 3), padding='same'):
        super(SSDBlock, self).__init__()
        self.num_cls = num_cls
        self.feb0 = FeatureExtractor(in_ch, mid_ch, kernel_sizes[0], padding=padding, bn=True, activation='relu')
        self.feb1 = FeatureExtractor(mid_ch, out_ch, kernel_sizes[1], out_stride, padding=padding, bn=True,
                                     activation='relu')
        self.classifier = SSDClassifier(out_ch, num_cls, num_anchor)
        self.addLayers([self.feb0, self.feb1, self.classifier])
        pass

    def forward(self, x):
        x = self.feb1(self.feb0(x))
        anchors = self.classifier(x)
        return x, anchors


class BackBone(NNet):
    def __init__(self, num_cls=21, img_size=(3, 300, 300), num_anchor=4):
        super(BackBone, self).__init__()
        self.resnet = ResNet(num_cls, img_size, num_res_block=4, include_top=False)
        in_ch, out_ch, mid_ch = self.resnet.in_chs[-1], self.resnet.out_chs[-1], self.resnet.mid_chs[-1]
        self.classifier = SSDClassifier(self.resnet.out_ch_last, num_cls, num_anchor)
        self.res_conv5 = ResConvBlock(in_ch, out_ch, mid_ch, 3, ResBlockB)

        # 调整步长
        conv4 = self.resnet.net[-1]
        block3 = conv4.blocks[-1]
        res2 = block3.res_list[-2]
        self.resnet.net[1].stride = 1  # MaxPool
        res2.Conv2d.stride = 1  # Conv4_3_2
        block3.shortcut.stride = 1  # Shortcut4_3

        self.addLayers([self.resnet, self.classifier, self.res_conv5])

    def forward(self, x):
        x = self.resnet(x)
        anchors = self.classifier(x)
        x = self.res_conv5(x)
        return x, anchors


# SSD
class SSD(NNet):
    def __init__(self, num_cls=21, img_size=(3, 300, 300)):
        super(SSD, self).__init__()
        self.ssd_blk_list = [
            BackBone(num_cls, img_size, num_anchor=g_num_anchors[0]),
            SSDBlock(2048, 1024, 1024, 1, num_cls, num_anchor=g_num_anchors[1], kernel_sizes=(3, 1)),
            SSDBlock(1024, 512, 256, 2, num_cls, num_anchor=g_num_anchors[2]),
            SSDBlock(512, 256, 128, 2, num_cls, num_anchor=g_num_anchors[3]),
            SSDBlock(256, 256, 128, 2, num_cls, num_anchor=g_num_anchors[4]),
            SSDBlock(256, 256, 128, 2, num_cls, num_anchor=g_num_anchors[5], padding='valid'),
        ]
        self.addLayers(self.ssd_blk_list)

    def forward(self, x):
        anchors = list()
        for anchor_layer in self.ssd_blk_list:
            x, anchor = anchor_layer(x)
            anchors.append(anchor)
        return torch.cat(anchors, dim=2)


class SSDLoss(Module):
    def __init__(self, threshold=0.5, scale_neg=3):
        super(SSDLoss, self).__init__()
        self.threshold = threshold
        self.scale_neg = scale_neg
        self.default_box = self.get_default_box()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        pass

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: {x, y, w, h, c0, c1, c2, ..., c20}, 0 <= ci <= 1
            y_true: {idx_img, c, x, y, w, h}

        Returns:
            None
        """
        pred_box = y_pred[:4]
        pred_label = y_pred[4:]

        # Pos
        pos_mask = self.get_pos(self.default_box, y_true)
        true_box = y_true[:4]
        true_label = y_true[4:]

        # loc loss
        g_hat_box = self.get_hat(true_box[pos_mask], self.default_box[pos_mask])
        loc_loss = self.smooth_l1(pred_box[pos_mask] - g_hat_box).sum()

        # conf loss
        conf_pos = pred_label[0][pos_mask] * true_label[pos_mask]

        neg_mask = torch.topk(pred_label[1:], pos_mask.sum() * self.scale_neg)
        conf_neg = pred_label[0][neg_mask]

        conf_loss = -self.log_softmax(conf_pos).sum() - self.log_softmax(conf_neg).sum()
        return loc_loss + conf_loss

    def get_default_box(
            self, m=6,
            scale_min=0.2,
            scale_max=0.9,
            ratios=None,
            num_anchors=None,
            maps_size=None,
    ):
        """get_default_box
            default_box: (cx, cy, w, h)

            return: dict {'xywh': xywh_boxes, 'ltrb': ltrb_boxes}

            s_k = s_min + (s_max - s_min) * (k - 1) / (m - 1), k = {1, 2, ..., m}

            w_k^a = s_k * sqrt(a_r), h_k^a = s_k / sqrt(a_r)

            s_k' = sqrt(s_k * s_{k + 1})

            c_x, c_y = [(i + 0.5)/size_k, (j + 0.5)/size_k]

        Args:
            m: m feature maps
            scale_min: the min scale of the default boxes.
            scale_max: the max scale of the default boxes.
            ratios: 可以对特殊的数据集进行统计分析, 设置针对性的比例. 同一算法中, 针对不同的尺寸的图像设置不同的比例.
            num_anchors: the number of anchor layers.
            maps_size: the size of feature maps.

        Returns:
            Dict
        """
        if ratios is None:
            ratios = g_ratios
        if num_anchors is None:
            num_anchors = g_num_anchors
        if maps_size is None:
            maps_size = g_maps_size

        scales_box = [self.get_scale(k, m, scale_min, scale_max) for k in range(1, m + 2)]
        scales_box_hat = [sqrt(scales_box[k] * scales_box[k + 1]) for k in range(m)]  # s'k
        scales_box = scales_box[:m]
        ratios_sqrt = [sqrt(ratio) for ratio in ratios]

        max_map_size = max(maps_size)
        feature_map_org = self._crate_default_box(max_map_size)

        xywh_boxes = list()
        for k in range(m):
            box = list()
            feature_map = feature_map_org[::, :maps_size[k], :maps_size[k]].clone()
            feature_map = feature_map.view(4, -1)
            feature_map[:2] /= maps_size[k]
            for idx in range(num_anchors[k] - 1):
                feature_map[2] = scales_box[k] * ratios_sqrt[idx]
                feature_map[3] = scales_box[k] / ratios_sqrt[idx]
                box.append(feature_map)

            feature_map[-2:] = scales_box_hat[k]
            box.append(feature_map)
            xywh_boxes.append(torch.cat(box, dim=-1))

        xywh_boxes = torch.cat(xywh_boxes, dim=-1)
        ltrb_boxes = xywh2ltrb(xywh_boxes, dim=0)

        return {'xywh': xywh_boxes, 'ltrb': ltrb_boxes}

    @staticmethod
    def get_pos(default_box, y_true, threshold=0.5):
        """get_pos
            pos = (idx_default, idx_true, cls)
        """
        pos = list()
        ltrb_true = xywh2ltrb(y_true[2:], dim=0)
        for i, gt_box in enumerate(ltrb_true):
            pos_mask = bbox_iou(default_box['xywh'], gt_box) > threshold
            pos.append(pos_mask)
        return pos

    @staticmethod
    def get_hat(gt_box, db_box):
        """get_hat
            box (cx, cy, w, h)

        Args:
            gt_box: ground truth box
            db_box: default bounding box

        Returns:
            Tensor
        """
        g_hat_box = torch.zeros_like(gt_box)
        g_hat_box[:2] = (gt_box[:2] - db_box[:2]) / db_box[2:]
        g_hat_box[2:] = torch.log(gt_box[2:] / db_box[2:])
        return g_hat_box

    @staticmethod
    def get_scale(k, m=6, scale_min=0.2, scale_max=0.9):
        return scale_min + (scale_max - scale_min) * (k - 1) / (m - 1)

    @staticmethod
    def _crate_default_box(map_size):
        x = torch.Tensor(range(map_size))
        x1 = x.reshape([1, map_size]) + 0.5
        x2 = x.reshape([map_size, 1]) + 0.5
        y1 = x1.expand(1, map_size, map_size)
        y2 = x2.expand(1, map_size, map_size)
        y_ones = torch.ones(2, map_size, map_size)

        return torch.cat([y1, y2, y_ones], dim=0)
