import torch
import torch.nn as nn
import numpy as np

from base.utils import xywh2ltrb, bbox_iou
from math import sqrt

from base.base_model import Module
from base.blocks import ResConvBlock, ResBlockB
from base.layers import FeatureExtractor
from base.model import ResNet, NNet

# ********************** SSD hyper-parameters begin ********************** #
g_img_size = (3, 300, 300)  # 输入图像尺寸
g_num_cls_anchors = (4, 6, 6, 6, 4, 4)  # anchor 的数量
g_maps_size = (38, 19, 10, 5, 3, 1)  # 特征图的尺寸
g_steps = (8, 16, 32, 64, 100, 300)  # 每层特征图中一个cell在原图中的跨度(尺寸)
g_ratios = (1, 2, 3)
g_scales = (21, 45, 99, 153, 207, 261, 315)  # 预测图像在原图中的基础跨度(尺寸或比例)
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
    """BackBone
        引入 RoIPooling
    """
    def __init__(self, num_cls=21, img_size=(3, 300, 300), num_anchor=4):
        super(BackBone, self).__init__()
        self.resnet = ResNet(num_cls, img_size, num_res_block=4, include_top=False)
        self.roi = nn.AdaptiveMaxPool2d((38, 38))
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
        x = self.roi(x)
        anchors = self.classifier(x)
        x = self.res_conv5(x)
        return x, anchors


# SSD
class SSD(NNet):
    def __init__(self, num_cls=21, img_size=(3, 300, 300)):
        super(SSD, self).__init__()
        self.ssd_blk_list = [
            BackBone(num_cls, img_size, num_anchor=g_num_cls_anchors[0]),
            SSDBlock(2048, 1024, 1024, 1, num_cls, num_anchor=g_num_cls_anchors[1], kernel_sizes=(3, 1)),
            SSDBlock(1024, 512, 256, 2, num_cls, num_anchor=g_num_cls_anchors[2]),
            SSDBlock(512, 256, 128, 2, num_cls, num_anchor=g_num_cls_anchors[3]),
            SSDBlock(256, 256, 128, 2, num_cls, num_anchor=g_num_cls_anchors[4]),
            SSDBlock(256, 256, 128, 2, num_cls, num_anchor=g_num_cls_anchors[5], padding='valid'),
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
        self.log_softmax = nn.LogSoftmax(dim=0)
        pass

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: {x, y, w, h, c0, c1, c2, ..., c20}, 0 <= ci <= 1
            y_true: {idx_img, c, x, y, w, h}

        Returns:
            None
        """
        loss_mean = 0.
        for k, pred in enumerate(y_pred):
            mask = y_true[:, 0] == k
            loss_mean += self.calc_one_img_loss(pred, y_true[mask])
            loss_mean = loss_mean * k / (k + 1)
        return loss_mean

    def calc_one_img_loss(self, y_pred, y_true):
        """calc_one_img_loss
            only calc a image loss
        """
        if len(y_true) == 0:
            return 0.0

        pred_box = y_pred[:4]
        pred_label = y_pred[4:]

        # Pos
        pos_box_mask = self.get_pos_mask(y_true)
        pos_mask = pos_box_mask[:, 0]
        true_label = pos_box_mask[:, 1]

        # loc loss
        hat_box = torch.Tensor(self.get_hat(pos_box_mask)).to(pred_box.device)
        pred_pos_box = pred_box[:, pos_mask]
        loc_loss = self.smooth_l1(pred_pos_box, hat_box).mean()

        # conf loss
        conf_pos = -self.log_softmax(pred_label[:, pos_mask])
        conf_pos = conf_pos[true_label, range(len(true_label))]

        neg_mask = np.array(np.ones(pred_box.shape[-1], dtype=np.bool))
        neg_mask[np.array(pos_mask, dtype=np.int32)] = False
        conf_neg = -self.log_softmax(pred_label[:, neg_mask])
        conf_neg, _ = torch.topk(conf_neg[0], len(pos_mask) * self.scale_neg)

        conf_loss = conf_pos.mean() + conf_neg.mean()
        return loc_loss + conf_loss

    def get_default_box(
            self, m=6,
            fig_size=300,
            ratios=None,
            num_anchors=None,
            maps_size=None,
            feat_scale=None,
            scale_xy=0.1,
            scale_wh=0.2,
            scale_min=0.2,
            scale_max=0.9,
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
            fig_size: the size of square input figure.
            scale_min: the min scale of the default boxes.
            scale_max: the max scale of the default boxes.
            ratios: 可以对特殊的数据集进行统计分析, 设置针对性的比例. 同一算法中, 针对不同的尺寸的图像设置不同的比例.
            num_anchors: the number of anchor layers.
            maps_size: the size of feature maps.
            feat_scale:
            scale_xy:
            scale_wh:

        Returns:
            Dict
        """
        if ratios is None:
            ratios = g_ratios
        if num_anchors is None:
            num_anchors = g_num_cls_anchors
        if maps_size is None:
            maps_size = g_maps_size
        if feat_scale is None:
            feat_scale = fig_size / np.array(g_steps)

        scales_box = np.array(g_scales) / fig_size
        scales_box_hat = [sqrt(scales_box[k] * scales_box[k + 1]) for k in range(m)]  # s'k
        scales_box = scales_box[:m]
        ratios_sqrt = [sqrt(ratio) for ratio in ratios]
        box_whs = list()
        for ratio in ratios_sqrt:
            if ratio == 1.:
                box_whs += [np.array([scales_box, scales_box]), np.array([scales_box_hat, scales_box_hat])]
            else:
                w = scales_box * ratio
                h = scales_box / ratio
                box_whs += [np.array([w, h]), np.array([h, w])]
        box_whs = np.array(box_whs).transpose([2, 1, 0]).clip(0, 1)

        max_map_size = max(maps_size)
        feature_map_org = self._crate_default_box(max_map_size)  # torch.Tensor

        xywh_boxes = list()
        for num_anchor, map_size, box_wh in zip(num_anchors, maps_size, box_whs):
            box = list()
            feature_map = feature_map_org[::, :map_size, :map_size].clone()  # torch.Tensor
            feature_map = feature_map.view(4, -1).numpy()
            feature_map[:2] /= map_size
            for idx in range(num_anchor):
                feature_map[-2:] = box_wh[:, idx:idx+1]
                box.append(feature_map.copy())

            xywh_boxes.append(np.concatenate(box, axis=-1))

        xywh_boxes = np.clip(np.concatenate(xywh_boxes, axis=-1), a_min=0, a_max=1)
        ltrb_boxes = xywh2ltrb(xywh_boxes, dim=0)

        return {'xywh': xywh_boxes, 'ltrb': ltrb_boxes}

    def get_pos_mask(self, y_true, threshold=0.5):
        """get_pos
            pos = (idx_true, idx_default)
        """
        ltrb_boxes = self.default_box['ltrb']
        y_true = np.array(y_true.to(torch.device('cpu')))
        gt_box = xywh2ltrb(y_true[:, 2:], dim=-1)

        # IoU: jaccard overlap
        iou = self.calc_iou(ltrb_boxes, gt_box)

        #  = 2.0
        dbox_mask = np.argmax(iou, axis=1)
        iou[range(len(dbox_mask)), dbox_mask] = 2.0

        bbox_mask = np.argmax(iou, axis=0)
        max_iou_bbox = iou[bbox_mask, range(len(bbox_mask))]

        # iou > threshold
        pos_thr_mask = np.argwhere(max_iou_bbox > threshold).reshape(-1)
        thr_bbox_mask = bbox_mask[pos_thr_mask]

        pos_true = y_true[thr_bbox_mask][:, 1:]
        pos_mask = np.concatenate([pos_thr_mask[:, np.newaxis], pos_true], axis=-1)
        return pos_mask

    @staticmethod
    def calc_iou(ltrb_boxes, list_boxes):
        iou_list = [bbox_iou(ltrb_boxes, gt_box, iou_type='iou').numpy()[np.newaxis, :] for gt_box in list_boxes]
        return np.concatenate(iou_list, axis=0)

    def get_hat(self, gt_box):
        """get_hat
            box (cx, cy, w, h)

        Args:
            gt_box: ground truth box

        Returns:
            Tensor
        """
        db_box = self.default_box['xywh'][:, np.array(gt_box[:, 0], dtype=np.int32)]
        gt_xywh_box = gt_box[:, -4:].T
        hat_box = np.zeros_like(gt_xywh_box)
        hat_box[:2] = (gt_xywh_box[:2] - db_box[:2]) / db_box[2:]
        hat_box[2:] = np.log(gt_xywh_box[2:] / db_box[2:])
        return hat_box

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
