import torch
import numpy as np
import torch.nn as nn

from utils import torch_utils, blocks, utils
from utils.base_model import Module
from utils.blocks import FeatureConvBlk
from utils.model import Model
from utils.parse_config import parse_model_cfg

from typing import List, Tuple

from utils.utils import wh_iou, smooth_BCE, FocalLoss, bbox_iou

ONNX_EXPORT = False


class Darknet53(Model):
    def __init__(self, img_size=(416, 416), index_out_block=tuple([-1])):
        """Darknet53

        :type index_out_block: Tuple[int]

        Param:
            img_size:
            index_out_block: (-3, -2, -1)

        Returns:
            feature_maps: out,
        """
        super(Darknet53, self).__init__()
        in_ch = 3
        img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        self.feature_map_index_list = list()
        self.blocks_list: List[Module] = list()
        self.blocks_list.append(
            blocks.FeatureExtractorBlk(in_ch, 32, 3, 1, padding='tiny-same', bn=True, activation='leaky'))

        # DownSample
        self._make_down_sample(32, 64, index_out_block)

        self.collect_layers(self.blocks_list)

    def forward(self, x):
        feature_maps = list()
        for i, block in enumerate(self.blocks_list):
            x = block(x)
            if i in self.feature_map_index_list:
                feature_maps.append(x.clone())
        return feature_maps

    def _make_down_sample(self, in_ch_base, out_ch_base, index_out_block, blocks_num=(1, 2, 8, 8, 4)):
        index_out = [len(blocks_num) + i for i in index_out_block]
        for i, num_block in enumerate(blocks_num):
            self._make_blocks(in_ch_base << i, out_ch_base << i, num_block)
            if i in index_out:
                self.feature_map_index_list.append(len(self.blocks_list) - 1)

    def _make_blocks(self, in_ch, out_ch, num_block):
        self.blocks_list.append(
            blocks.FeatureExtractorBlk(in_ch, out_ch, 3, 2, padding='tiny-same', bn=True, activation='leaky'))

        in_ch, out_ch = self.swap(in_ch, out_ch)
        for i in range(num_block):
            self.blocks_list.append(blocks.DarknetBlk(in_ch, out_ch))


class YOLO_SPP(Model):
    def __init__(self, num_cls, img_size=(416, 416), anchors=None, anchor_strides=None):
        super(YOLO_SPP, self).__init__()
        self.num_cls = num_cls
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        if not anchors:
            anchors = torch.Tensor(
                [[[10, 13], [16, 30], [33, 23]],
                 [[30, 61], [62, 45], [59, 119]],
                 [[116, 90], [156, 198], [373, 326]]]).flip(0)
        if not anchor_strides:
            anchor_strides = [32, 16, 8]
        anchors = anchors if isinstance(anchors, torch.Tensor) else torch.Tensor(anchors)
        self.anchor_vec = [anchors[i] / anchor_strides[i] for i in range(len(anchors))]
        self.out_ch_last = (num_cls + 5) * len(anchors[0])

        # Model
        # backbone
        self.backbone = Darknet53(img_size=img_size, index_out_block=(-3, -2, -1))

        # FeatureConvBlk
        self.fcb = FeatureConvBlk(in_ch=1024, out_ch=512, kernels_size=(1, 3, 1),
                                  in_ch_first=self.backbone.out_ch_last)

        self.spp = blocks.SPPBlk(in_ch_first=self.fcb.out_ch_last)  # SPP
        self.yolo = blocks.YOLOBlk(1024, 512, self.anchor_vec, out_ch_last=self.out_ch_last)  # YOLOBlk
        num_layers = self.spp.count_modules() + self.fcb.count_modules() + self.backbone.count_modules()
        self.index_anchors = [num_layers + i for i in self.yolo.index_anchors]

        self.collect_layers([self.backbone, self.fcb, self.spp, self.yolo], bool_out=True)

    def forward(self, x):
        feature_maps = self.backbone(x)
        x = self.spp(self.fcb(feature_maps[-1]))
        anchor1, anchor2, anchor3 = self.yolo([x, feature_maps[1], feature_maps[0]])
        return [anchor1, anchor2, anchor3]


class YOLOV3_SPP(Model):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(YOLOV3_SPP, self).__init__()
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = self.create_modules(self.module_defs, img_size, cfg)
        self.yolo_layers = self.get_index('YOLOLayer')

        # YOLOV3_SPP Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, augment=False, verbose=False):
        if not augment:
            return self.forward_once(x)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                y.append(self.forward_once(xi)[0])
            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            ver_str = ''

        # Augment images (inference and test only)
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(x, s[1]),  # scale
                           ), 0)

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    ver_str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == "YOLOLayer":
                yolo_out.append(module(x, out))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), ver_str)
                ver_str = ''

        if self.training:  # train
            return yolo_out
        elif ONNX_EXPORT:  # export
            # x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            # return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
            p = torch.cat(yolo_out, dim=0)
            return p
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            return x, p

    def info(self, verbose=False):
        """打印模型的信息
            :param verbose:
            :return:
        """
        torch_utils.model_info(self, verbose)


class YoloLoss(object):
    def __init__(self, multi_gpu, cfg):
        """YoloLoss

        Args:
            multi_gpu: bool, is the model running multi gpu? True or False
            cfg: keys = ('num_cls', 'hyp', 'ratio', 'anchors'),
                   num_cls: number of classes; hyp: hyper parameters; ratio: giou ratio; anchor: anchors / strides

        Returns:
            None
        """
        self.multi_gpu = multi_gpu  # is the model running multi gpu.
        self.num_cls = cfg['num_cls']  # number of classes
        self.hyp = cfg['hyp']  # hyper parameters
        self.giou_ratio = cfg['ratio']  # GIou ratio
        self.anchors_vec = cfg['anchors']  # anchors / strides

    def __call__(self, y_pred, y_true):
        device = y_pred[0].device
        loss_cls = torch.zeros(1, device=device)  # Tensor(0)
        loss_box = torch.zeros(1, device=device)  # Tensor(0)
        loss_obj = torch.zeros(1, device=device)  # Tensor(0)

        target_cls, target_box, indices, anchors = self.build_targets(y_pred, y_true)  # targets

        # Define criteria
        reduction = 'mean'  # Loss reduction (sum or mean)
        BCE_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['cls_pw']], device=device),
                                       reduction=reduction)
        BCE_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['obj_pw']], device=device),
                                       reduction=reduction)

        # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        label_pos, label_neg = smooth_BCE(eps=0.0)

        # focal loss
        fl_gamma = self.hyp['fl_gamma']  # focal loss gamma
        if fl_gamma > 0:
            BCE_cls, BCE_obj = FocalLoss(BCE_cls, fl_gamma), FocalLoss(BCE_obj, fl_gamma)

        # per output
        count_targets = 0  # targets
        for jdx, pred in enumerate(y_pred):  # layer index, layer predictions
            idx_img, idx_anchor, grid_y, grid_x = indices[jdx]  # image, anchor, grid_y, grid_x
            target_obj = torch.zeros_like(pred[..., 0], device=device)  # target obj

            num_target = idx_img.shape[0]  # number of targets
            if num_target:
                count_targets += num_target  # cumulative targets
                # 对应匹配到正样本的预测信息
                # prediction subset corresponding to targets
                pred_sub = pred[idx_img, idx_anchor, grid_y, grid_x]

                # GIoU
                pred_xy = pred_sub[..., :2].sigmoid()
                pred_wh = pred_sub[..., 2:4].exp().clamp(max=1E3) * anchors[jdx]
                pred_box = torch.cat((pred_xy, pred_wh), 1)  # predicted box
                giou = bbox_iou(pred_box.t(), target_box[jdx].t(), ltrb=False,
                                iou_type='GIoU')  # giou(prediction, target)
                loss_box += (1.0 - giou).mean()  # giou loss

                # Obj giou ratio
                target_obj[idx_img, idx_anchor, grid_y, grid_x] = \
                    (1.0 - self.giou_ratio) + self.giou_ratio * giou.detach().clamp(0).type(target_obj.dtype)

                # Class
                if self.num_cls > 1:  # cls loss (only if multiple classes)
                    pred_tar = torch.full_like(pred_sub[:, 5:], label_neg, device=device)  # targets
                    pred_tar[range(num_target), target_cls[jdx]] = label_pos
                    loss_cls += BCE_cls(pred_sub[:, 5:], pred_tar)  # BCE

            loss_obj += BCE_obj(pred[..., 4], target_obj)  # obj loss

        # 乘上每种损失的对应权重
        loss_box *= self.hyp['giou']
        loss_obj *= self.hyp['obj']
        loss_cls *= self.hyp['cls']

        # loss = loss_box + loss_obj + loss_cls
        return {"box_loss": loss_box, "obj_loss": loss_obj, "class_loss": loss_cls}

    def build_targets(self, y_pred, y_true):
        device = y_true.device

        # Build targets for compute_loss(), input y_true(idx_img, class, x, y, w, h)
        num_targets = y_true.shape[0]
        gain = torch.ones(6, device=device)
        overlap_offsets = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=y_true.device).float()

        # target_cls, target_box, indices, anchors
        target_cls, target_box, indices, anchors = [], [], [], []
        for i, anchor_vec in enumerate(self.anchors_vec):
            anchor_vec = anchor_vec.to(device)
            gain[2:] = torch.tensor(y_pred[i].shape)[[3, 2, 3, 2]]  # xywh gain

            # Match targets to anchors
            match_anchors, y_true_gain, offsets = [], y_true * gain, 0
            if num_targets:
                match_anchors, y_true_gain, offsets = self.match2anchors(anchor_vec, y_true_gain, overlap_offsets, gain,
                                                                         self.hyp)

            # Define
            indic, tar_box, anchor, cls = self.deal_targets_gain(match_anchors, y_true_gain, offsets, anchor_vec)

            # Append
            indices.append(indic)  # image, anchor, grid indices(x, y)
            target_box.append(tar_box)  # gt box相对anchor的x,y偏移量以及w,h
            anchors.append(anchor)  # anchors
            target_cls.append(cls)  # class
            if cls.shape[0]:  # if any targets
                # 目标的标签数值不能大于给定的目标类别数
                assert cls.max() < self.num_cls, 'Model accepts %g classes labeled from 0-%g, ' \
                                                 'however you labelled match_anchors class %g. ' \
                                                 'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                                     self.num_cls, self.num_cls - 1, cls.max())

        return target_cls, target_box, indices, anchors

    def build_targets_v2(self, y_pred, y_true):
        rect_style = 'valid'
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        num_targets = y_true.shape[0]
        gain = torch.ones(6, device=y_true.device)  # normalized to grid space gain
        lap_off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=y_true.device).float()  # overlap offsets

        # target_cls, target_box, indices, anchors
        target_cls, target_box, indices, anchors = [], [], [], []
        for i, anchor_vec in enumerate(self.anchors_vec):  # [89, 101, 113]
            # 获取该yolo predictor对应的anchors
            anchor = anchor_vec.to(y_true.device)
            gain[2:] = torch.tensor(y_pred[i].shape)[[3, 2, 3, 2]]  # ltrb gain

            # Match targets to anchors
            match_anchors, targets_gain, offsets = [], y_true * gain, 0
            if num_targets:  # 如果存在target的话
                match_anchors, targets_gain, offsets = self.match2anchors_v2(anchor, targets_gain, num_targets,
                                                                             rect_style, lap_off, gain, self.hyp)

            # Define
            indic, tar_box, anchor, cls = self.deal_targets_gain(targets_gain, offsets, anchor, match_anchors)

            # Append
            indices.append(indic)  # image, anchor, grid indices(x, y)
            target_box.append(tar_box)  # gt box相对anchor的x,y偏移量以及w,h
            anchors.append(anchor)  # anchors
            target_cls.append(cls)  # class
            if cls.shape[0]:  # if any targets
                # 目标的标签数值不能大于给定的目标类别数
                assert cls.max() < self.num_cls, 'Model accepts %g classes labeled from 0-%g, ' \
                                                 'however you labelled match_anchors class %g. ' \
                                                 'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                                     self.num_cls, self.num_cls - 1, cls.max())

        return target_cls, target_box, indices, anchors

    @staticmethod
    def match2anchors(anchor_vec, y_true_gain, overlap_offsets, gain, hyp, rect_type='rect2'):
        num_anchor = anchor_vec.shape[0]
        num_targets = y_true_gain.shape[0]
        # anchor tensor, same as .repeat_interleave(num_targets)
        # [0, ..., num_anchor - 1] -> shape = [num_anchor, 1] -> shape = [num_anchor, num_targets]
        anchor_mask = torch.arange(num_anchor).view(num_anchor, 1).repeat(1, num_targets)
        # iou(3,n) = wh_iou(anchors(3,2), grid_wh(n,2)), a ground truth box may be in two anchors.
        mask = utils.wh_iou(anchor_vec, y_true_gain[:, 4:6]) > hyp['iou_t']

        # 获取iou大于阈值的 anchor 与 target 对应信息
        match_anchors = anchor_mask[mask]  # filter
        y_true_gain = y_true_gain.repeat(num_anchor, 1, 1)[mask]
        offsets = 0

        # overlaps
        grid_xy = y_true_gain[:, 2:4]  # grid xy
        zeros_xy = torch.zeros_like(grid_xy)
        if rect_type == 'rect2':
            offset_gain = 0.2  # offset
            j, k = ((grid_xy % 1. < offset_gain) & (grid_xy > 1.)).T
            match_anchors = torch.cat((match_anchors, match_anchors[j], match_anchors[k]), 0)
            y_true_gain = torch.cat((y_true_gain, y_true_gain[j], y_true_gain[k]), 0)
            offsets = torch.cat((zeros_xy, zeros_xy[j] + overlap_offsets[0],
                                 zeros_xy[k] + overlap_offsets[1]), 0) * offset_gain
        elif rect_type == 'rect4':
            offset_gain = 0.5  # offset
            j, k = ((grid_xy % 1. < offset_gain) & (grid_xy > 1.)).T
            l, m = ((grid_xy % 1. > (1 - offset_gain)) & (grid_xy < (gain[[2, 3]] - 1.))).T
            match_anchors = torch.cat(
                (match_anchors, match_anchors[j], match_anchors[k], match_anchors[l], match_anchors[m]), 0)
            y_true_gain = torch.cat(
                (y_true_gain, y_true_gain[j], y_true_gain[k], y_true_gain[l], y_true_gain[m]), 0)
            offsets = torch.cat((zeros_xy, zeros_xy[j] + overlap_offsets[0], zeros_xy[k] + overlap_offsets[1],
                                 zeros_xy[l] + overlap_offsets[2], zeros_xy[m] + overlap_offsets[3]), 0) * offset_gain
        return match_anchors, y_true_gain, offsets

    @staticmethod
    def match2anchors_v2(anchors, y_true_gain, num_targets, rect_type, lap_off, gain, hyp):
        num_anchors = anchors.shape[0]  # number of anchors
        # anchor tensor, same as .repeat_interleave(num_targets)
        # [0, ..., num_anchor - 1] -> shape = [num_anchor, 1] -> shape = [num_anchor, num_targets]
        anchor_mask = torch.arange(num_anchors).view(num_anchors, 1).repeat(1, num_targets)
        # iou(3,n) = wh_iou(anchors(3,2), grid_wh(n,2))
        mask = wh_iou(anchors, y_true_gain[:, 4:6]) > hyp['iou_t']
        # targets_gain.repeat(num_anchors, 1, 1): [num_targets, 6] -> [3, num_targets, 6]
        # 获取iou大于阈值的anchor与target对应信息
        match_anchors = anchor_mask[mask]  # filter
        y_true_gain = y_true_gain.repeat(num_anchors, 1, 1)[mask]
        offsets = 0

        # overlaps
        grid_xy = y_true_gain[:, 2:4]  # grid xy
        zeros_xy = torch.zeros_like(grid_xy)
        if rect_type == 'rect2':
            offset_gain = 0.2  # offset
            j, k = ((grid_xy % 1. < offset_gain) & (grid_xy > 1.)).T
            match_anchors = torch.cat((match_anchors, match_anchors[j], match_anchors[k]), 0)
            y_true_gain = torch.cat((y_true_gain, y_true_gain[j], y_true_gain[k]), 0)
            offsets = torch.cat((zeros_xy, zeros_xy[j] + lap_off[0], zeros_xy[k] + lap_off[1]), 0) * offset_gain
        elif rect_type == 'rect4':
            offset_gain = 0.5  # offset
            j, k = ((grid_xy % 1. < offset_gain) & (grid_xy > 1.)).T
            l, m = ((grid_xy % 1. > (1 - offset_gain)) & (grid_xy < (gain[[2, 3]] - 1.))).T
            match_anchors = torch.cat(
                (match_anchors, match_anchors[j], match_anchors[k], match_anchors[l], match_anchors[m]), 0)
            y_true_gain = torch.cat(
                (y_true_gain, y_true_gain[j], y_true_gain[k], y_true_gain[l], y_true_gain[m]), 0)
            offsets = torch.cat((zeros_xy, zeros_xy[j] + lap_off[0], zeros_xy[k] + lap_off[1],
                                 zeros_xy[l] + lap_off[2], zeros_xy[m] + lap_off[3]), 0) * offset_gain
        return match_anchors, y_true_gain, offsets

    @staticmethod
    def deal_targets_gain(match_anchors, y_true_gain, offsets, anchor_vec):
        # match_anchors, y_true_gain, offsets, anchor_vec
        # Define
        # long等于to(torch.int64), 数值向下取整
        idx_img, cls = y_true_gain[:, :2].long().T  # image index, class
        grid_xy = y_true_gain[:, 2:4]  # grid xy
        grid_wh = y_true_gain[:, 4:6]  # grid wh
        grid_ij = (grid_xy - offsets).long()  # 匹配 targets 所在的 grid cell 左上角坐标
        grid_i, grid_j = grid_ij.T  # grid xy indices

        # Append
        indic = (idx_img, match_anchors, grid_j, grid_i)  # image index, anchor, grid indic(x, y)
        target_box = torch.cat((grid_xy - grid_ij, grid_wh), 1)  # ground truth box 相对 anchor 的 x, y 偏移量以及 w, h
        anchor = anchor_vec[match_anchors]  # anchor

        return indic, target_box, anchor, cls
