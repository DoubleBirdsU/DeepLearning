import torch
import numpy as np

from utils import torch_utils, blocks
from utils.base_model import Module
from utils.blocks import FeatureConvBlk
from utils.model import Model
from utils.parse_config import parse_model_cfg

from typing import List, Tuple

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
        self.output_size = self._make_down_sample(32, 64, index_out_block)

        self.collect_layers(self.blocks_list)

    def __call__(self, x):
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
        return out_ch_base << (len(blocks_num) - 1)

    def _make_blocks(self, in_ch, out_ch, num_block):
        self.blocks_list.append(
            blocks.FeatureExtractorBlk(in_ch, out_ch, 3, 2, padding='tiny-same', bn=True, activation='leaky'))

        in_ch, out_ch = self.swap(in_ch, out_ch)
        for i in range(num_block):
            self.blocks_list.append(blocks.DarknetBlk(in_ch, out_ch))


class YOLO_SPP(Model):
    def __init__(self, img_size=(416, 416)):
        super(YOLO_SPP, self).__init__()
        # backbone
        self.backbone = Darknet53(img_size=img_size, index_out_block=(-3, -2, -1))

        # FeatureConvBlk
        self.fcb = FeatureConvBlk(in_ch=1024, out_ch=512, kernels_size=(1, 3, 1),
                                  in_ch_first=self.backbone.output_size)

        self.spp = blocks.SPPBlk()  # SPP
        self.yolo = blocks.YOLOBlk(1024, 512)  # YOLOBlk

        self.collect_layers([self.backbone, self.fcb, self.spp, self.yolo])

    def __call__(self, x):
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

    def __call__(self, x, augment=False, verbose=False):
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
