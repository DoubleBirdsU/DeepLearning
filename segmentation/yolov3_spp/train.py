import argparse
import glob
import math
import os

import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from models import YOLOV3_SPP, YOLO_SPP, YoloLoss
from train_utils.train_eval_utils import Trainer
from utils.blocks import YOLOBlk
from utils.datasets import LoadImageAndLabels
from utils.parse_config import parse_data_cfg
from utils.utils import check_file


def train(hyper):
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    results_file = "results.txt"
    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    img_size_train = opt.img_size
    img_size_test = opt.img_size  # test image sizes
    multi_scale = opt.multi_scale

    # Image sizes
    # 图像要设置成32的倍数
    grid_size = 32  # (pixels) grid size
    assert math.fmod(img_size_test, grid_size) == 0, "--img-size %g must be a %g-multiple" % (img_size_test, grid_size)
    grid_min, grid_max = img_size_test // grid_size, img_size_test // grid_size
    if multi_scale:
        img_size_min = opt.img_size // 1.5
        img_size_max = opt.img_size // 0.667

        # 将给定的最大，最小输入尺寸向下调整到32的整数倍
        grid_min, grid_max = img_size_min // grid_size, img_size_max // grid_size
        img_size_min, img_size_max = int(grid_min * grid_size), int(grid_max * grid_size)
        img_size_train = img_size_max  # initialize with max size
        print("Using multi_scale training, image range[{}, {}]".format(img_size_min, img_size_max))

    # configure run
    # init_seeds()  # 初始化随机种子，保证结果可复现
    data_dict = parse_data_cfg(data)
    train_path = data_dict["train"]
    test_path = data_dict["valid"]
    num_cls = 1 if opt.single_cls else int(data_dict["classes"])  # number of classes
    hyper["cls"] *= num_cls / 80  # update coco-tuned hyp['cls'] to current dataset
    hyper["obj"] *= img_size_test / 320

    # Remove previous results
    for file in glob.glob(results_file):
        os.remove(file)

    # Initialize model
    # model = YOLOV3_SPP(cfg).to(device)
    model = YOLO_SPP(num_cls).to(device)

    # 是否冻结权重，只训练predictor的权重
    if isinstance(model, YOLOV3_SPP):
        weights = './weights/yolov3-spp-ultralytics-512.pt'
    else:
        weights = './weights/yolov3spp-0.pt'
    if isinstance(model, YOLOV3_SPP) and False:
        if opt.freeze_layers:
            # 索引减一对应的是predictor的索引，YOLOLayer并不是predictor
            output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if
                                    isinstance(module, YOLOBlk)]
            # 冻结除predictor和YOLOLayer外的所有层
            freeze_layer_indices = [x for x in range(len(model.module_list)) if
                                    (x not in output_layer_indices) and
                                    (x - 1 not in output_layer_indices)]
            # Freeze non-output layers
            # 总共训练3x2=6个parameters
            for idx in freeze_layer_indices:
                for parameter in model.module_list[idx].parameters():
                    parameter.requires_grad_(False)
        else:
            # 如果freeze_layer为False，默认仅训练除darknet53之后的部分
            # 若要训练全部权重，删除以下代码
            darknet_end_layer = 74  # only yolov3spp cfg
            # Freeze darknet53 layers
            # 总共训练21x3+3x2=69个parameters
            for idx in range(darknet_end_layer + 1):  # [0, 74]
                for parameter in model.module_list[idx].parameters():
                    parameter.requires_grad_(False)
    else:
        if opt.freeze_layers:
            model.freeze_layers(model.index_anchors)

    # optimizer
    params_grad = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params_grad, lr=hyper["lr0"], momentum=hyper["momentum"],
                          weight_decay=hyper["weight_decay"], nesterov=True)

    start_epoch = 0
    if weights.endswith(".pt") or weights.endswith(".pth"):
        epochs, start_epoch = loadCKPT(model, optimizer, epochs, weights, results_file, device, True)

    train_loader = None
    bool_trainer = False
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    # dataset
    if bool_trainer:
        # 训练集的图像尺寸指定为 multi_scale_range 中最大的尺寸
        train_loader = dataLoader(train_path, img_size_train, batch_size, True, hyper, opt.rect,
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls,
                                  num_workers=num_workers,
                                  pin_memory=True)

    # 验证集的图像尺寸指定为 img_size(512)
    test_loader = dataLoader(test_path, img_size_test, batch_size, True, hyper,
                             cache_images=opt.cache_images,
                             single_cls=opt.single_cls,
                             num_workers=num_workers,
                             pin_memory=True)

    # Model parameters
    loss_cfg = {
        'num_cls': num_cls,  # attach number of classes to model
        'hyp': hyper,  # attach hyper parameters to model
        'ratio': 1.0,  # giou loss ratio (obj_loss = 1.0 or giou)
        'anchors': model.anchor_vec,  # anchors
    }

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lr_lambda = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyper["lrf"]) + hyper["lrf"]  # cosine
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    trainer = Trainer(model, optimizer, loss=YoloLoss(multi_gpu=multi_gpu, cfg=loss_cfg),
                      lr_lambda=lr_lambda, last_epoch=start_epoch)
    if bool_trainer:
        print("starting training for %g epochs..." % epochs)
        print('Using %g data loader workers' % num_workers)
        trainer.fit_generate(train_loader,
                             epochs=epochs,
                             test_loader=test_loader,
                             print_freq=50,
                             save_best=True,
                             multi_scale=multi_scale,
                             img_size=img_size_train,
                             grid_min=grid_min,
                             grid_max=grid_max,
                             grid_size=grid_size,
                             device=device, warmup=True)
    else:
        trainer.evaluate(test_loader, device=device)
    pass


def dataLoader(data_path, img_size=416, batch_size=16, augment=True, hyper=None, rect=False,
               cache_images=False, single_cls=False, num_workers=16, pin_memory=True):
    r"""dataLoader

    Args:
        data_path:
        img_size: (=416)
        batch_size: (=16)
        augment: (=False)
        hyper: (=None)
        rect: (=False)
        cache_images: (=False)
        single_cls: (=False)
        num_workers: (=16)
        pin_memory: (=False)

    Returns:
        None
    """
    dataset = LoadImageAndLabels(
        data_path, img_size, batch_size, augment=augment,
        hyp=hyper, rect=rect, cache_images=cache_images, single_cls=single_cls)

    data_loader = data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=not rect, pin_memory=pin_memory, collate_fn=dataset.collate_fn)

    return data_loader


def loadCKPT(model, optimizer, epochs, weights_path, file_results, device, training=False):
    """loadCKPT

    Args:
        model:
        optimizer:
        epochs:
        weights_path:
        file_results:
        device:
        training:

    Returns:
        None
    """
    ckpt = torch.load(weights_path, map_location=device)
    start_epochs = 0

    # load model
    try:
        ckpt["model"] = {k: v for k, v in ckpt["model"].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt["model"], strict=False)
    except KeyError as e:
        msg = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
              "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
        raise KeyError(msg) from e

    if training:
        # load optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])

        # load results
        if ckpt.get("training_results") is not None:
            with open(file_results, "w") as file:
                file.write(ckpt["training_results"])  # write results.txt

        # epochs
        if ckpt['epoch']:
            start_epoch = ckpt["epoch"] + 1
            if epochs < start_epoch:
                print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                      (opt.weights, ckpt['epoch'], epochs))
                epochs += ckpt['epoch']  # finetune additional epochs

    # delete ckpt
    del ckpt

    return epochs, start_epochs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help="*.cfg path")
    parser.add_argument('--data', type=str, default='data/my_data.data', help='*.data path')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyper parameters path')
    parser.add_argument('--multi-scale', type=bool, default=True,
                        help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--savebest', type=bool, default=False, help='only save best checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov3spp.pt',
                        help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze-layers', type=bool, default=True, help='Freeze non-output layers')
    opt = parser.parse_args()

    # 检查文件是否存在
    opt.cfg = check_file(opt.cfg)
    opt.data = check_file(opt.data)
    opt.hyp = check_file(opt.hyp)

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(comment=opt.name)
    train(hyp)
