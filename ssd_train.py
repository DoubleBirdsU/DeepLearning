import argparse
import dill

import yaml
import torch

from base.model import NNet
from base.parse_config import parse_data_cfg
from base.torch_utils import load_breakpoint
from segmentation.ssd.ssd_models import SSD, SSDLoss
from segmentation.ssd.ssd_utils import dataLoader


def train(arg_params, hyper_params, net_cls):
    """train

    Args:
        arg_params:
        hyper_params: hyper-parameters
        net_cls: NNet
    """
    use_cuda = not arg_params.no_cuda and torch.cuda.is_available()
    torch.manual_seed(arg_params.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data Loader
    data_dict = parse_data_cfg(arg_params.data)
    train_path = data_dict["train"]
    test_path = data_dict["valid"]
    batch_size = 4
    kwargs = {
        'cache_images': False,
        'single_cls': False,
        'num_workers': 2,
        'pin_memory': True
    } if use_cuda else {}
    # 训练集的图像尺寸指定为 multi_scale_range 中最大的尺寸
    train_loader = dataLoader(train_path, 300, batch_size, True, hyper_params, False, **kwargs)

    # 验证集的图像尺寸指定为 img_size(512)
    test_loader = dataLoader(test_path, 512, 8, True, hyper_params, **kwargs)

    # Model
    img_size = (3, 300, 300)
    net = net_cls(num_cls=21, img_size=img_size).to(device)
    net.compile(optimizer='adam', loss=SSDLoss(), device=device)

    # fit 断点续训
    cp_callback = load_breakpoint(net, 'VOC2012', save_weights_only=True, save_best_only=True, pickle_module=dill)
    net.fit_generator(train_data=train_loader,
                      batch_size=batch_size,
                      epochs=4,
                      validation_data=test_loader,
                      callbacks=[cp_callback],
                      validation_batch_size=2)


def evaluate(arg_params, hyper_params, net_cls):
    use_cuda = not arg_params.no_cuda and torch.cuda.is_available()
    torch.manual_seed(arg_params.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data Loader
    data_dict = parse_data_cfg(arg_params.data)
    test_path = data_dict["valid"]
    batch_size = 2
    kwargs = {
        'cache_images': False,
        'single_cls': False,
        'num_workers': 2,
        'pin_memory': True
    } if use_cuda else {}
    # 验证集的图像尺寸指定为 img_size(512)
    test_loader = dataLoader(test_path,
                             300,
                             batch_size,
                             True,
                             hyper_params,
                             **kwargs)

    # Model
    img_size = (3, 300, 300)
    net = net_cls(num_cls=21, img_size=img_size).to(device)
    net.compile(optimizer='adam', loss=SSDLoss(), device=device)

    # fit 断点续训
    net.load_weights('weights/VOC2012/VOC2012_SSD_weight.pt.best')
    net.evaluate_generator(test_loader, batch_size)


def test():
    y_true = torch.Tensor([[0.00000, 5.00000, 0.48467, 0.20905, 0.64400, 0.41810],
                           [0.00000, 6.00000, 0.09633, 0.31521, 0.19267, 0.15385],
                           [0.00000, 14.00000, 0.85567, 0.33619, 0.09800, 0.17982],
                           [0.00000, 14.00000, 0.73867, 0.23529, 0.04400, 0.08192],
                           [0.00000, 14.00000, 0.60867, 0.22530, 0.04000, 0.05794],
                           [0.00000, 14.00000, 0.64367, 0.22630, 0.03000, 0.06793],
                           [0.00000, 14.00000, 0.55667, 0.23229, 0.06000, 0.08392],
                           [0.00000, 17.00000, 0.37467, 0.79000, 0.46400, 0.24800]])

    ssd_loss = SSDLoss()
    # ssd_loss.get_pos_mask(y_true)
    ssd_loss.calc_one_img_loss(torch.Tensor(ssd_loss.default_box['xywh']), y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data-root', type=str, default='/home/wnyl/.dataset',
                        help='For Saved dataset.')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help="*.cfg path")
    parser.add_argument('--data', type=str, default='data/subdata/my_data.data', help='*.data path')
    parser.add_argument('--hyp', type=str, default='cfg/ssd_hyp.yaml', help='hyper parameters path')
    parser.add_argument('--multi-scale', type=bool, default=True,
                        help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--save-best', type=bool, default=False, help='only save best checkpoint')
    parser.add_argument('--no-test', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov3spp.pt',
                        help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze-layers', type=bool, default=True, help='Freeze non-output layers')

    args = parser.parse_args()
    # 检查文件是否存在
    # args.data = 'data/test_01/my_data.data'
    # train(args, hyp, SSD)
    net_filename = 'cfg/lenet.yaml'
    with open(net_filename) as f:
        net_dict = yaml.load(f, Loader=yaml.SafeLoader)
        net = NNet(net_dict)
    print(net_dict)
