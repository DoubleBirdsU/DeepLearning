import glob
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from base.utils import xywh2ltrb, ltrb2xywh

help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']

# get orientation in exif tag
# 找到图像exif信息中对应旋转信息的key值
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def exif_size(img):
    """
    获取图像的原始img size
    通过exif的orientation信息判断图像是否有旋转，如果有旋转则返回旋转前的size
    :param img: PIL图片
    :return: 原始图像的size
    """
    # Returns exif-corrected PIL size
    size = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270  顺时针翻转90度
            size = (size[1], size[0])
        elif rotation == 8:  # rotation 90  逆时针翻转90度
            size = (size[1], size[0])
    except:
        # 如果图像的exif信息中没有旋转信息，则跳过
        pass

    return size


class LoadImageAndLabels(Dataset):  # for training/testing
    def __init__(
            self, path,
            img_size=416,
            batch_size=16,
            augment=False,
            hyp=None,
            rect=False,
            image_weights=False,
            cache_images=False,
            single_cls=False,
            pad=0.0,
            rank=-1
    ):
        try:
            path = str(Path(path))
            parent = str(Path(path).parent) + os.sep
            if os.path.isfile(path):  # file
                with open(path, "r") as f:
                    f = f.read().splitlines()
                    # local to global path
                    f = [x.replace("./", parent) if x.startswith("./") else x for x in f]
            elif os.path.isdir(path):  # folder
                f = glob.iglob(path + os.sep + "*.*")
            else:
                raise Exception("%s does not exist" % path)

            # local to global path
            self.img_files = [x.replace("./", parent) for x in f if os.path.splitext(x)[-1].lower() in img_formats]
        except:
            raise Exception("Error loading data from %s. See %s" % (path, help_url))

        num_files = len(self.img_files)
        assert num_files > 0, "No images found in %s. See %s" % (path, help_url)

        # batch index
        idx_batch = np.floor(np.arange(num_files) / batch_size).astype(np.int)
        num_batch = idx_batch[-1] + 1  # number of batches

        self.num_imgs = num_files  # number of images
        self.batch = idx_batch  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)

        # Define labels
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt")
                            for x in self.img_files]

        # Read image shapes (wh)
        shape_path = path.replace(".txt", "") + ".shapes"  # shapefile path
        try:
            with open(shape_path, "r") as f:  # read existing shapefile
                shapes = [x.split() for x in f.read().splitlines()]
                # 判断现有的shape文件中的行数(图像个数)是否与当前数据集中图像个数相等
                # 如果不相等则认为是不同的数据集，故重新生成shape文件
                assert len(shapes) == num_files, "shapefile out of async"
        except:
            # tqdm 库会显示处理的进度
            shapes = [exif_size(Image.open(f)) for f in tqdm(self.img_files, desc="Reading image shapes")]
            # 将所有图片的shape信息保存在.shape文件中
            np.savetxt(shape_path, shapes, fmt="%g")  # overwrite existing (if any)

        self.shapes = np.array(shapes, dtype=np.float64)

        # Rectangular Training https://github.com/ultralytics/yolov3/issues/232
        # 如果为ture，训练网络时，会使用类似原图像比例的矩形(让最长边为img_size)，而不是img_size x img_size
        if self.rect:
            # Sort by aspect ratio
            shapes = self.shapes  # wh
            aspect_ratio = shapes[:, 1] / shapes[:, 0]  # aspect ratio
            # argsort 函数返回的是数组值从小到大的索引值
            # 按照长宽比例进行排序
            idx_rect = aspect_ratio.argsort()
            self.img_files = [self.img_files[i] for i in idx_rect]
            self.label_files = [self.label_files[i] for i in idx_rect]
            self.shapes = shapes[idx_rect]  # wh
            aspect_ratio = aspect_ratio[idx_rect]

            # set training image shapes
            shapes = [[1, 1]] * num_batch  # nb: number of batches
            for i in range(num_batch):
                ari = aspect_ratio[idx_batch == i]  # bi: batch index
                mini, maxi = ari.min(), ari.max()

                # 如果高/宽小于1(w > h)，将w设为img_size
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                # 如果高/宽大于1(w < h)，将h设置为img_size
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            # 计算每个batch输入网络的shape值
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32. + pad).astype(np.int) * 32

        # cache labels
        self.imgs = [None] * num_files
        # label: [class, x, y, w, h]
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * num_files
        create_data_subset, extract_bounding_boxes, labels_loaded = False, False, False
        # number mission, found, empty, data sunset, duplicate
        num_miss, num_found, num_empty, num_subset, num_duplicate = 0, 0, 0, 0, 0
        # 这里分别命名是为了防止出现rect为False/True时混用导致计算的mAP错误
        # 当rect为True时会对self.images和self.labels进行从新排序
        if rect is True:
            np_labels_path = str(Path(self.label_files[0]).parent) + ".rect.npy"  # saved labels in *.npy file
        else:
            np_labels_path = str(Path(self.label_files[0]).parent) + ".norect.npy"
        if os.path.isfile(np_labels_path):
            shapes = np_labels_path  # print string
            x = np.load(np_labels_path, allow_pickle=True)
            if len(x) == num_files:
                self.labels = x
                labels_loaded = True
        else:
            shapes = path.replace("images", "labels")

        # 处理进度条只在第一个进程中显示
        if rank in [-1, 0]:
            pbar = tqdm(self.label_files)
        else:
            pbar = self.label_files

        for i, file in enumerate(pbar):
            if labels_loaded is True:
                label = self.labels[i]
            else:
                try:
                    with open(file, "r") as f:
                        label = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except Exception as e:
                    num_miss += 1  # file missing
                    continue

            # 如果标注信息不为空的话
            if label.shape[0]:
                assert label.shape[1] == 5, "> 5 label columns: %s" % file
                assert (label >= 0).all(), "negative labels: %s" % file
                assert (label[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s" % file

                # 检查每一行，看是否有重复信息
                if np.unique(label, axis=0).shape[0] < label.shape[0]:  # duplicate rows
                    num_duplicate += 1
                if single_cls:
                    label[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = label
                num_found += 1  # file found

                # create sub-dataset (a smaller dataset)
                if create_data_subset and num_subset < 1E4:
                    if num_subset == 0:
                        create_folder(path="./datasubset")
                        os.makedirs("./datasubset/images")
                    exclude_classes = 43
                    if exclude_classes not in label[:, 0]:
                        num_subset += 1
                        with open("./datasubset/images.txt", "a") as f:
                            f.write(self.img_files[i] + "\n")

                # Extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(label):
                        f = "%s%sclassifier%s%g_%g_%s" % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)  # make new output folder

                        # 将相对坐标转为绝对坐标
                        # b: x, y, w, h
                        b = x[1:] * [w, h, w, h]  # box
                        # 将宽和高设置为宽和高中的最大值
                        b[2:] = b[2:].max()  # rectangle to square
                        # 在宽和高方向添加padding
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        # 将坐标格式从 x,y,w,h -> xmin,ymin,xmax,ymax
                        b = xywh2ltrb(b.reshape(-1, 4)).revel().astype(np.int)

                        # 裁剪bbox坐标到图片内
                        b[[0, 2]] = np.clip[b[[0, 2]], 0, w]
                        b[[1, 3]] = np.clip[b[[1, 3]], 0, h]
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), "Failure extracting classifier boxes"
            else:
                num_empty += 1  # file empty

            # 处理进度条只在第一个进程中显示
            if rank in [-1, 0]:
                pbar.desc = "Caching labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)" % (
                    shapes, num_found, num_miss, num_empty, num_duplicate, num_files)
        assert num_found > 0 or num_files == 20288, "No labels found in %s. See %s" % (
            os.path.dirname(file) + os.sep, help_url)

        # 如果标签信息没有被保存成numpy的格式，且训练样本数大于1000则将标签信息保存成numpy的格式
        if not labels_loaded and num_files > 1000:
            print("Saving labels to %s for faster future loading" % np_labels_path)
            np.save(np_labels_path, self.labels)  # save for next time

        # Cache images into memory for faster training (Warning: large datasets may exceed system RAM)
        if cache_images:  # if training
            gigabyte = 0  # Gigabytes of cached images 用于记录缓存图像占用RAM大小
            pbar = tqdm(range(len(self.img_files)), desc="Caching images")
            self.img_hw0, self.img_hw = [None] * num_files, [None] * num_files
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gigabyte += self.imgs[i].nbytes  # 用于记录缓存图像占用RAM大小
                pbar.desc = "Caching images (%.1fGB)" % (gigabyte / 1E9)

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for file in tqdm(self.img_files, desc="Detecting corrupted images"):
                try:
                    _ = io.imread(file)
                except:
                    print("Corrupted image detected: %s" % file)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        if self.mosaic:
            # load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None
        else:
            # load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scale_up=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()  # label: class, x, y, w, h
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment image space
            if not self.mosaic:
                img, labels = random_affine(img, labels,
                                            degrees=hyp["degrees"],
                                            translate=hyp["translate"],
                                            scale=hyp["scale"],
                                            shear=hyp["shear"])

            # Augment color space
            augment_hsv(img, h_gain=hyp["hsv_h"], s_gain=hyp["hsv_s"], v_gain=hyp["hsv_v"])

        num_labels = len(labels)  # number of labels
        if num_labels:
            # convert ltrb to xywh
            labels[:, 1:5] = ltrb2xywh(labels[:, 1:5])

            # Normalize coordinates 0-1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if num_labels:
                    labels[:, 1] = 1 - labels[:, 1]  # 1 - x_center

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if num_labels:
                    labels[:, 2] = 1 - labels[:, 2]  # 1 - y_center

        labels_out = torch.zeros((num_labels, 6))  # nL: number of labels
        if num_labels:
            # labels_out[:, 0] = index
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes, index

    def coco_index(self, index):
        """该方法是专门为cocotools统计标签信息准备，不对图像和标签作任何处理"""
        # load image
        # path = self.img_files[index]
        # img = cv2.imread(path)  # BGR
        # import matplotlib.pyplot as plt
        # plt.imshow(img[:, :, ::-1])
        # plt.show()

        # assert img is not None, "Image Not Found " + path
        # o_shapes = img.shape[:2]  # orig hw
        o_shapes = self.shapes[index][::-1]  # wh to hw

        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        # img = img[:, :, ::-1].transpose(2, 0, 1)
        # img = np.ascontiguousarray(img)

        # load labels
        labels = []
        x = self.labels[index]
        if x.size > 0:
            labels = x.copy()  # label: class, x, y, w, h
        return torch.from_numpy(labels), o_shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, index = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, index


def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, "Image Not Found " + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def load_mosaic(imgs, index):
    """
    将四张图片拼接在一张马赛克图像中
    :param imgs:
    :param index: 需要获取的图像索引
    :return:
    """
    # loads images in a mosaic

    labels4 = []  # 拼接图像的label信息
    s = imgs.img_size
    # 随机初始化拼接图像的中心点坐标
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    # 从dataset中随机寻找三张图像进行拼接
    indices = [index] + [random.randint(0, len(imgs.labels) - 1) for _ in range(3)]  # 3 additional image indices

    img4 = None  # base image with 4 tiles
    ll, tl, rl, bl = 0, 0, 0, 0  # large image: (l, t, r, b)
    ls, ts, rs, bs = 0, 0, 0, 0  # small image: (l, t, r, b)

    # 遍历四张图像进行拼接
    for i, index in enumerate(indices):
        # load image
        img, _, (h, w) = load_image(imgs, index)

        # place img in img4
        if i == 0:  # top left
            # 创建马赛克图像
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            ll, tl, rl, bl = max(xc - w, 0), max(yc - h, 0), xc, yc
            # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)
            ls, ts, rs, bs = w - (rl - ll), h - (bl - tl), w, h
        elif i == 1:  # top right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            ll, tl, rl, bl = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
            ls, ts, rs, bs = 0, h - (bl - tl), min(w, rl - ll), h
        elif i == 2:  # bottom left
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            ll, tl, rl, bl = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
            ls, ts, rs, bs = w - (rl - ll), 0, max(xc, w), min(bl - tl, h)
        elif i == 3:  # bottom right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            ll, tl, rl, bl = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
            ls, ts, rs, bs = 0, 0, min(w, rl - ll), min(bl - tl, h)

        # 将截取的图像区域填充到马赛克图像的相应位置
        img4[tl:bl, ll:rl] = img[ts:bs, ls:rs]  # img4[ymin:ymax, xmin:xmax]
        # 计算pad(图像边界与马赛克边界的距离，越界的情况为负值)
        pad_w = ll - ls
        pad_h = tl - ts

        # Labels 获取对应拼接图像的labels信息
        x = imgs.labels[index]
        labels = x.copy()  # 深拷贝，防止修改原数据
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            # 计算标注数据在马赛克图像中的
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + pad_w
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + pad_h
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + pad_w
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + pad_h
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=imgs.hyp['degrees'],
                                  translate=imgs.hyp['translate'],
                                  scale=imgs.hyp['scale'],
                                  shear=imgs.hyp['shear'],
                                  border=-s // 2)  # border to remove

    return img4, labels4


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, ltrb]

    # 给定的输入图像的尺寸(416/512/640)，等于img4.shape / 2
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        # [4*n, 3] -> [n, 8]
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]  # [n, 4]
        y = xy[:, [1, 3, 5, 7]]  # [n, 4]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T  # [n, 4]

        # reject warped points outside of image
        # 对坐标进行裁剪，防止越界
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]

        # 计算调整后的每个box的面积
        area = w * h
        # 计算调整前的每个box的面积
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        # 计算每个box的比例
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        # 选取长宽大于4个像素，且调整前后面积比例大于0.2，且比例小于10的box
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def letterbox(img: np.ndarray,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放调整到指定大小
    :param img:
    :param new_shape:
    :param color:
    :param auto:
    :param scale_fill:
    :param scale_up:
    :return:
    """

    shape = img.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP) 对于大于指定输入大小的图片进行缩放,小于的不变
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimun rectangle 保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍(416x416)，如果是(512x512)可以保证是64的整数倍
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scale_fill:  # stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # wh ratios

    dw /= 2  # divide padding into 2 sides 将padding分到上下，左右两侧
    dh /= 2

    # shape:[h, w]  new_unpad:[w, h]
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def create_folder(path="./new_folder"):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
