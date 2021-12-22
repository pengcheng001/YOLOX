from yolox.data import COCODataset, ValTransform
from yolox.exp import get_exp
import cv2
import numpy as np


import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import Trainer, launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_omp, get_num_devices
from yolox.data import DataPrefetcher

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser




args = make_parser().parse_args()
exp = get_exp(args.exp_file, args.name)
exp.merge(args.opts)

if not args.experiment_name:
    args.experiment_name = exp.exp_name
# exp.get_data_loader(2, False)

train_loader = exp.get_data_loader(
        batch_size=2,
        is_distributed=False,
        no_aug=False,
        cache_img=False,
    )

prefetcher = DataPrefetcher(train_loader)


colors = [[255, 50, 15], [203, 150, 255],
         [25, 250, 20], [200, 15, 175]]
    # print(inp.shape) 
    # print(tar.shape)
    # print("end\n\n")


# datsset = exp.dataset
# train_loader = iter(train_loader)
# for i in range(5000):
#     img, labels, img_info, img_id = next(train_loader)
#     # img, labels, img_info, img_id = datsset[[True, i]]
for i in range(500):
    print('start---')
    img, labels = prefetcher.next()
    img = img[0].cpu().numpy()
    labels = labels[0].cpu().numpy()
    img = img.transpose((1,2,0))
    img = np.ascontiguousarray(img, dtype=np.uint8)
    for label in labels:
        cls_id = int(label[0])
        box = label[1:5]
        keypt1 = label[5:8]
        keypt2 = label[8:11]
        keypt3 = label[11:14]
        keypt4 = label[14:17]
        if box[2] <=0 or box[3] <=0:
            continue
        else:
            x1 = int(box[0] - box[2] / 2)
            y1 = int(box[1] - box[3] / 2)
            x2 = int(box[2] + x1)
            y2 = int(box[3] + y1)
            cv2.rectangle(img, (x1, y1), (x2,y2), (0,255,0), 1)
            cv2.putText(img, str(cls_id), (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (205, 15, 12), 1)
            if keypt1[2] != 0:
                cv2.circle(
                        img, (int(keypt1[0]), int(keypt1[1])), 2, colors[0], 2)
                cv2.putText(img, '1', (int(keypt1[0]), int(keypt1[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 15, 255), 2)
            if keypt2[2] != 0:
                cv2.circle(
                        img, (int(keypt2[0]), int(keypt2[1])), 2, colors[1], 2)
                cv2.putText(img, '2', (int(keypt2[0]), int(keypt2[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 15, 255), 2)
            if keypt3[2] != 0:
                cv2.circle(
                        img, (int(keypt3[0]), int(keypt3[1])), 2, colors[2], 2)
                cv2.putText(img, "3", (int(keypt3[0]), int(keypt3[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 15, 255), 2)
            if keypt4[2] != 0:
                cv2.circle(
                        img, (int(keypt4[0]), int(keypt4[1])), 2, colors[3], 2)
                cv2.putText(img, "4", (int(keypt4[0]), int(keypt4[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 15, 255), 2)
    cv2.imshow("show", img)
    # image = img[:, ::-1]
    # cv2.imshow("show_mr", image)
    
    cv2.waitKey(0)