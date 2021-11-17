#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
from tqdm import tqdm

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.data.datasets import HOLO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        '--save_txt_res',
        action="store_true",
        help="whether to save the inference result txt of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize_h", default=None, type=int, help="test img size")
    parser.add_argument("--tsize_w", default=None, type=int, help="test img size")
    parser.add_argument("--bbox_min", default=None, type=int,
                        help="filter bboox less than the min value")
    parser.add_argument("--bbox_max", default=None, type=int,
                        help="filter bbbox more than the max value")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )

    parser.add_argument(
        "--filter_15pixel",
        dest="filter_15pixel",
        default=False,
        action="store_true",
        help="filter_15pixel.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=HOLO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def res_for_holo_banchmark(self, output, img_info, cls_conf, bbox_range=None):
        ratio = img_info['ratio']
        if output is None:
            return ""
        output = output.cpu()
        bboxes = output[:, 0:4]
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 5] * output[:, 4]

        lines = ''
        for i in range(len(bboxes)):
            category_id = int(cls[i] + 1)
            score = scores[i]
            if score < cls_conf:
                continue
            x1 = bboxes[i][0]
            y1 = bboxes[i][1]
            x2 = bboxes[i][2]
            y2 = bboxes[i][3]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            if bbox_range is not None and (h < bbox_range[0] or h > bbox_range[1]):
                print("continue the box:{}, category_id: {},  size out range: {}".format(
                    h, category_id, bbox_range))
                continue
            lines += "{} {} {} {} {} {}\n".format(category_id, score, x1, y1, x2, y2)
        return lines

    def visual(self, output, img_info, cls_conf=0.35, save_txt=False, filter_15pixel=False):
        ratio = img_info["ratio"]
        img_height = img_info['height']
        img_width = img_info['width']
        img = img_info["raw_img"]
        if output is None:
            return img, ""
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        lines = ''
        if save_txt:
            for i in range(len(bboxes)):
                vehicle_id = [1, 2, 9, 11, 12]
                category_id = int(cls[i] + 1)
                score = scores[i]
                if score < cls_conf:
                    continue
                ratio_h = img_height / 384
                ratio_w = img_width / 768

                x1 = max(bboxes[i][0], 0)
                y1 = max(bboxes[i][1], 0)
                x2 = min(bboxes[i][2], img_width - 1)
                y2 = min(bboxes[i][3], img_height - 1)
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                # if category_id not in vehicle_id and filter_15pixel and (h / ratio_h) < 15.0:
                #     continue
                # elif category_id in vehicle_id and filter_15pixel and min(w / ratio_w, h / ratio_h) < 15.0:
                #     continue
                if filter_15pixel and (h / ratio_h) < 15.0:
                    continue
                x1 = bboxes[i][0]
                y1 = bboxes[i][1]
                x2 = bboxes[i][2]
                y2 = bboxes[i][3]
                lines += "{} {} {} {} {} {}\n".format(category_id, score, x1, y1, x2, y2)
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res, lines


def image_demo(predictor, vis_folder, path, current_time, save_result, txt_folder, save_txt_res, filter_15pixel):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in tqdm(files):
        outputs, img_info = predictor.inference(image_name)
        result_image, txt_lines = predictor.visual(
            outputs[0], img_info, predictor.confthre, save_txt_res, filter_15pixel)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            # logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
            if save_txt_res:
                image_name_pre = os.path.splitext(os.path.basename(image_name))[0]
                save_folder = os.path.join(
                    txt_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                )
                os.makedirs(save_folder, exist_ok=True)
                txt_path = os.path.join(save_folder, image_name_pre+".txt")
                # res_txt_lines = predictor.res_for_holo_banchmark(outputs[0], img_info, predictor.confthre, bbox_range)
                # logger.info("Saving detection result in {}".format(txt_path))
                with open(txt_path, 'w') as f:
                    f.writelines(txt_lines)
        # ch = cv2.waitKey(0)
        # if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #     break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.save_txt_res:
        txt_folder = os.path.join(file_name, "txt_res")
        os.makedirs(txt_folder, exist_ok=True)
    else:
        txt_folder = None

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize_w is not None and args.tsize_h is not None:
        exp.test_size = (args.tsize_h, args.tsize_w)
    print(exp.test_size)
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    if args.bbox_min is not None and args.bbox_max is not None:
        bbox_range = (args.bbox_min, args.bbox_max)
    else:
        bbox_range = None
    predictor = Predictor(model, exp, HOLO_CLASSES, trt_file, decoder,
                          args.device, args.fp16, args.legacy)
    current_time = time.localtime()
    if args.demo == "image":
        # image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
        image_demo(predictor, vis_folder, args.path, current_time,
                   args.save_result, txt_folder, args.save_txt_res, args.filter_15pixel)

    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
