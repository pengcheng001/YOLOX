#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
from tqdm import trange
import onnxruntime

import cv2

import torch
import numpy as np

from yolox.data.data_augment import ValTransform
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
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.25, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
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
        "--txt",
        dest="txt",
        default=False,
        action="store_true",
        help="write res into txt.",
    )
    parser.add_argument(
        "--filter_15pix",
        dest="filter_15pix",
        default=False,
        action="store_true",
        help="filter 15pixel.",
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
    # logger.warning(image_names)
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

    def inference(self, img, infer_time_out=None):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img_path = img
            img = cv2.imread(img_path)
            try:
                img_show = img.copy()
            except:
                logger.error(img_path)
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
            if infer_time_out is not None:
                infer_time_out["Infer_time"] = "{:.4f}s".format(time.time() - t0)

        # img = img[0].cpu().numpy().copy()
        # img = img.transpose((1, 2, 0))
        # img = np.ascontiguousarray(img, dtype=np.uint8)
        # img_info["raw_img"] = img
        # img_info["ratio"] = 1
        if outputs is None:
            cv2.imshow('img', img_show)
            cv2.waitKey(0)
        return outputs, img_info

    def whether_slected(self, bbox_in, cate_id, model_input_size, images_size):
        model_input_height = model_input_size[0]
        model_input_width = model_input_size[1]
        image_size_h = images_size[0]
        image_size_w = images_size[1]
        down_sample_h_ratio = image_size_h / model_input_height
        down_sample_w_ratio = image_size_w / model_input_width
        filter_height = model_input_height / 3.0
        bbox = bbox_in.copy()
        # logger.info(bbox)
        bbox[0] = bbox[0] / down_sample_w_ratio
        bbox[2] = bbox[2] / down_sample_w_ratio
        bbox[1] = bbox[1] / down_sample_h_ratio
        bbox[3] = bbox[3] / down_sample_h_ratio
        # for vehicles
        if cate_id in [1, 2, 3, 9]:
            if bbox[2] * bbox[3] > 200:
                return True
            else:
                return False
        # for pedenstrin
        if cate_id in [4]:
            if bbox[2] > 5 and (bbox[2]*bbox[3]) > 90:
                return True
            else:
                return False
        # cone anti_collision_bar water_horse
        if cate_id in [5, 6, 7]:
            if bbox[2]*bbox[3] > 100:
                return True
            else:
                return False
        # for ground lock
        if cate_id in [8]:
            if ((bbox[1]+bbox[3])) > filter_height and (bbox[2] * bbox[3]) > 100:
                return True
            else:
                return False
        # for weel rod,speed_bump
        if cate_id in [10, 11]:
            if (bbox[2] > 9) and (bbox[3] > 9):
                return True
            else:
                return False
        return False

    def visual(self, output, img_info, cls_conf=0.35, filter=None, max_pix=15.0):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        img_height = img_info['height']
        img_width = img_info['width']
        model_input_h = 352
        model_input_w = 608
        if output is None:
            return img, ""
        output = output.cpu()

        bboxes = output[:, 0:4]
        keypoint_reg = output[:, 7:15]
        keypoint_cls = output[:, 15:19]

        # preprocessing: resize
        bboxes /= ratio
        keypoint_reg /= ratio
        # print(ratio)

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        filter_flag = [True for _ in range(len(bboxes))]
        slected_cate_id = [1, 2, 3, 4, 5, 10]

        lines = ''
        # if True:
        for i in range(len(bboxes)):
            # vehicle_id = [1, 2, 9, 11, 12]
            category_id = int(cls[i] + 1)
            if category_id not in slected_cate_id:
                continue

            score = scores[i]
            if score < cls_conf:
                continue
            # ratio_h = img_height / model_input_h
            # ratio_w = img_width / model_input_w

            # ratio_h = 720 / 256.0
            # ratio_w = 1280 / 512.0

            # resize_ratio_h = 1080.0 / 720.0
            # resize_ratio_w = 1920.0 / 1280.0

            # x1 = max(bboxes[i][0], 0) / resize_ratio_w
            # y1 = max(bboxes[i][1], 0) / resize_ratio_h
            # x2 = min(bboxes[i][2], img_width - 1) / resize_ratio_w
            # y2 = min(bboxes[i][3], img_height - 1) / resize_ratio_h
            # box = bboxes[i].copy()

            # model_sacle_bbox[0] = model_sacle_bbox[0] * ratio_w
            # model_sacle_bbox[2] = model_sacle_bbox[2] * ratio_w

            # model_sacle_bbox[1] = model_sacle_bbox[1] * ratio_h
            # model_sacle_bbox[3] = model_sacle_bbox[3] * ratio_h

            x1 = max(bboxes[i][0], 0)
            y1 = max(bboxes[i][1], 0)
            x2 = min(bboxes[i][2], img_width - 1)
            y2 = min(bboxes[i][3], img_height - 1)

            w = x2 - x1 + 1
            h = y2 - y1 + 1
            box = [x1, y1, w, h]
            slected_flag = self.whether_slected(
                box, category_id, (model_input_h, model_input_w), (img_height, img_width))
            if not slected_flag:
                continue
            filter_flag[i] = False
            # down_size = h / ratio_h
            # down_size_w = w / ratio_w

            # if (1 == category_id) and down_size < 15:
            #     continue
            # elif 10 == category_id and down_size_w < 15:
            #     continue
            # elif down_size < 5:
            #     continue
            # # if category_id not in vehicle_id and filter_15pixel and (h / ratio_h) < 15.0:
            # #     continue
            # # elif category_id in vehicle_id and filter_15pixel and min(w / ratio_w, h / ratio_h) < 15.0:
            # #     continue

            # # if down_size < max_pix and filter:
            #     continue
            x1 = bboxes[i][0]
            y1 = bboxes[i][1]
            x2 = bboxes[i][2]
            y2 = bboxes[i][3]
            lines += "{},{},{},{},{},{},".format(category_id, score, x1, y1, x2, y2)
            for j in range(4):
                if keypoint_cls[i][j] == 1:
                    lines += "{},{},{}, ".format(
                        keypoint_reg[i][0 + j*2], keypoint_reg[i][1 + j*2], 1)
                else:
                    lines += "{},{},{},".format(
                        0, 0, 0, 0, 0)
            lines += "\n"
        # vis_res = vis(img, bboxes, scores, cls, cls_conf,
        #               self.cls_names, keypoint_reg, keypoint_cls, ratio_h, max_pix)
        vis_res = vis(img, bboxes, scores, cls, cls_conf,
                      self.cls_names, keypoint_reg, keypoint_cls, filter_flag)
        return vis_res, lines


def image_demo(predictor, vis_folder, csv_folder, path, current_time, save_result, save_as_txt, filter_15pixel):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    name_index = 0
    with trange(len(files)) as t:
        for i in t:
            image_name = files[i]
            if image_name is None:
                continue
            post_pix = {}
            outputs, img_info = predictor.inference(image_name, post_pix)
            result_image, res_lines = predictor.visual(
                outputs[0], img_info, predictor.confthre, filter_15pixel, 15.0)
            base_file_name = image_name[len(path)+1:]
            base_sub_dir = os.path.split(base_file_name)[0]
            t.set_description("INFO {}:".format(i))
            if save_result:
                save_folder = os.path.join(
                    vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                )
                save_file_name = os.path.join(save_folder, base_sub_dir)
                os.makedirs(save_file_name, exist_ok=True)
                save_file_name = os.path.join(save_file_name, os.path.basename(image_name))
                # save_file_name = os.path.join(save_file_name, str(name_index)+".jpeg")
                name_index += 1
                post_pix["det_res_path"] = save_file_name
                # logger.info("Saving detection result in {}".format(save_file_name))
                cv2.imwrite(save_file_name, result_image)
            if True:
                image_name_pre = os.path.splitext(os.path.basename(image_name))[0]
                save_folder = os.path.join(
                    csv_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                )
                txt_path = os.path.join(save_folder, base_sub_dir)
                os.makedirs(txt_path, exist_ok=True)
                if save_as_txt:
                    txt_path = os.path.join(txt_path, image_name_pre+".txt")
                else:
                    txt_path = os.path.join(txt_path, image_name_pre+".csv")
                with open(txt_path, 'w') as f:
                    if save_as_txt:
                        res_lines = res_lines.replace(",", " ")
                    f.writelines(res_lines)
            t.set_postfix(post_pix)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        csv_folder = os.path.join(file_name, "csv_res")
        os.makedirs(vis_folder, exist_ok=True)
        os.makedirs(csv_folder, exist_ok=True)

    args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("The test size: {}".format(exp.test_size))
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
    trt_file = None
    decoder = None
    predictor = Predictor(
        model, exp, HOLO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()

    image_demo(predictor, vis_folder, csv_folder, args.path, current_time,
               args.save_result, args.txt, args.filter_15pix)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
