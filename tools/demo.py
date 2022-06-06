#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch
import numpy as np
from tqdm import trange

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import HOLO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, vis, match_keypoints
from yolox.utils import postprocess_kp_det as postprocess

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
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
        default="cpu",
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
            outputs, kp_det = postprocess(
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

        return outputs, kp_det, img_info

    def visual(self, output, img_info, cls_conf=0.35, filter=None, max_pix=15.0):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        img_height = img_info['height']
        img_width = img_info['width']
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

        lines = ''
        if True:
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
                down_size = h / ratio_h
                # if category_id not in vehicle_id and filter_15pixel and (h / ratio_h) < 15.0:
                #     continue
                # elif category_id in vehicle_id and filter_15pixel and min(w / ratio_w, h / ratio_h) < 15.0:
                #     continue

                if down_size < max_pix and filter:
                    continue
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
        vis_res = vis(img, bboxes, scores, cls, cls_conf,
                      self.cls_names, keypoint_reg, keypoint_cls, ratio_h, max_pix)
        return vis_res, lines

    def visual_det_kp(self, output, img_info, cls_conf=0.35, filter=None, max_pix=15.0):
        img = img_info["raw_img"]
        ratio = img_info["ratio"]
        img_height = img_info['height']
        img_width = img_info['width']
        if output is None:
            return img, ""
        bboxes = output[:, 0:4]
        keypoint_reg = output[:, 6:14]
        keypoint_cls = output[:, 14:18]
        scores = output[:, 5]
        cls = output[:, 4]
        bboxes /= ratio
        keypoint_reg /= ratio
        vis_res = vis(img, bboxes, scores, cls, cls_conf,
                      self.cls_names, keypoint_reg, keypoint_cls)
        lines = ''
        if True:
            for i in range(len(bboxes)):
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
                down_size = h / ratio_h
                if down_size < max_pix and filter:
                    continue
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

        return vis_res, lines


def visual_kp_det(img, kp_det, img_info, cls_conf=0.25):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    if kp_det is None:
        return img
    kp_det = kp_det.cpu()
    keypoints = kp_det[:, 0:2]
    keypoints /= ratio
    cls = kp_det[:, 4]
    scores = kp_det[:, 2] * kp_det[:, 3]
    kp_cls = kp_det[:, 4].numpy()
    for i in range(len(keypoints)):
        if scores[i] < cls_conf:
            continue
        # logger.error(keypoints)
        vis_kp = [int(keypoints[i][0]), int(keypoints[i][1])]
        cv2.circle(img, vis_kp, 3, (0, 255, 15), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(int(kp_cls[i] - 9)), (int(keypoints[i][0]) +
                    5, int(keypoints[i][1]) + 5), font, 0.5,  (0, 255, 15), 2)
    return img


# def image_demo(predictor, vis_folder, path, current_time, save_result):
#     if os.path.isdir(path):
#         files = get_image_list(path)
#     else:
#         files = [path]
#     files.sort()
#     for image_name in files:
#         outputs, kp_det, img_info = predictor.inference(image_name)
#         if save_result:
#             save_folder = os.path.join(
#                 vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
#             )
#             save_img_name = image_name[len(path):len(image_name)]
#             # save_file_name = os.path.join(save_folder, sub_dir, os.path.basename(image_name))
#             save_file_name = os.path.join(save_folder, save_img_name)
#             os.makedirs(os.path.split(save_file_name)[0], exist_ok=True)
#         if outputs[0] is None:
#             save_img = cv2.imread(image_name)
#             logger.info("Saving detection result in {}".format(save_file_name))
#             cv2.imwrite(save_file_name, save_img)
#             continue

#         match_result = match_keypoints(outputs[0], kp_det[0], 0.25, img_info)
#         result_image, lines = predictor.visual_det_kp(match_result, img_info, predictor.confthre)
#         # visual_kp_det(result_image, kp_det[0], img_info, predictor.confthre)
#         if save_result:
#             logger.info("Saving detection result in {}".format(save_file_name))
#             cv2.imwrite(save_file_name, result_image)
#         ch = cv2.waitKey(0)
#         if ch == 27 or ch == ord("q") or ch == ord("Q"):
    # break
def image_demo(predictor, vis_folder, csv_folder, path, current_time, save_result, save_as_txt, filter_15pixel):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()

    with trange(len(files)) as t:
        for i in t:
            image_name = files[i]
            post_pix = {}
            outputs, kp_det, img_info = predictor.inference(image_name, post_pix)
            match_result = match_keypoints(outputs[0], kp_det[0], 0.25, img_info, is_merge=True)

            result_image, res_lines = predictor.visual_det_kp(
                match_result, img_info, predictor.confthre, filter_15pixel, 15.0)
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
