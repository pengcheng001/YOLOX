#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
from loguru import logger
import numpy as np

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES, HOLO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    return parser


def preproc_(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = (input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (input_size[1], input_size[0]),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    print(input_size)
    print(resized_img.shape)
    print(padded_img.shape)
    padded_img[: input_size[0], : input_size[1]] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


if __name__ == '__main__':
    args = make_parser().parse_args()

    input_shape = [384, 768]
    origin_img = cv2.imread(args.image_path)
    img, ratio = preproc_(origin_img, [384, 768])

    session = onnxruntime.InferenceSession(args.model)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    logger.error(output[0][0][0])
    # logger.error(output[0][0])
    predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:18]
    keypoints_reg = predictions[:, 18:26]
    keypoints_cls = predictions[:, 26:34]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy[:, 0:4:2] /= ratio[1]
    boxes_xyxy[:, 1:4:2] /= ratio[0]
    keypoints_reg[:, 0:8:2] /= ratio[1]
    keypoints_reg[:, 1:8:2] /= ratio[0]
    dets = multiclass_nms(boxes_xyxy, keypoints_reg, keypoints_cls,
                          scores, nms_thr=0.45, score_thr=0.25)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds, final_key_reg, final_key_cls = dets[:,
                                                                                       :4], dets[:, 4], dets[:, 5], dets[:, 6:14], dets[:, 14:]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=args.score_thr, class_names=HOLO_CLASSES, keypoint_cls=final_key_cls, keypoint_reg=final_key_reg)

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, args.image_path.split("/")[-1])
    print(output_path)
    cv2.imwrite(output_path, origin_img)
