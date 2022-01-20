#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False,  kp_det_cls=[13, 14, 15, 16]):
    box_corner = prediction.new(prediction.shape)
    bbox_corner_kp_cache = box_corner.copy()
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in range(len(prediction))]
    kp_det = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        keypoint_conf_k1, keypoint_class_pre_k1 = torch.max(
            image_pred[:, (5 + num_classes + 8):(5 + num_classes + 10)], 1, keepdim=True)

        keypoint_conf_k2, keypoint_class_pre_k2 = torch.max(
            image_pred[:, 5 + num_classes + 10:5 + num_classes + 12],  1, keepdim=True)

        keypoint_conf_k3, keypoint_class_pre_k3 = torch.max(
            image_pred[:, 5 + num_classes + 12:5 + num_classes + 14], 1, keepdim=True)

        keypoint_conf_k4, keypoint_class_pre_k4 = torch.max(
            image_pred[:, 5 + num_classes + 14:5 + num_classes + 16], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(
        ), image_pred[:, 5 + num_classes:5 + num_classes + 8], keypoint_class_pre_k1, keypoint_class_pre_k2, keypoint_class_pre_k3, keypoint_class_pre_k4), 1)
        # Detections ordered as (cx, cy, class_conf, class_pred)
        bbox_corner_kp_caches = torch.cat(
            (bbox_corner_kp_cache[i, :, :2], class_conf, class_pred.float(),), 1)
        detections = detections[conf_mask]
        bbox_corner_kp_caches = bbox_corner_kp_caches[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        bbox_corner_kp_caches = bbox_corner_kp_caches[nms_out_index]
        kp_det_flag = None
        kp_id = 0
        for kp_cls in kp_det_cls:
            select_cls_flag = (bbox_corner_kp_caches[:, 3] == kp_cls)
            kp_det_flag = (kp_det_flag or select_cls_flag)
            bbox_corner_kp_caches[select_cls_flag][:, 3] = kp_id
            kp_id += 1
        det_flag = [True ^ f for f in kp_det_flag]
        bbox_corner_kp_caches = bbox_corner_kp_caches[kp_det_flag]
        detections = detections[det_flag]
        if output[i] is None:
            output[i] = detections
            kp_det = bbox_corner_kp_caches
        else:
            output[i] = torch.cat((output[i], detections))
            kp_det[i] = torch.cat((kp_det[i], bbox_corner_kp_caches))

    return output, kp_det


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def adjust_keypoint_ann(keypoints, scale_ratio, padw, padh, w_max, h_max):
    filtered_flag = ((keypoints[:, ::3] * scale_ratio + padw) < 0) \
        + ((keypoints[:, ::3] * scale_ratio + padw) > w_max) \
        + ((keypoints[:, 1::3] * scale_ratio + padh) > h_max) \
        + ((keypoints[:, 1::3] * scale_ratio + padh) < 0)
    keypoints[:, 2::3][filtered_flag] = 0
    keypoints[:, ::3] = keypoints[:, ::3] * scale_ratio
    keypoints[:, 1::3] = keypoints[:, 1::3] * scale_ratio
    return keypoints


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes
