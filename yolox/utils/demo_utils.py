#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
from loguru import logger

import numpy as np

__all__ = ["mkdir", "nms", "multiclass_nms", "demo_postprocess"]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, keypoints_reg, keypoints_cls, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, keypoints_reg, keypoints_cls, scores, nms_thr, score_thr)


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def multiclass_nms_class_agnostic(boxes, keypoints_reg, keypoints_cls, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
    key_cls1 = keypoints_cls[:, 0:2].argmax(1)
    key_cls2 = keypoints_cls[:, 2:4].argmax(1)
    key_cls3 = keypoints_cls[:, 4:6].argmax(1)
    key_cls4 = keypoints_cls[:, 6:8].argmax(1)
    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_keypoints_reg = keypoints_reg[valid_score_mask]
    valid_keypoints_cls = keypoints_cls[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    key_cls1 = key_cls1[valid_score_mask]
    key_cls2 = key_cls2[valid_score_mask]
    key_cls3 = key_cls3[valid_score_mask]
    key_cls4 = key_cls4[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None],
                valid_keypoints_reg[keep], key_cls1[keep, None], key_cls2[keep, None], key_cls3[keep, None], key_cls4[keep, None]], 1
        )
    return dets


def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))
    bbox_ch = 18
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    # for o in outputs[..., bbox_ch:bbox_ch+2][0]:
    #     print(o)
    outputs[..., bbox_ch:bbox_ch+2] = np.sign(outputs[..., bbox_ch:bbox_ch+2]) * ((np.exp(
        np.abs(outputs[..., bbox_ch:bbox_ch+2])) - 1)) * expanded_strides + outputs[..., :2]
    # print(outputs[..., bbox_ch:bbox_ch+2])

    outputs[..., bbox_ch + 2:bbox_ch + 4] = np.sign(outputs[..., bbox_ch + 2:bbox_ch + 4]) * ((np.exp(
        np.abs(outputs[..., bbox_ch + 2:bbox_ch+4])) - 1)) * expanded_strides + outputs[..., :2]

    outputs[..., bbox_ch + 4:bbox_ch + 6] = np.sign(outputs[..., bbox_ch + 4:bbox_ch+6]) * ((np.exp(
        np.abs(outputs[..., bbox_ch + 4:bbox_ch+6])) - 1)) * expanded_strides + outputs[..., :2]

    outputs[..., bbox_ch + 6:bbox_ch + 8] = np.sign(outputs[..., bbox_ch + 6:bbox_ch+8]) * ((np.exp(
        np.abs(outputs[..., bbox_ch + 6:bbox_ch+8])) - 1)) * expanded_strides + outputs[..., :2]

    return outputs
