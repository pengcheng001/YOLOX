#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np
from loguru import logger

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
    "postprocess_kp_det",
    "match_keypoints",
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

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        # logger.info(image_pred.shape)
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
        detections = detections[conf_mask]
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
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def euclidean_metric(matrix_a, matrix_b):
    max_rows_count = max(len(matrix_a), len(matrix_b))
    append_count_matrix_a = max_rows_count-len(matrix_a)
    append_count_matrix_b = max_rows_count-len(matrix_b)
    append_matrix_a = np.zeros((append_count_matrix_a, 2))
    append_matrix_b = np.zeros((append_count_matrix_b, 2))
    cs = np.r_[matrix_a, append_matrix_a]
    fs = np.r_[matrix_b, append_matrix_b]
    cs_2 = np.power(cs, 2)
    fs_2 = np.power(fs, 2)
    cs_2_s = np.expand_dims(cs_2.sum(1), axis=1)
    fs_2_s = np.expand_dims(fs_2.sum(1), axis=1)
    cs_2_se = np.repeat(cs_2_s, cs_2_s.shape[0], axis=1)
    fs_2_se = np.repeat(fs_2_s, cs_2_s.shape[0], axis=1)
    TM3 = cs @ fs.T
    Dis = cs_2_se + fs_2_se.T - 2 * TM3
    return Dis

def format_keypoints(bboxes_kp_, conf):
    bboxes_kp = bboxes_kp_.clone().cpu()
    bboxes = bboxes_kp[:, 0:4]
    keypoint_reg = bboxes_kp[:, 7:15]
    keypoint_cls = bboxes_kp[:, 15:19]
    categroy = bboxes_kp[:,6]
    # bboxes /= ratio
    # keypoint_reg /= ratio
    scores = bboxes_kp[:,4] * bboxes_kp[:,5]
    slected_bbox_flag = (scores > conf)
    
def match_keypoints(bboxes_kp_, det_kp_, conf, img_info, min_matched_dis=750, is_merge=True):
    # bboxes_kp ordered as (x1, y1, x2, y2, obj_conf, class_pred, k1_reg, k2_reg, k3_reg, k4_reg, k1_vis, k2_vis, k3_vis, k4_vis)
    # det_kp ordered as (x, y, obj_conf, class_conf, class_pred)
    if bboxes_kp_ is None:
        return None
    bboxes_kp = bboxes_kp_.clone().cpu()
    det_kp = det_kp_.clone().cpu()
    ratio = img_info['ratio']
    bboxes = bboxes_kp[:, 0:4]
    keypoint_reg = bboxes_kp[:, 7:15]
    keypoint_cls = bboxes_kp[:, 15:19]
    categroy = bboxes_kp[:,6]
    # bboxes /= ratio
    # keypoint_reg /= ratio
    scores = bboxes_kp[:,4] * bboxes_kp[:,5]

    slected_bbox_flag = (scores > conf)
    slected_bbox = bboxes[slected_bbox_flag]
    slected_category = categroy[slected_bbox_flag].unsqueeze(1)
    slected_kp_reg = keypoint_reg[slected_bbox_flag]
    slected_kp_cls = keypoint_cls[slected_bbox_flag]
    slected_scores = scores[slected_bbox_flag].unsqueeze(1)
    # x,y,x,y,cate,socre,kp_reg, kp_vis
    slected_bbox_kp = torch.cat((slected_bbox, slected_category, slected_scores, slected_kp_reg, slected_kp_cls),1)
    if is_merge is False:
        return slected_bbox_kp
    # match_fg =  np.zeros((slected_bbox_kp.shape[0], 12))
   
    det_kp_scores = det_kp[:, 2] * det_kp[:, 3]
    slected_det_kp_fg = (det_kp_scores > conf)
    slected_det_kp =det_kp[slected_det_kp_fg]
    
    
    fg_vis_kp1 = (slected_bbox_kp[:, 14] == 1)
    reg_kp1_matrix = slected_bbox_kp[fg_vis_kp1][:, 6:8].numpy()
    fg_vis_kp2 = (slected_bbox_kp[:, 15] == 1)
    reg_kp2_matrix = slected_bbox_kp[fg_vis_kp2][:, 8:10].numpy()
    fg_vis_kp3 = (slected_bbox_kp[:, 16] == 1)
    reg_kp3_matrix = slected_bbox_kp[fg_vis_kp3][:, 10:12].numpy()
    fg_vis_kp4 = (slected_bbox_kp[:, 17] == 1)
    reg_kp4_matrix = slected_bbox_kp[fg_vis_kp4][:, 12:14].numpy()
    
    fg_kp1 = (slected_det_kp[:,4] == 9)
    det_kp1_matrix = slected_det_kp[fg_kp1][:,0:2].numpy()
    dis_kp1 = euclidean_metric(det_kp1_matrix, reg_kp1_matrix)
    if len(dis_kp1) >=1:
        min_index = np.argmin(dis_kp1,1)
        for i in range(len(min_index)):
            min_dis = dis_kp1[i][min_index[i]]
            if min_dis < min_matched_dis:
                reg_kp1_matrix[min_index[i]]=det_kp1_matrix[i]
  
        slected_bbox_kp[:, 6:8][fg_vis_kp1] = torch.from_numpy(reg_kp1_matrix) 
    fg_kp2 = (slected_det_kp[:,4] == 10)
    det_kp2_matrix = slected_det_kp[fg_kp2][:,0:2].numpy()
    dis_kp2 = euclidean_metric(det_kp2_matrix, reg_kp2_matrix)
    if len(dis_kp2) >=1:
        min_index = np.argmin(dis_kp2,1)
        for i in range(len(min_index)):
            min_dis = dis_kp2[i][min_index[i]]
            if min_dis < min_matched_dis:
                reg_kp2_matrix[min_index[i]]=det_kp2_matrix[i]
        slected_bbox_kp[:, 8:10][fg_vis_kp2] = torch.from_numpy(reg_kp2_matrix)
    

    fg_kp3 = (slected_det_kp[:, 4] == 11)
    det_kp3_matrix = slected_det_kp[fg_kp3][:,0:2].numpy()
    dis_kp3 = euclidean_metric(det_kp3_matrix, reg_kp3_matrix)
    
    if len(dis_kp3) >=1:
        min_index = np.argmin(dis_kp3,1)
        for i in range(len(min_index)):
            min_dis = dis_kp3[i][min_index[i]]
            if min_dis < min_matched_dis:
                reg_kp3_matrix[min_index[i]]=det_kp3_matrix[i]
        slected_bbox_kp[:, 10:12][fg_vis_kp3] = torch.from_numpy(reg_kp3_matrix)
    
    fg_kp4 = (slected_det_kp[:,4] == 12)
    det_kp4_matrix = slected_det_kp[fg_kp4][:,0:2].numpy()
    dis_kp4 = euclidean_metric(det_kp4_matrix, reg_kp4_matrix)
    if len(dis_kp4) >=1:
        min_index = np.argmin(dis_kp4,1)
        for i in range(len(min_index)):
            min_dis = dis_kp4[i][min_index[i]]
            if min_dis < min_matched_dis:
                reg_kp4_matrix[min_index[i]]=det_kp4_matrix[i]
        slected_bbox_kp[:, 12:14][fg_vis_kp4] = torch.from_numpy(reg_kp4_matrix)

    # det_kp_xy = slected_det_kp[:,0:2] / ratio
    # slected_det_kp[:, 0:2] = det_kp_xy
    
    # outline_kp = []
    # interact_kp = []
    # solo_kp = []
    # for i in range(len(slected_det_kp)):
    #     kp_x  = slected_det_kp[i][0]
    #     kp_y  = slected_det_kp[i][1]
    #     more_than_x0 =  (kp_x - slected_bbox_kp[:, 0] > -4)
    #     less_than_x1 =  (slected_bbox_kp[:, 2] - kp_x > -4)
        
    #     innner_x = less_than_x1 & more_than_x0
    #     more_than_y0 =  (kp_y - slected_bbox_kp[:, 1] > -20)
    #     less_than_y1 =  (slected_bbox_kp[:, 3] - kp_y > -20)
    #     inner_y = less_than_y1 & more_than_y0 
    #     inner = inner_y & innner_x
        
    #     if torch.sum(inner) >=1 :
    #         bbox_index = (inner ==True).nonzero(as_tuple=True)[0].cpu().numpy()[0]
    #         solo_kp.append([bbox_index, slected_det_kp[i].cpu().numpy()])
    #     # elif torch.sum(inner) >1:
    #     #     bbox_index = (inner ==True).nonzero(as_tuple=True)[0].cpu().numpy()
    #     #     interact_kp.append([bbox_index, slected_det_kp[i].cpu().numpy()])
    # for m in solo_kp:
    #     bbox_indexs = m[0]
    #     det_kp_match = m[1]
    #     kp_xy = det_kp_match[0:2]
    #     kp_xy = torch.from_numpy(kp_xy)
    #     cate = int(det_kp_match[4] - 9)
    #     cache_index = 4 + (cate*2)
    #     kp_index = 6 + (cate*2)
    #     kp_vis_index = 14 + cate
    #     for bbox_index in bbox_indexs:
    #         if match_fg[bbox_index, cate] == 0:
    #             if slected_bbox_kp[bbox_index][kp_vis_index]==0:
    #                 continue
    #             match_fg[bbox_index][cache_index:cache_index+2] = slected_bbox_kp.numpy()[bbox_index][kp_index:kp_index+2]
    #             slected_bbox_kp[bbox_index][kp_index:kp_index+2]=kp_xy
    #             slected_bbox_kp[bbox_index][kp_vis_index]=1
    #             match_fg[bbox_index, cate] =1
    #         else:
    #             reg_kp = torch.from_numpy(match_fg[bbox_index][cache_index: cache_index+2])
    #             match_kp1 = slected_bbox_kp[bbox_index][kp_index:kp_index+2]
    #             sub_dis0 = reg_kp - match_kp1
    #             sub_dis1 = reg_kp - kp_xy
    #             sub_dis0 = torch.pow(sub_dis0,2).sum()
    #             sub_dis1 = torch.pow(sub_dis1,2).sum()
    #             if sub_dis1 < sub_dis0:
    #                 slected_bbox_kp[bbox_index][kp_index:kp_index+2]=kp_xy
    return slected_bbox_kp
  

def postprocess_kp_det(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False,  kp_det_cls=[9,10,11,12]):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in range(len(prediction))]
    kp_output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        # logger.info(image_pred.shape)
        keypoint_conf_k1, keypoint_class_pre_k1 = torch.max(
            image_pred[:, (5 + num_classes + 8):(5 + num_classes + 10)], 1, keepdim=True)

        keypoint_conf_k2, keypoint_class_pre_k2 = torch.max(
            image_pred[:, 5 + num_classes + 10:5 + num_classes + 12],  1, keepdim=True)

        keypoint_conf_k3, keypoint_class_pre_k3 = torch.max(
            image_pred[:, 5 + num_classes + 12:5 + num_classes + 14], 1, keepdim=True)

        keypoint_conf_k4, keypoint_class_pre_k4 = torch.max(
            image_pred[:, 5 + num_classes + 14:5 + num_classes + 16], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred, k1_reg, k2_reg, k3_reg, k4_reg, k1_vis, k2_vis, k3_vis, k4_vis)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(
        ), image_pred[:, 5 + num_classes:5 + num_classes + 8], keypoint_class_pre_k1, keypoint_class_pre_k2, keypoint_class_pre_k3, keypoint_class_pre_k4), 1)
        detections = detections[conf_mask]
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
        kp_fg = None
        for kp_cls in  kp_det_cls:
            kp_fg_cur =  detections[:, 6] ==  kp_cls
            if kp_fg is None:
                kp_fg = kp_fg_cur
            else:
                kp_fg += kp_fg_cur
        obj_fg = (kp_fg == False)
        kp_det_out = detections[kp_fg]
        #kp det (x, y, obj_conf, cls_conf, cls_pre)
        kp_point_det = kp_det_out.new_ones((kp_det_out.shape[0], 5))
    
        kp_point_det[:,0] = ((kp_det_out[:,0] + kp_det_out[:,2]) / 2)
        kp_point_det[:,1] = ((kp_det_out[:,1] + kp_det_out[:,3]) / 2)
        kp_point_det[:,2] = kp_det_out[:,4]
        kp_point_det[:,3] = kp_det_out[:,5]
        kp_point_det[:,4] = kp_det_out[:,6]
        

        if output[i] is None:
            output[i] = detections[obj_fg]
        else:
            output[i] = torch.cat((output[i], detections[obj_fg]))
        if kp_output[i] is None:
            kp_output[i] = kp_point_det
        else:
            kp_output[i] = torch.cat((kp_output[i], kp_point_det))

    return output, kp_output



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
