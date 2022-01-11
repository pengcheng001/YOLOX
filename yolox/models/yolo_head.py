#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.num_head = 12
        # self.eta = nn.Parameter(torch.ones(self.num_head))
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.keypoint_cls_convs = nn.ModuleList()
        self.keypoint_reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.keypoint_cls_k1 = nn.ModuleList()
        self.keypoint_cls_k2 = nn.ModuleList()
        self.keypoint_cls_k3 = nn.ModuleList()
        self.keypoint_cls_k4 = nn.ModuleList()
        self.keypoint_reg_k1 = nn.ModuleList()
        self.keypoint_reg_k2 = nn.ModuleList()
        self.keypoint_reg_k3 = nn.ModuleList()
        self.keypoint_reg_k4 = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv
        self.tanh = nn.Tanh()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.keypoint_cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.keypoint_reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.keypoint_cls_k1.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.keypoint_cls_k2.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.keypoint_cls_k3.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.keypoint_cls_k4.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.keypoint_reg_k1.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.keypoint_reg_k2.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.keypoint_reg_k3.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.keypoint_reg_k4.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.keypoint_cls_k1:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.keypoint_cls_k2:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.keypoint_cls_k3:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.keypoint_cls_k4:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.keypoint_reg_k1:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.keypoint_reg_k2:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.keypoint_reg_k3:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.keypoint_reg_k4:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # def uncertainty_loss(self, losses):
    #     assert len(losses) == len(self.eta)
    #     total_loss = torch.Tensor(losses) * torch.exp(-self.eta) * 0.5 + self.eta
    #     return total_loss.sum()

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            keypoint_cls_x = x
            keypoint_reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            keypoint_cls_feat = self.keypoint_cls_convs[k](keypoint_cls_x)
            keypoint_cls_k1_output = self.keypoint_cls_k1[k](keypoint_cls_feat)
            keypoint_cls_k2_output = self.keypoint_cls_k2[k](keypoint_cls_feat)
            keypoint_cls_k3_output = self.keypoint_cls_k3[k](keypoint_cls_feat)
            keypoint_cls_k4_output = self.keypoint_cls_k4[k](keypoint_cls_feat)

            keypoint_reg_feat = self.keypoint_reg_convs[k](keypoint_reg_x)
            keypoint_reg_k1_output = self.keypoint_reg_k1[k](keypoint_reg_feat)
            keypoint_reg_k2_output = self.keypoint_reg_k2[k](keypoint_reg_feat)
            keypoint_reg_k3_output = self.keypoint_reg_k3[k](keypoint_reg_feat)
            keypoint_reg_k4_output = self.keypoint_reg_k4[k](keypoint_reg_feat)

            # keypoint_reg_k1_output = self.tanh(keypoint_reg_k1_output)
            # keypoint_reg_k2_output = self.tanh(keypoint_reg_k2_output)
            # keypoint_reg_k3_output = self.tanh(keypoint_reg_k3_output)
            # keypoint_reg_k4_output = self.tanh(keypoint_reg_k4_output)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output,
                                    keypoint_reg_k1_output,
                                    keypoint_reg_k2_output,
                                    keypoint_reg_k3_output,
                                    keypoint_reg_k4_output,
                                    keypoint_cls_k1_output,
                                    keypoint_cls_k2_output,
                                    keypoint_cls_k3_output,
                                    keypoint_cls_k4_output, ], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid(),
                     keypoint_reg_k1_output,
                     keypoint_reg_k2_output,
                     keypoint_reg_k3_output,
                     keypoint_reg_k4_output,
                     keypoint_cls_k1_output,
                     keypoint_cls_k2_output,
                     keypoint_cls_k3_output,
                     keypoint_cls_k4_output, ], 1)

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)

            if self.decode_in_inference:
                logger.info("yolox head using self.decode_in_inference")
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                # logger.info(
                #     "yolox head not using self.decode_in_inference, output shape: {}".format(outputs.shape))
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes + 16
        bbox_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        # print("-------------------------------")
        # output[..., bbox_ch:bbox_ch +
        #        8] = (output[..., bbox_ch:bbox_ch+8] + grid.repeat(1, 1, 4)) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        bbox_ch = 5 + self.num_classes
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        outputs[..., bbox_ch:bbox_ch + 8] = torch.sign(outputs[..., bbox_ch:bbox_ch+8]) * ((torch.exp(
            torch.abs(outputs[..., bbox_ch:bbox_ch+8])) - 1)) * strides + outputs[..., 0:2].repeat(1, 1, 4)
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_ch = 5 + self.num_classes
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:bbox_ch]  # [batch, n_anchors_all, n_cls]
        keypoint_preds = outputs[:, :, bbox_ch:(bbox_ch+8)]  # [batch, n_anchors_all, n_keypt_reg]
        keypoint_cls_pres = outputs[:, :, (bbox_ch+8):]  # [batch, n_anchors_all, n_keypt_cls]

        keypoint_reg_preds_k1 = keypoint_preds[:, :, :2]
        keypoint_reg_preds_k2 = keypoint_preds[:, :, 2:4]
        keypoint_reg_preds_k3 = keypoint_preds[:, :, 4:6]
        keypoint_reg_preds_k4 = keypoint_preds[:, :, 6:8]

        keypoint_cls_pres_k1 = keypoint_cls_pres[:, :, :2]
        keypoint_cls_pres_k2 = keypoint_cls_pres[:, :, 2:4]
        keypoint_cls_pres_k3 = keypoint_cls_pres[:, :, 4:6]
        keypoint_cls_pres_k4 = keypoint_cls_pres[:, :, 6:8]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        keypoint_cls_k1_targets = []
        keypoint_cls_k2_targets = []
        keypoint_cls_k3_targets = []
        keypoint_cls_k4_targets = []
        keypoint_reg_k1_targets = []
        keypoint_reg_k2_targets = []
        keypoint_reg_k3_targets = []
        keypoint_reg_k4_targets = []
        keypoint_vis_targets = []
        vehicle_flags = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                keypoint_cls_target_k1 = outputs.new_zeros(0, 2)
                keypoint_cls_target_k2 = outputs.new_zeros(0, 2)
                keypoint_cls_target_k3 = outputs.new_zeros(0, 2)
                keypoint_cls_target_k4 = outputs.new_zeros(0, 2)
                keypoint_reg_target_k1 = outputs.new_zeros(0, 2)
                keypoint_reg_target_k2 = outputs.new_zeros(0, 2)
                keypoint_reg_target_k3 = outputs.new_zeros(0, 2)
                keypoint_reg_target_k4 = outputs.new_zeros(0, 2)
                keypoint_vis = outputs.new_zeros(0, 4)
                vehicle_flag = outputs.new_zeros(0)
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                keypoints_v = (labels[batch_idx, :num_gt, 7::3])
                # gt_keypoint_cls_per_image_k1 = labels[batch_idx, :num_gt, 7]
                # gt_keypoint_cls_per_image_k2 = labels[batch_idx, :num_gt, 10]
                # gt_keypoint_cls_per_image_k3 = labels[batch_idx, :num_gt, 13]
                # gt_keypoint_cls_per_image_k4 = labels[batch_idx, :num_gt, 16]
                gt_keypoint_cls_per_image_k1 = keypoints_v[:, 0]
                gt_keypoint_cls_per_image_k2 = keypoints_v[:, 1]
                gt_keypoint_cls_per_image_k3 = keypoints_v[:, 2]
                gt_keypoint_cls_per_image_k4 = keypoints_v[:, 3]

                gt_keypoint_reg_per_image_k1 = labels[batch_idx, :num_gt, 5:7]
                gt_keypoint_reg_per_image_k2 = labels[batch_idx, :num_gt, 8:10]
                gt_keypoint_reg_per_image_k3 = labels[batch_idx, :num_gt, 11:13]
                gt_keypoint_reg_per_image_k4 = labels[batch_idx, :num_gt, 14:16]
                gt_keypoint_vis = labels[batch_idx, :num_gt, 7::3]
                gt_categord_id = labels[batch_idx, :num_gt, 0]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                keypoint_reg_target_k1 = gt_keypoint_reg_per_image_k1[matched_gt_inds]
                keypoint_reg_target_k2 = gt_keypoint_reg_per_image_k2[matched_gt_inds]
                keypoint_reg_target_k3 = gt_keypoint_reg_per_image_k3[matched_gt_inds]
                keypoint_reg_target_k4 = gt_keypoint_reg_per_image_k4[matched_gt_inds]

                keypoint_reg_target_k1 = self.get_keypoint_reg_target(
                    outputs.new_zeros((num_fg_img, 2)),
                    keypoint_reg_target_k1,
                    expanded_strides[0][fg_mask],
                    x_shifts=x_shifts[0][fg_mask],
                    y_shifts=y_shifts[0][fg_mask],
                    bbox=reg_target
                )
                keypoint_reg_target_k2 = self.get_keypoint_reg_target(
                    outputs.new_zeros((num_fg_img, 2)),
                    keypoint_reg_target_k2,
                    expanded_strides[0][fg_mask],
                    x_shifts=x_shifts[0][fg_mask],
                    y_shifts=y_shifts[0][fg_mask],
                    bbox=reg_target


                )
                keypoint_reg_target_k3 = self.get_keypoint_reg_target(
                    outputs.new_zeros((num_fg_img, 2)),
                    keypoint_reg_target_k3,
                    expanded_strides[0][fg_mask],
                    x_shifts=x_shifts[0][fg_mask],
                    y_shifts=y_shifts[0][fg_mask],
                    bbox=reg_target


                )
                keypoint_reg_target_k4 = self.get_keypoint_reg_target(
                    outputs.new_zeros((num_fg_img, 2)),
                    keypoint_reg_target_k4,
                    expanded_strides[0][fg_mask],
                    x_shifts=x_shifts[0][fg_mask],
                    y_shifts=y_shifts[0][fg_mask],
                    bbox=reg_target

                )
                keypoint_cls_target_k1 = F.one_hot(
                    gt_keypoint_cls_per_image_k1[matched_gt_inds].to(torch.int64), 2) * pred_ious_this_matching.unsqueeze(-1)
                keypoint_cls_target_k2 = F.one_hot(
                    gt_keypoint_cls_per_image_k2[matched_gt_inds].to(torch.int64), 2) * pred_ious_this_matching.unsqueeze(-1)
                keypoint_cls_target_k3 = F.one_hot(
                    gt_keypoint_cls_per_image_k3[matched_gt_inds].to(torch.int64), 2) * pred_ious_this_matching.unsqueeze(-1)
                keypoint_cls_target_k4 = F.one_hot(
                    gt_keypoint_cls_per_image_k4[matched_gt_inds].to(torch.int64), 2) * pred_ious_this_matching.unsqueeze(-1)

                keypoint_vis = gt_keypoint_vis[matched_gt_inds]
                # vehicle_ids = [1, 10, 11]
                vehicle_flag = gt_categord_id[matched_gt_inds]
                # for v_id in vehicle_ids:
                #     # logger.info("gt_matched_classes: {}".format(gt_matched_classes))
                #     # logger.info('before flag: {}'.format(vehicle_flag))
                #     vehicle_flag += (gt_matched_classes == v_id)
                #     # logger.info('after judge id: {}, vehicle filag: {}'.format(v_id, vehicle_flag))

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            keypoint_cls_k1_targets.append(keypoint_cls_target_k1)
            keypoint_cls_k2_targets.append(keypoint_cls_target_k2)
            keypoint_cls_k3_targets.append(keypoint_cls_target_k3)
            keypoint_cls_k4_targets.append(keypoint_cls_target_k4)
            keypoint_reg_k1_targets.append(keypoint_reg_target_k1)
            keypoint_reg_k2_targets.append(keypoint_reg_target_k2)
            keypoint_reg_k3_targets.append(keypoint_reg_target_k3)
            keypoint_reg_k4_targets.append(keypoint_reg_target_k4)
            keypoint_vis_targets.append(keypoint_vis)
            vehicle_flags.append(vehicle_flag)
            # logger.info("keypoint_visshape: {}".format(keypoint_vis.shape))
            # logger.info("vehicle_flagshape: {}\n".format(vehicle_flag.shape))
            # logger.info("keypoint_vis_flat4 id shape: {}\n".format(keypoint_vis_flat4.shape))
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        keypoint_cls_k1_targets = torch.cat(keypoint_cls_k1_targets, 0)
        keypoint_cls_k2_targets = torch.cat(keypoint_cls_k2_targets, 0)
        keypoint_cls_k3_targets = torch.cat(keypoint_cls_k3_targets, 0)
        keypoint_cls_k4_targets = torch.cat(keypoint_cls_k4_targets, 0)
        keypoint_reg_k1_targets = torch.cat(keypoint_reg_k1_targets, 0)
        keypoint_reg_k2_targets = torch.cat(keypoint_reg_k2_targets, 0)
        keypoint_reg_k3_targets = torch.cat(keypoint_reg_k3_targets, 0)
        keypoint_reg_k4_targets = torch.cat(keypoint_reg_k4_targets, 0)
        keypoint_vis_targets = torch.cat(keypoint_vis_targets, 0)
        vehicle_flags = torch.cat(vehicle_flags, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        keypoint_vis_targets = (keypoint_vis_targets == 1)
        keypoint_vis_flat1 = keypoint_vis_targets[:, 0]
        keypoint_vis_flat2 = keypoint_vis_targets[:, 1]
        keypoint_vis_flat3 = keypoint_vis_targets[:, 2]
        keypoint_vis_flat4 = keypoint_vis_targets[:, 3]
        vehicle_ids = [1, 10, 11]
        vehicle_flags_target = (vehicle_flags == 0)
        for vehicle_id in vehicle_ids:
            vehicle_flags_target += (vehicle_flags == vehicle_id)
        # logger.info("vechicle id shape: {}".format(car_flag.shape))
        # logger.info("keypoint_vis_targets id shape: {}".format(keypoint_vis_targets.shape))
        # logger.info("keypoint_vis_flat4 id shape: {}\n".format(keypoint_vis_flat4.shape))
        # loss_keypoint_cls_k1 = (
        #     self.bcewithlog_loss(keypoint_cls_pres_k1.view(-1, 2)
        #                          [fg_masks][vehicle_flags_target], keypoint_cls_k1_targets[vehicle_flags_target])
        # ).sum() / vehicle_flags_target.sum()

        # loss_keypoint_cls_k2 = (
        #     self.bcewithlog_loss(keypoint_cls_pres_k2.view(-1, 2)
        #                          [fg_masks][vehicle_flags_target], keypoint_cls_k2_targets[vehicle_flags_target])
        # ).sum() / vehicle_flags_target.sum()

        # loss_keypoint_cls_k3 = (
        #     self.bcewithlog_loss(keypoint_cls_pres_k3.view(-1, 2)
        #                          [fg_masks][vehicle_flags_target], keypoint_cls_k3_targets[vehicle_flags_target])
        # ).sum() / vehicle_flags_target.sum()

        # loss_keypoint_cls_k4 = (
        #     self.bcewithlog_loss(keypoint_cls_pres_k4.view(-1, 2)
        #                          [fg_masks][vehicle_flags_target], keypoint_cls_k4_targets[vehicle_flags_target])
        # ).sum() / vehicle_flags_target.sum()

        loss_keypoint_cls_k1 = (
            self.bcewithlog_loss(keypoint_cls_pres_k1.view(-1, 2)
                                 [fg_masks], keypoint_cls_k1_targets)
        ).sum() / num_fg

        loss_keypoint_cls_k2 = (
            self.bcewithlog_loss(keypoint_cls_pres_k2.view(-1, 2)
                                 [fg_masks], keypoint_cls_k2_targets)
        ).sum() / num_fg

        loss_keypoint_cls_k3 = (
            self.bcewithlog_loss(keypoint_cls_pres_k3.view(-1, 2)
                                 [fg_masks], keypoint_cls_k3_targets)
        ).sum() / num_fg

        loss_keypoint_cls_k4 = (
            self.bcewithlog_loss(keypoint_cls_pres_k4.view(-1, 2)
                                 [fg_masks], keypoint_cls_k4_targets)
        ).sum() / num_fg

        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg

        loss_peypoint_reg_k1 = (
            self.l1_loss(keypoint_reg_preds_k1.view(-1, 2)[fg_masks][keypoint_vis_flat1],
                         keypoint_reg_k1_targets[keypoint_vis_flat1])).sum() / keypoint_vis_flat1.sum()
        loss_peypoint_reg_k2 = (
            self.l1_loss(keypoint_reg_preds_k2.view(-1, 2)[fg_masks][keypoint_vis_flat2],
                         keypoint_reg_k2_targets[keypoint_vis_flat2])).sum() / keypoint_vis_flat2.sum()
        loss_peypoint_reg_k3 = (
            self.l1_loss(keypoint_reg_preds_k3.view(-1, 2)[fg_masks][keypoint_vis_flat3],
                         keypoint_reg_k3_targets[keypoint_vis_flat3])).sum() / keypoint_vis_flat3.sum()
        loss_peypoint_reg_k4 = (
            self.l1_loss(keypoint_reg_preds_k4.view(-1, 2)[fg_masks][keypoint_vis_flat4],
                         keypoint_reg_k4_targets[keypoint_vis_flat4])).sum() / keypoint_vis_flat4.sum()

        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
            key_reg_weight = 5.0

        else:
            loss_l1 = 0.0
            key_reg_weight = 1.0

            # loss_peypoint_reg_k1 = 0.0
            # loss_peypoint_reg_k2 = 0.0
            # loss_peypoint_reg_k3 = 0.0
            # loss_peypoint_reg_k4 = 0.0

        reg_weight = 5.0
        key_cls_weight = 3.0
        obj_weight = 2.0
        loss = reg_weight * loss_iou + obj_weight * loss_obj + loss_cls + loss_l1 \
            + key_cls_weight * (loss_keypoint_cls_k1 + loss_keypoint_cls_k2 +
                                loss_keypoint_cls_k3 + loss_keypoint_cls_k4)  \
            + key_reg_weight*(loss_peypoint_reg_k1 + loss_peypoint_reg_k2 +
                              loss_peypoint_reg_k3 + loss_peypoint_reg_k4)

        return (
            loss,
            reg_weight * loss_iou,
            obj_weight * loss_obj,
            loss_cls,
            loss_l1,
            key_cls_weight * loss_keypoint_cls_k1,
            key_cls_weight * loss_keypoint_cls_k2,
            key_cls_weight * loss_keypoint_cls_k3,
            key_cls_weight * loss_keypoint_cls_k4,
            key_reg_weight * loss_peypoint_reg_k1,
            key_reg_weight * loss_peypoint_reg_k2,
            key_reg_weight * loss_peypoint_reg_k3,
            key_reg_weight * loss_peypoint_reg_k4,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    def get_keypoint_reg_target(self, keypoint_reg_target, gt, stride, x_shifts, y_shifts, bbox, eps=1e-8):
        # keypoint_reg_target[:, 0] = (gt[:, 0] / stride - x_shifts) / (bbox[:, 2]/stride)
        # keypoint_reg_target[:, 1] = (gt[:, 1] / stride - y_shifts) / (bbox[:, 3]/stride)

        # keypoint_reg_target[:, 0] = ((gt[:, 0] - bbox[:, 0]) / stride) / (bbox[:, 2]/stride)
        # keypoint_reg_target[:, 1] = ((gt[:, 1] - bbox[:, 1]) / stride) / (bbox[:, 3]/stride)
        keypoint_reg_target[:, 0] = torch.sign(
            gt[:, 0] - bbox[:, 0] + eps)*torch.log(torch.abs((gt[:, 0] - bbox[:, 0]))/stride + 1)
        keypoint_reg_target[:, 1] = torch.sign(
            gt[:, 1] - bbox[:, 1] + eps)*torch.log(torch.abs((gt[:, 1] - bbox[:, 1]))/stride + 1)
        # keypoint_reg_target[:, 0] = (((gt[:, 0] - bbox[:, 0]) / (bbox[:, 2]/2))) / stride
        # keypoint_reg_target[:, 1] = (((gt[:, 1] - bbox[:, 1]) / (bbox[:, 3]/2))) / stride
        # print(keypoint_reg_target)
        # logger.info(keypoint_reg_target[:, 1])
        # logger.info(torch.log(torch.abs((gt[:, 1] - bbox[:, 1]))/stride + 1))
        return keypoint_reg_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
