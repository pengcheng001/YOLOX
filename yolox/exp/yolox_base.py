#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_exp import BaseExp
import cv2


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 9+4
        self.depth = 1.00
        self.width = 1.00
        self.act = 'relu'
        self.export_tda4 = False

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (352, 608)  # (height, width)

        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 3
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.data_dir = None
        self.train_ann = "remain_category_train.json"
        self.val_ann = "remain_category_train.json"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 2.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 0.5
        self.enable_mixup = True

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 200
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 256.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 80
        self.min_lr_ratio = 0.005
        self.ema = True

        self.weight_decay = 5e-8
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (352, 608)
        self.test_conf = 0.25
        self.nmsthre = 0.45

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels,
                             act=self.act, export_tda4=self.export_tda4)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False, filter_bbox=False, min_input_h=15, legacy=False
    ):
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                    legacy=legacy,),
                cache=cache_img,
                filter_bbox=filter_bbox,
                min_input_h=15,
            )
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                legacy=legacy,),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1:4:2] = targets[..., 1:4:2] * scale_x
            targets[..., 5:17:3] = targets[..., 5:17:3] * scale_x
            targets[..., 2:5:2] = targets[..., 2:5:2] * scale_y
            targets[..., 6:17:3] = targets[..., 6:17:3] * scale_y
        return inputs, targets

    def filter_bbox(self, input, target):
        '''
        this is hard code, it will filter some bbox for you design
        不搞减速带了，不搞挡轮杆了，所以类别少了两类，一共9类，最后加4个车辆点的bbox，就是
        13个类别。其实原来挡轮杆和减速带的类别id是10（9），11（10）
        '''
        filter_on = True
        if filter_on:
            batch_size = target.shape[0]
            new_target = target.new_zeros(target.shape)
            nlabel = (target.sum(dim=2) > 0).sum(dim=1)
            for batch in range(batch_size):
                gt_num = nlabel[batch]
                if gt_num == 0:
                    continue
                gt = target[batch, :gt_num, :]
                remain_ann = []
                # for vehicles,
                vehicle_flag = gt[:, 0] == 0
                vehicles_gt = gt[vehicle_flag]
                vehicles_area_flag = (vehicles_gt[:, 3]*vehicles_gt[:, 4]) > 255
                vehicles_filtered = vehicles_gt[(vehicles_area_flag)]
                remain_ann.append(vehicles_filtered)

                tricycle_flag = gt[:, 0] == 1
                tricycle_gt = gt[tricycle_flag]
                tricycle_area_flag = (tricycle_gt[:, 3]*tricycle_gt[:, 4]) > 64
                tricycle_filtered = tricycle_gt[tricycle_area_flag]
                remain_ann.append(tricycle_filtered)

                cycle_flag = gt[:, 0] == 2
                cycle_gt = gt[cycle_flag]
                cycle_area_flag = (cycle_gt[:, 3]*cycle_gt[:, 4]) > 50
                cycle_filtered = cycle_gt[cycle_area_flag]
                remain_ann.append(cycle_filtered)

                barrier_gate_flag = gt[:, 0] == 8
                barrier_gate_gt = gt[barrier_gate_flag]
                barrier_gate_area_flag = (barrier_gate_gt[:, 3]*barrier_gate_gt[:, 4]) > 50
                barrier_gate_filtered = barrier_gate_gt[barrier_gate_area_flag]
                remain_ann.append(barrier_gate_filtered)

                # for pedenstrin
                pedenstrain_gt_flag = gt[:, 0] == 3
                pedenstrain_gt = gt[pedenstrain_gt_flag]
                pedenstrain_filter_width_flag = (pedenstrain_gt[:, 3] > 5)
                pedenstrain_filter_area_flag = (pedenstrain_gt[:, 3] * pedenstrain_gt[:, 4]) > 50
                pedenstrain_gt_filter_flag = pedenstrain_filter_width_flag & pedenstrain_filter_area_flag
                pedenstrain_filtered = pedenstrain_gt[pedenstrain_gt_filter_flag]
                remain_ann.append(pedenstrain_filtered)

                # cone anti_collision_bar water_horse
                cone_gt_flag = gt[:, 0] == 4
                cone_gt = gt[cone_gt_flag]
                cone_area_flag = (cone_gt[:, 3]*cone_gt[:, 4]) > 36
                cone_filtered = cone_gt[cone_area_flag]
                remain_ann.append(cone_filtered)

                anti_collision_bar_gt_flag = gt[:, 0] == 6
                anti_collison_bar_gt = gt[anti_collision_bar_gt_flag]
                anti_collision_bar_area_flag = (
                    anti_collison_bar_gt[:, 3]*anti_collison_bar_gt[:, 4]) > 36
                anti_collision_bar_filtered = anti_collison_bar_gt[anti_collision_bar_area_flag]
                remain_ann.append(anti_collision_bar_filtered)

                water_horse = gt[:, 0] == 5
                water_horse_gt = gt[water_horse]
                water_horse_area_flag = (water_horse_gt[:, 3]*water_horse_gt[:, 4]) > 49
                water_horse_filtered = water_horse_gt[water_horse_area_flag]
                remain_ann.append(water_horse_filtered)

                # for ground lock
                ground_lock_flag = (gt[:, 0] == 7)
                ground_lock_gt = gt[ground_lock_flag]
                ground_lock_area_flag = (ground_lock_gt[:, 3]*ground_lock_gt[:, 4]) > 25
                ground_lock_filtered = ground_lock_gt[ground_lock_area_flag]
                remain_ann.append(ground_lock_filtered)

                # for kp as det
                for kp_ind in range(9, 13):
                    kp_as_det = (gt[:, 0] == kp_ind)
                    kp_as_det_gt = gt[kp_as_det]
                    kp_as_det_area_flag = (kp_as_det_gt[:, 3]*kp_as_det_gt[:, 4]) > 64
                    kp_as_det_filtered = kp_as_det_gt[kp_as_det_area_flag]
                    remain_ann.append(kp_as_det_filtered)

                cat_target = torch.cat(remain_ann, 0)
                #cat_target = torch.cat((vehicles_filtered, pedenstrain_filter_gt, cones_filter_gt), 0)
                reamin_num = cat_target.shape[0]
                new_target[batch, :reamin_num, :] = cat_target

                # show
                show = False
                if show:
                    inps_cpu = input.cpu().numpy()
                    nlabel = (target.sum(dim=2) > 0).sum(dim=1)
                    for b in range(batch_size):
                        show_img = inps_cpu[b].transpose(1, 2, 0).astype(np.uint8)
                        show_img = np.ascontiguousarray(show_img)
                        gt_num = nlabel[b]
                        for box_index in range(gt_num):
                            show_bbox = new_target[b, box_index, 0:5].cpu().numpy()
                            cate_id = show_bbox[0]
                            x1 = int(show_bbox[1] - show_bbox[3] / 2)
                            y1 = int(show_bbox[2] - show_bbox[4] / 2)

                            x2 = int(show_bbox[1] + show_bbox[3] / 2)
                            y2 = int(show_bbox[2] + show_bbox[4] / 2)
                            cv2.rectangle(show_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                            cv2.putText(show_img, str(show_bbox[0])[
                                        0:4]+"-"+str(show_bbox[4])[0:4], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                            # cv2.putText(show_img, str(show_bbox[4])[0:4],(x1+10, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255),1)

                    debug_save_dir = '/home/pengcheng/code/yolox/debug_show/'
                    global save_index
                    cv2.imwrite(debug_save_dir+str(save_index)+'.jpeg', show_img)
                    save_index += 1

            return new_target
        else:
            return target

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
            name="images" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            is_val=True
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)
