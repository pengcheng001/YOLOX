#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from time import sleep
from loguru import logger

import cv2
import numpy as np
from pycocotools.coco import COCO

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="images",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        filter_bbox=False,
        min_input_h=15,
        class_num=9,
        is_val=False,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "parking/HOLO_2D")
            # data_dir = os.path.join(get_yolox_datadir())
        self.data_dir = data_dir
        self.json_file = json_file
        self.is_val = is_val

        logger.info(os.path.join(self.data_dir, "annotations", self.json_file))
        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))

        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.class_ids = self.class_ids[0:class_num]
        cats = self.coco.loadCats(self.class_ids)
        # for kp det
        for i in range(class_num, class_num+4):
            self.class_ids.append(i+1)
            cats.append({"id": i+1, 'name': "kp_det_{}".format(i - class_num)})
        self._classes = tuple([c["name"] for c in cats])
        self.imgs = None
        self.name = name
        self.img_size = img_size
        self.filter_bbox = filter_bbox
        self.min_input_h = min_input_h
        self.preproc = preproc
        self.class_num = class_num
        self.annotations = self._load_coco_annotations()
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.data_dir + "/img_resized_kp_as_det_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            logger.info("Cache images to numpy, Shape: {}".format((len(self.ids), max_h, max_w)))
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def filter_bbox_for_val(self, target):
        filter_on = True
        if filter_on:
            gt = target
            remain_ann = []
            # for vehicles,
            vehicle_flag = gt[:, 0] == 0
            vehicles_gt = gt[vehicle_flag]
            vehicles_area_flag = (vehicles_gt[:, 3]*vehicles_gt[:, 4]) > 400
            vehicles_filtered = vehicles_gt[(vehicles_area_flag)]
            remain_ann.append(vehicles_filtered)

            tricycle_flag = gt[:, 0] == 1
            tricycle_gt = gt[tricycle_flag]
            tricycle_area_flag = (tricycle_gt[:, 3]*tricycle_gt[:, 4]) > 300
            tricycle_filtered = tricycle_gt[tricycle_area_flag]
            remain_ann.append(tricycle_filtered)

            cycle_flag = gt[:, 0] == 2
            cycle_gt = gt[cycle_flag]
            cycle_area_flag = (cycle_gt[:, 3]*cycle_gt[:, 4]) > 200
            cycle_filtered = cycle_gt[cycle_area_flag]
            remain_ann.append(cycle_filtered)

            barrier_gate_flag = gt[:, 0] == 8
            barrier_gate_gt = gt[barrier_gate_flag]
            barrier_gate_area_flag = (barrier_gate_gt[:, 3]*barrier_gate_gt[:, 4]) > 300
            barrier_gate_filtered = barrier_gate_gt[barrier_gate_area_flag]
            remain_ann.append(barrier_gate_filtered)

            # for pedenstrin
            pedenstrain_gt_flag = gt[:, 0] == 3
            pedenstrain_gt = gt[pedenstrain_gt_flag]
            pedenstrain_filter_width_flag = (pedenstrain_gt[:, 3] > 15)
            pedenstrain_filter_area_flag = (pedenstrain_gt[:, 3] * pedenstrain_gt[:, 4]) > 200
            pedenstrain_gt_filter_flag = pedenstrain_filter_width_flag & pedenstrain_filter_area_flag
            pedenstrain_filtered = pedenstrain_gt[pedenstrain_gt_filter_flag]
            remain_ann.append(pedenstrain_filtered)

            # cone anti_collision_bar water_horse
            cone_gt_flag = gt[:, 0] == 4
            cone_gt = gt[cone_gt_flag]
            cone_area_flag = (cone_gt[:, 3]*cone_gt[:, 4]) > 100
            cone_filtered = cone_gt[cone_area_flag]
            remain_ann.append(cone_filtered)

            anti_collision_bar_gt_flag = gt[:, 0] == 6
            anti_collison_bar_gt = gt[anti_collision_bar_gt_flag]
            anti_collision_bar_area_flag = (
                anti_collison_bar_gt[:, 3]*anti_collison_bar_gt[:, 4]) > 352
            anti_collision_bar_filtered = anti_collison_bar_gt[anti_collision_bar_area_flag]
            remain_ann.append(anti_collision_bar_filtered)

            water_horse = gt[:, 0] == 5
            water_horse_gt = gt[water_horse]
            water_horse_area_flag = (water_horse_gt[:, 3]*water_horse_gt[:, 4]) > 300
            water_horse_filtered = water_horse_gt[water_horse_area_flag]
            remain_ann.append(water_horse_filtered)

            # for ground lock
            ground_lock_flag = (gt[:, 0] == 7)
            ground_lock_gt = gt[ground_lock_flag]
            ground_lock_area_flag = (ground_lock_gt[:, 3]*ground_lock_gt[:, 4]) > 200
            ground_lock_filtered = ground_lock_gt[ground_lock_area_flag]
            remain_ann.append(ground_lock_filtered)

            # for kp as det
            for kp_ind in range(9, 13):
                kp_as_det = (gt[:, 0] == kp_ind)
                kp_as_det_gt = gt[kp_as_det]
                kp_as_det_area_flag = (kp_as_det_gt[:, 3]*kp_as_det_gt[:, 4]) > 25.0
                kp_as_det_filtered = kp_as_det_gt[kp_as_det_area_flag]
                remain_ann.append(kp_as_det_filtered)

            new_target = np.concatenate(remain_ann, 0)
            return new_target
        else:
            return target

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        ann_kp_ad_dets = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                bbox_h = y2 - y1 + 1
                bbox_w = x2 - x1 + 1
                obj["clean_bbox"] = [x1, y1, x2, y2]
                if("keypoints" in obj.keys()):
                    if len(obj['keypoints']) == 0:
                        obj["keypoints"] = [
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                        ]
                    elif len(obj['keypoints']) < 4:
                        kp_count = len(obj['keypoints'])
                        for i in range(kp_count, 4):
                            obj['keypoints'].append([0, 0, 0])
                    else:
                        assert len(obj['keypoints']) == 4
                        for kp_ind in range(4):
                            assert len(obj['keypoints'][kp_ind]) == 3
                            if obj['keypoints'][kp_ind][2] == 1 and obj['category_id'] == 1:  # only using vechile
                                kp_bbox_rad_h = bbox_h / 4
                                kp_bbox_rad_w = bbox_w / 4
                                kp_bbox = [
                                    obj['keypoints'][kp_ind][0] - kp_bbox_rad_w,
                                    obj['keypoints'][kp_ind][1] - kp_bbox_rad_h,
                                    obj['keypoints'][kp_ind][0] + kp_bbox_rad_w,
                                    obj['keypoints'][kp_ind][1] + kp_bbox_rad_h,
                                    kp_ind + self.class_num,
                                ]
                                ann_kp_ad_dets.append(kp_bbox)
                else:
                    obj["keypoints"] = [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ]
                objs.append(obj)

        num_objs = len(objs) + len(ann_kp_ad_dets)

        res = np.zeros((num_objs, 17))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0: 4] = obj["clean_bbox"]
            res[ix, 4] = cls
            for ik in range(4):
                res[ix, ik*3+5: ik*3+8] = obj["keypoints"][ik]
                res[ix, ik*3+5: ik*3+7] *= r
        for kp_det_ind in range(len(objs), num_objs):
            kp_det = ann_kp_ad_dets[kp_det_ind - len(objs)]
            res[kp_det_ind, : 5] = kp_det
            # for ik in range(4):
            #     res[ix, ik*3+5: ik*3+8] = [0, 0, 0]
        res[:, : 4] *= r
        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        # if self.is_val:
        #     res = self.filter_bbox_for_val(res)
        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpeg"
        )
        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        if img is None:
            logger.error("can find the image: {}".format(img_file))
        assert img is not None

        return img

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        return img, res.copy(), img_info, np.array([id_])

    @ Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
