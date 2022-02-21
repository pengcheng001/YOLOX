import numpy as np
import re
import os
from loguru import logger

class BoxKpInstance():
    def __init__(self):
        self.bbox= []
        self.cate = None
        self.score = None
        self.wheel_keypoints=[] 

TEXT_EXT = ['.txt']
def get_text_list(path):
    text_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in TEXT_EXT:
                text_names.append(apath)
    return text_names

def get_bbox_kp(text_path, gt=False):
    text_names = get_text_list(text_path)
    ann_map = dict()
    pattern=r'\s+'
    for text_name in text_names:
        bk = BoxKpInstance()
        with open(text_name) as f:
            basename = os.path.basename(text_name) 
            for line in f.readlines():
                
                line = line.rstrip('\r\n').strip()
                line_data = [d for d in re.split(pattern, line)]
                line_data = [float(d) for d in re.split(pattern, line)]
                
                if gt:
                    #cate x1, y1, x2, y2, kp1, kp2, kp3, kp4. kp = [x, y, v]
                    bk.cate = line_data[0]
                    kp_tmp = np.zeros((4,3))
                    for i in range(4):
                        l_index = i * 3 + 5
                        r_index = l_index + 3
                        kp = line_data[l_index:r_index]
                        if kp[2] == 1:
                            kp_tmp[i,:] = kp
                    bk.bbox.append(np.array(line_data[1:5]))
                    bk.wheel_keypoints.append(kp_tmp)
                else:
                    #cate score x1, y1, x2, y2, kp1, kp2, kp3, kp4. kp = [x, y, v]
                    bk.cate = line_data[0]
                    bk.score = line_data[1]
                    kp_tmp = np.zeros((4,3))
                    for i in range(4):
                        l_index = i * 3 + 6
                        r_index = l_index + 3
                        kp = line_data[l_index:r_index]
                        if kp[2] == 1:
                            kp_tmp[i,:] = kp
                    bk.bbox.append(np.array(line_data[2:6]))
                    bk.wheel_keypoints.append(kp_tmp)
        bk.bbox = np.array(bk.bbox)
        bk.wheel_keypoints = np.array(bk.wheel_keypoints)
        ann_map[basename] = bk
    return ann_map
 
def match_iou(bboxes1, bboxes2, thresh_conf):
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.array([])
    area1 = (bboxes1[:, 2] - bboxes1[:,0] + 1) * (bboxes1[:, 3] - bboxes1[:,1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:,0] + 1) * (bboxes2[:, 3] - bboxes2[:,1] + 1)
    matched = []
    for bbox1_index in range(len(bboxes1)):
        x1_a = bboxes1[bbox1_index,0]
        y1_a = bboxes1[bbox1_index,1]
        x2_a = bboxes1[bbox1_index,2]
        y2_a = bboxes1[bbox1_index,3]
        
        xx1 = np.maximum(x1_a, bboxes2[:,0])
        yy1 = np.maximum(y1_a, bboxes2[:,1])
        xx2 = np.minimum(x2_a, bboxes2[:,2])
        yy2 = np.minimum(y2_a, bboxes2[:,3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou_val = inter / (area2 + area1[bbox1_index] - inter)
        min_iou_index = np.argmax(iou_val)
        min_iou_val = iou_val[min_iou_index]
        if min_iou_val > thresh_conf:
            matched.append([bbox1_index, min_iou_index])
    return np.array(matched)

def compute_kps_metric(gt_dir, pre_dir, iou_threshold, kp_rad_threshold):
    gt_instances = get_bbox_kp(gt_dir, True)
    pre_instances = get_bbox_kp(pre_dir)
    error_dis_record = {}
    error_vis_record = {}
    kps_vis_TP = 0.0
    kps_vis_TF = 0.0
    kps_vis_FP = 0.0
    kps_vis_FN = 0.0
    kps_TP = 0.0
    kps_FN = 0.0
    kps_error = []
    for key in pre_instances:
        if key not in gt_instances.keys():
            logger.error("can not find the gt: {}".format(key))
        gt_instance = gt_instances[key]
        pre_instance = pre_instances[key]
        gt_bbox = gt_instance.bbox
        pre_bbox = pre_instance.bbox
        gt_wheel_kps = gt_instance.wheel_keypoints
        pre_wheel_kps = pre_instance.wheel_keypoints
        matcheds = match_iou(pre_bbox, gt_bbox, iou_threshold)
        for matched in matcheds:
            matched_pre_index = matched[0]
            matched_gt_index = matched[1]
            gt_wheel_kp =  gt_wheel_kps[matched_gt_index]
            pre_wheel_kp =  pre_wheel_kps[matched_pre_index]
            for i in range(4):
                gt_vis = gt_wheel_kp[i,2]
                gt_kp = gt_wheel_kp[i,0:2]
                pre_vis = pre_wheel_kp[i,2]
                pre_kp = pre_wheel_kp[i,0:2]
                if gt_vis == 1:
                    if pre_vis == 1:
                        kps_vis_TP += 1.0
                        dis_2 = np.power(gt_kp - pre_kp, 2).sum()
                        distance = np.sqrt(dis_2)
                        kps_error.append(distance)
                        if distance < kp_rad_threshold:
                            kps_TP += 1.0
                        else:
                            kps_FN += 1.0 
                            if key not in error_dis_record.keys():
                                error_dis_record[key] = [np.concatenate((gt_wheel_kp[i], pre_wheel_kp[i], gt_bbox[matched_gt_index]), axis = 0)]
                            else:
                                error_dis_record[key].append(np.concatenate((gt_wheel_kp[i], pre_wheel_kp[i], gt_bbox[matched_gt_index]), axis = 0))        
                    else:
                        kps_vis_FN += 1.0
                        if key not in error_vis_record.keys():
                            error_vis_record[key] = [np.concatenate((gt_wheel_kp[i], pre_wheel_kp[i], gt_bbox[matched_gt_index]), axis = 0)]
                        else:
                            error_vis_record[key].append(np.concatenate((gt_wheel_kp[i], pre_wheel_kp[i], gt_bbox[matched_gt_index]), axis = 0))      
                else:
                    if pre_vis == 1:
                        kps_vis_FP += 1.0
                        if key  not in error_vis_record.keys():
                            error_vis_record[key] = [np.concatenate((gt_wheel_kp[i], pre_wheel_kp[i], gt_bbox[matched_gt_index]), axis = 0)]
                        else:
                           error_vis_record[key].append(np.concatenate((gt_wheel_kp[i], pre_wheel_kp[i], gt_bbox[matched_gt_index]), axis = 0))
                    else:
                        kps_vis_TF += 1.0
                        
    vis_precision = kps_vis_TP / (kps_vis_TP + kps_vis_FP)
    vis_recall = kps_vis_TP / (kps_vis_TP + kps_vis_FN)
    kp_error_np = np.array(kps_error)
    kp_std = np.std(kp_error_np)
    kp_mean = np.mean(kp_error_np)
    kp_acc = kps_TP / (kps_FN + kps_TP)
    logger.warning("\nSummary of key point indicators：\nVisible point accuracy={},\ttVisible point recall={},\tKeypoint Accuracy={},\tkey point error={},\tkeypoint error standard deviation={}\n".format(vis_precision,vis_recall,kp_acc,kp_mean,kp_std))
    return (vis_precision, vis_recall, kp_acc, kp_mean, kp_std)
                
if __name__ == "__main__":
    path_gt = '/home/pengcheng/code/YOLOX/datasets/COCO/test/txt'
    path_pre = '/home/pengcheng/code/YOLOX/YOLOX_outputs20220207/yolox_s-kp_det/csv_res/2022_02_17_16_58_38'
    v_p, v_r, k_a, k_m, k_s =compute_kps_metric(path_gt, path_pre, 0.75, 10.0)
   