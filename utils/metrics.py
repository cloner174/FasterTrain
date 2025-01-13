#
import torch
from collections import defaultdict


class Metrics:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(list)
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def get_metrics(self):
        return {k: sum(v)/len(v) for k, v in self.metrics.items()}
    
    def get_all_metrics(self):
        return self.metrics
    


# Functions at fixed IoU=0.5 and conf=0.5

def iou(boxA, boxB):
    """
    boxA, boxB in [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union


def compute_pr_f1(results, coco_gt, iou_threshold=0.5, conf_threshold=0.5):
    """
    results: list of dicts with keys:
       'image_id', 'category_id', 'bbox', 'score'
    coco_gt: COCO object for ground truth
    """
    
    # Filter out predictions below confidence threshold
    filtered_preds = [
        r for r in results
        if r['score'] >= conf_threshold
    ]
    
    # Build GT structures
    gt_boxes_map = {}
    for img_id in coco_gt.getImgIds():
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        
        boxes = []
        cats = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x2, y2 = x + w, y + h
            boxes.append([x, y, x2, y2])
            cats.append(ann['category_id'])
        
        gt_boxes_map[img_id] = {
            'boxes': boxes,
            'cats': cats,
            'matched': [False] * len(boxes)
        }
    
    TP = 0
    FP = 0
    for pred in filtered_preds:
        img_id = pred['image_id']
        
        # Convert pred bbox to [x1, y1, x2, y2]
        x1, y1, w, h = pred['bbox']
        pred_box = [x1, y1, x1 + w, y1 + h]
        pred_cat = pred['category_id']
        best_iou = 0
        best_gt_idx = -1
        
        # Search all GT boxes in this image
        for i, gt_box in enumerate(gt_boxes_map[img_id]['boxes']):
            if gt_boxes_map[img_id]['matched'][i]:
                # Already matched with a higher-score detection
                continue
            
            gt_cat = gt_boxes_map[img_id]['cats'][i]
            iou_val = iou(pred_box, gt_box)
            if (iou_val > best_iou) and (gt_cat == pred_cat):
                best_iou = iou_val
                best_gt_idx = i
        
        # Decide if we have a TP
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            TP += 1
            gt_boxes_map[img_id]['matched'][best_gt_idx] = True
        else:
            FP += 1
    
    # Count FN
    FN = 0
    for img_id in gt_boxes_map:
        for matched_flag in gt_boxes_map[img_id]['matched']:
            if not matched_flag:
                FN += 1
    
    precision = TP / (TP + FP + 1e-16)
    recall    = TP / (TP + FN + 1e-16)
    f1        = 2 * precision * recall / (precision + recall + 1e-16)
    
    return precision, recall, f1


#cloner174