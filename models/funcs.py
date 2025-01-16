# in the name of God
#
from tqdm import tqdm
import torch
from ..utils.metrics import compute_pr_f1
from pycocotools.cocoeval import COCOeval



def compute_validation_loss(model, data_loader, device):
    
    model.train()
    
    running_loss_cls = 0.0
    running_loss_box = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            
            valid_images, valid_targets = zip(*[
                (img, tgt) for img, tgt in zip(images, targets)
                if tgt['boxes'].size(0) != 0
            ])
            
            if not valid_images:
                continue
            
            valid_images = list(img.to(device) for img in valid_images)
            valid_targets = [{k: v.to(device) for k, v in t.items()} for t in valid_targets]
            loss_dict = model(valid_images, valid_targets)
            loss_classifier = loss_dict.get('loss_classifier', 0.0)
            loss_box_reg = loss_dict.get('loss_box_reg', 0.0)
            running_loss_cls += float(loss_classifier)
            running_loss_box += float(loss_box_reg)
            num_batches += 1
    
    val_cls_loss = running_loss_cls / max(num_batches, 1)
    val_box_loss = running_loss_box / max(num_batches, 1)
    
    return val_cls_loss, val_box_loss


def evaluate(model, data_loader, device, coco_gt):
    """
       stats: List of 12 COCO stats from coco_eval.stats
              [AP_50_95, AP_50, AP_75, AP_small, AP_medium, AP_large,
               AR_1, AR_10, AR_100, AR_small, AR_medium, AR_large]
    """
    model.eval()
    results = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluation"):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                    box = box.to('cpu').numpy()
                    score = score.to('cpu').item()
                    label = label.to('cpu').item()

                    x1, y1 = box[0], box[1]
                    x2, y2 = box[2], box[3]
                    result = {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [box[0], box[1], box[2], box[3]],
                        "score": score
                    }
                    results.append(result)
    
    if len(results) == 0:

        empty_stats = [0.0]*12
        return empty_stats, 0.0, 0.0, 0.0
    
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    
    precision, recall, f1 = compute_pr_f1(
        results, coco_gt, iou_threshold=0.5, conf_threshold=0.5
    )
    
    return stats, precision, recall, f1



def evaluate2(model, data_loader, device, coco_gt, 
              iou_type='bbox', 
              iou_threshold=0.5, 
              conf_threshold=0.5):
    """
    Returns:
        stats: List of 12 COCO stats from coco_eval.stats:
               [AP_50_95, AP_50, AP_75, AP_small, AP_medium, AP_large,
                AR_1, AR_10, AR_100, AR_small, AR_medium, AR_large]
        precision, recall, f1_score (at iou_threshold=0.5, conf_threshold=0.5)
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluation"):
            
            images = [img.to(device) for img in images]
            
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                
                for box, score, label in zip(output['boxes'],
                                             output['scores'],
                                             output['labels']):
                    score = score.item()
                    label = label.item()
                    
                    # 1) Skip background predictions
                    if label == 0:
                        continue
                    
                    # 2) skip low-confidence predictions
                    if score < conf_threshold:
                        continue
                    
                    # Convert from [x1, y1, x2, y2] -> [x, y, w, h]
                    x1, y1, x2, y2 = box.cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    
                    result = {
                        "image_id":    image_id,
                        "category_id": label,
                        "bbox":        [x1, y1, w, h],
                        "score":       score
                    }
                    results.append(result)
    
    if len(results) == 0:
        empty_stats = [0.0]*12
        return empty_stats, 0.0, 0.0, 0.0
    
    coco_dt = coco_gt.loadRes(results)
    
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    
    precision, recall, f1 = compute_pr_f1(
        results, coco_gt, iou_threshold=iou_threshold, conf_threshold=conf_threshold
    )
    
    return stats, precision, recall, f1

