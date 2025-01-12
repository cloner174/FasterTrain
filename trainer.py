#
import os
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from models.fasterrcnn_mobilenet_v3_large_320_fpn import get_model
from utils.data_loader import get_data_loaders
from utils.metrics import Metrics
from utils.plotter import plot_metrics
from utils.utils import load_config, save_checkpoint, load_checkpoint, set_seed
from tqdm import tqdm
import time
import argparse
import pandas as pd
from collections import defaultdict
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    raise ImportError("Please install pycocotools: pip install pycocotools")



def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, amp=False):
    
    model.train()
    
    running_loss = defaultdict(float)
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch} Training"):
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if amp:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        
        if amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
        
        for k, v in loss_dict.items():
            running_loss[k] += v.item()
    
    avg_losses = {f"train/{k}": v / len(data_loader) for k, v in running_loss.items()}
    
    return avg_losses


def evaluate(model, data_loader, device, coco_gt, iou_threshold=0.5, max_det=300):
    
    model.eval()
    
    results = []
    running_val_loss = defaultdict(float)
    with torch.no_grad():
        
        for images, targets in tqdm(data_loader, desc="Validation"):
            
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            
            loss_dict = model(images, targets)
            for k, v in loss_dict.items():
                running_val_loss[k] += v.item()
            
            for target, output in zip(targets, outputs):
                
                image_id = target["image_id"].item()
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    
                    x1, y1, x2, y2 = box
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": bbox,
                        "score": float(score)
                    })
    
    if results:
        coco_dt = coco_gt.loadRes(results)
    else:
        coco_dt = COCO()
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.iouThrs = [iou_threshold]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    precision = coco_eval.stats[0]  # mAP@0.50
    recall = coco_eval.stats[1]     # Recall
    mAP50 = coco_eval.stats[1]
    mAP50_95 = coco_eval.stats[2]   # mAP@0.50:0.95
    
    metrics = {
        "metrics/precision(B)": precision,
        "metrics/recall(B)": recall,
        "metrics/mAP50(B)": mAP50,
        "metrics/mAP50-95(B)": mAP50_95
    }
    
    avg_val_losses = {f"val/{k}": v / len(data_loader) for k, v in running_val_loss.items()}
    
    return metrics, avg_val_losses



def main():
    
    parser = argparse.ArgumentParser(description="Faster R-CNN Training Script")
    parser.add_argument('--config', type=str, default='arg.yaml', help='Path to the config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    set_seed(config.get('seed', 0), config.get('deterministic', True))
    
    device = torch.device('cpu')
    if config['device'] == 'cpu':
        device = torch.device('cpu')
    
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']}")
    
    else:
        device = torch.device('cpu')
    
    num_classes = config['nc']
    model = get_model(pretrained=config.get('pretrained', True), num_classes=num_classes)
    model.to(device)
    
    data_loader_train, data_loader_val = get_data_loaders(
        
        config['data'],
        batch_size=config['batch'],
        num_workers=config['workers'],
        args = config,
        #img_size=config['imgsz'],
        #cache = config.get('seed', 0)
    )
    
    optimizer = None
    if config['optimizer'].lower() == 'sgd':
        
        optimizer = optim.SGD([
            {'params': model.backbone.parameters(), 'lr': config['lr0'], 'pg': 'pg0'},
            {'params': model.rpn.parameters(), 'lr': config['lr0'], 'pg': 'pg1'},
            {'params': model.roi_heads.parameters(), 'lr': config['lr0'], 'pg': 'pg2'}
        ], momentum=config['momentum'], weight_decay=config['weight_decay'])
    
    elif config['optimizer'].lower() == 'adam':
        
        optimizer = optim.Adam([
            {'params': model.backbone.parameters(), 'lr': config['lr0'], 'pg': 'pg0'},
            {'params': model.rpn.parameters(), 'lr': config['lr0'], 'pg': 'pg1'},
            {'params': model.roi_heads.parameters(), 'lr': config['lr0'], 'pg': 'pg2'}
        ], weight_decay=config['weight_decay'])
    
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    if config.get('cos_lr', False):
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['lrf'])
    else:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    amp = config.get('amp', False)
    scaler = torch.cuda.amp.GradScaler() if amp else None
    
    #metrics_logger = Metrics()
    
    save_dir = config.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    results = []
    
    coco_gt = COCO(os.path.join(config['val'], 'annotations.json'))
    
    start_epoch = 1
    if config.get('resume', False):
        checkpoint_path = os.path.join(save_dir, 'last_weights.pth')
        if os.path.exists(checkpoint_path):
            model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
            start_epoch += 1
            print(f"Resumed from checkpoint {checkpoint_path}, starting at epoch {start_epoch}")
    
    best_mAP = 0.0
    patience = config.get('patience', 20)
    patience_counter = 0
    for epoch in range(start_epoch, config['epochs'] + 1):
        epoch_start_time = time.time()
        
        train_losses = train_one_epoch(model, optimizer, data_loader_train, device, epoch, scaler, amp)
        
        val_metrics, val_losses = evaluate(model, data_loader_val, device, coco_gt, iou_threshold=config.get('iou', 0.5), max_det=config.get('max_det', 300))
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        lr_pg0 = optimizer.param_groups[0]['lr']
        lr_pg1 = optimizer.param_groups[1]['lr']
        lr_pg2 = optimizer.param_groups[2]['lr']
        
        epoch_results = {
            "epoch": epoch,
            "time": epoch_time,
            "train/box_loss": train_losses.get('train/box_loss', 0.0),
            "train/cls_loss": train_losses.get('train/cls_loss', 0.0),
            "train/dfl_loss": train_losses.get('train/dfl_loss', 0.0),
            "metrics/precision(B)": val_metrics.get("metrics/precision(B)", 0.0),
            "metrics/recall(B)": val_metrics.get("metrics/recall(B)", 0.0),
            "metrics/mAP50(B)": val_metrics.get("metrics/mAP50(B)", 0.0),
            "metrics/mAP50-95(B)": val_metrics.get("metrics/mAP50-95(B)", 0.0),
            "val/box_loss": val_losses.get('val/box_loss', 0.0),
            "val/cls_loss": val_losses.get('val/cls_loss', 0.0),
            "val/dfl_loss": val_losses.get('val/dfl_loss', 0.0),
            "lr/pg0": lr_pg0,
            "lr/pg1": lr_pg1,
            "lr/pg2": lr_pg2
        }
        
        print(f"Epoch {epoch} | Time: {epoch_time:.2f}s | Train Box Loss: {epoch_results['train/box_loss']:.4f} | "
              f"Train Cls Loss: {epoch_results['train/cls_loss']:.4f} | Train DFL Loss: {epoch_results['train/dfl_loss']:.4f}")
        
        results.append(epoch_results)
        
        current_mAP = epoch_results["metrics/mAP50-95(B)"]
        if current_mAP > best_mAP:
            best_mAP = current_mAP
            patience_counter = 0
            
            if config.get('save', True):
                save_checkpoint({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(save_dir, 'best_weights.pth'))
                print(f"Best model saved at epoch {epoch} with mAP50-95: {best_mAP:.4f}")
        
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        
        if config.get('save', True):
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(save_dir, 'last_weights.pth'))
            print(f"Last model saved at epoch {epoch}")
    
    if config.get('save', True):
        df = pd.DataFrame(results)
        csv_path = os.path.join(save_dir, 'results.csv')
        df.to_csv(csv_path, index=False)
        print(f"All epoch results saved to {csv_path}")
    
    if config.get('save', True) and config.get('plots', True):
        
        metrics_dict = defaultdict(list)
        for res in results:
            for key, value in res.items():
                if key not in ["epoch", "time", "lr/pg0", "lr/pg1", "lr/pg2"]:
                    metrics_dict[key].append(value)
        
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        plot_metrics(metrics_dict, os.path.join(save_dir, config.get('plot_path', 'metrics.png')))
        print("Training metrics saved and plotted.")


if __name__ == "__main__":
    main()

#cloner174