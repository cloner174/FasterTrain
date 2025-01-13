#
import yaml
import torch
import os
import random
import numpy as np
import pandas as pd
import json
from PIL import Image


def load_config(config_path):
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config



def save_checkpoint(epoch, model, optimizer, lr_scheduler, save_path, is_best = False):
    
    """Saves a checkpoint of the current model state."""
    save_path = save_path.replace('.pth', '_best.pth') if is_best else save_path
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
    }, save_path)




def load_checkpoint(model, filename , device, optimizer = None, lr_scheduler = None):
    
    checkpoint = torch.load(filename, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and lr_scheduler is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print(f"Model , Optimizer and lr scheduler are loaded from epoch: {checkpoint['epoch']}")
        return  model, optimizer, lr_scheduler
    
    elif optimizer is not None :
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model and Optimizer are loaded from epoch: {checkpoint['epoch']}")
        return  model, optimizer
    
    else:
        print(f"Model is loaded from epoch: {checkpoint['epoch']}")
        return  model




def set_seed(seed=0, deterministic=True):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    



def csv_to_coco(csv_file, image_dir, output_json, categories):
    """
    image_dir: دایرکتوری تصاویر
    output_json: مسیر خروجی فایل JSON
    categories: لیستی از دسته‌بندی‌ها به صورت دیکشنری [{'id':1, 'name':'class1'}, ...]
    """
    df = pd.read_csv(csv_file)
    
    category_mapping = {cat['name']: cat['id'] for cat in categories}
    
    images = []
    annotations = []
    annotation_id = 1
    image_id = 1
    image_id_mapping = {}
    for index, row in df.iterrows():
        
        img_name = row['image_name']
        img_path = os.path.join(image_dir, row['image_path'])
        
        if img_name not in image_id_mapping:
            
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except:
                print(f"Cannot open image {img_path}. Skipping...")
                continue
            
            image_info = {
                "id": image_id,
                "file_name": img_name,
                "height": height,
                "width": width
            }
            
            images.append(image_info)
            image_id_mapping[img_name] = image_id
            current_image_id = image_id
            image_id += 1
        
        else:
            current_image_id = image_id_mapping[img_name]
        
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        
        category_name = row['class']
        
        if category_name not in category_mapping:
            print(f"Category {category_name} not in category mapping. Skipping...")
            continue
        
        category_id = category_mapping[category_name]
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        area = bbox_width * bbox_height
        
        annotation = {
            "id": annotation_id,
            "image_id": current_image_id,
            "category_id": category_id,
            "bbox": [xmin, ymin, bbox_width, bbox_height],
            "area": area,
            "iscrowd": 0
        }
        
        annotations.append(annotation)
        annotation_id += 1
    
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=4)
    
    print(f"Successfully converted {csv_file} to {output_json}")



#cloner174