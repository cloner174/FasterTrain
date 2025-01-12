import os
import numpy as np
import torch
import yaml
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
#from utils.utils import csv_to_coco


class CustomDataset(Dataset):
    
    def __init__(self, annotations_file, img_dir, transform=None):
        
        with open(annotations_file) as f:
            self.annotations = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform
    
    
    def __len__(self):
        
        return len(self.annotations['images'])
    
    
    def __getitem__(self, idx):
        
        img_info = self.annotations['images'][idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        img = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = []
        for ann in self.annotations['annotations']:
            if ann['image_id'] == img_info['id']:
                boxes.append(ann['bbox'])            # [x, y, width, height]
                labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if self.transform:
            transformed = self.transform(image=np.array(img), bboxes=boxes, labels=labels)
            image_np = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        else:
            image_np = transforms.ToTensor()(img)
        
        target = {}
        image_id = int( img_info['id'] )
        target["boxes"] = boxes
        target["labels"] = labels
        target['image_id'] = torch.tensor([image_id], dtype=torch.int64)
        
        return image_np, target


def get_transform(train, **args):
    
    transforms_list = []
    if train:
        resize = args.get('resize', False)
        width = height = args.get('imgsz', None)
        horizontalflip = args.get('horizontalflip', False)
        colorjitter = args.get('colorjitter', False)
        randombrightnesscontrast = args.get('randombrightnesscontrast', False)
        tofloat = args.get('tofloat', False)
        normalize = args.get('normalize', False)
        min_visibility_for_transform = args.get('min_visibility_for_transform', 0.3)
        
        if resize or width is not None and height is not None:
            if width is None or height is None :
                raise ValueError('Please Set size to some value in config when resize = True !')
            transforms_list.append ( A.Resize(width=width, height=height) )
        else:
            print('Set imgsz to defualt : 640*640')
            transforms_list.append ( A.Resize(width=640, height=640) )
        
        if horizontalflip :
            transforms_list.append(  A.HorizontalFlip(p=0.5) )
        
        if colorjitter :
            transforms_list.append(  A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3) )
        
        if randombrightnesscontrast :
            transforms_list.append(  A.RandomBrightnessContrast(p=0.3) )
        
        if tofloat :
            transforms_list.append(  A.ToFloat(max_value=255.0) )
        
        if normalize :
            transforms_list.append(  A.Normalize(mean=(0.485, 0.456, 0.406),  std=(0.229, 0.224, 0.225) ) )
        
    transforms_list.append( ToTensorV2() )
    
    return A.Compose( transforms_list, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=min_visibility_for_transform))


def collate_fn(batch):
    return tuple(zip(*batch))
#def collate_fn(batch):
#  batch = [data for data in batch if data is not None and data[0] is not None]
#  return tuple(zip(*batch))

def convert_bbox_format(boxes, image_size):
    
    converted_boxes = []
    img_width, img_height = image_size
    for box in boxes:
        
        x_center, y_center, width, height = box
        xmin = (x_center - width / 2) * img_width
        ymin = (y_center - height / 2) * img_height
        xmax = (x_center + width / 2) * img_width
        ymax = (y_center + height / 2) * img_height
        
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_width, xmax)
        ymax = min(img_height, ymax)
        converted_boxes.append([xmin, ymin, xmax, ymax])
    
    return converted_boxes


def get_data_loaders(data_config, batch_size, num_workers, **args):
    
    with open(data_config) as f:
        data = yaml.safe_load(f)
    
    train_annotations = os.path.join( data['train'], 'annotations.json')
    val_annotations = os.path.join( data['val'], 'annotations.json')
    
    dataset_train = CustomDataset(train_annotations, data['train'], transform=get_transform(train=True, args = args))
    dataset_val = CustomDataset(val_annotations, data['val'], transform=get_transform(train=False))
    
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, collate_fn=collate_fn)
    
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, collate_fn=collate_fn)
    
    return data_loader_train, data_loader_val

#cloner174