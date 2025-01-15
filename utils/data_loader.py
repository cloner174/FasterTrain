#
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
from pycocotools.coco import COCO
#from utils.utils import csv_to_coco


class TraficDataset(Dataset):

    def __init__(self, annotation_file, root_dir, transform=None, pass_img_and_target_to_transform = False):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.pass_img_and_target_to_transform = pass_img_and_target_to_transform
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        
        img = Image.open(os.path.join(self.root_dir , path)).convert('RGB')
        
        num_objs = len(coco_annotation)
        boxes = []
        labels = []
        for i in range(num_objs):
                xmin = coco_annotation[i]['bbox'][0]
                ymin = coco_annotation[i]['bbox'][1]
                xmax = coco_annotation[i]['bbox'][2] #+ xmin
                ymax = coco_annotation[i]['bbox'][3] #+ ymin
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(coco_annotation[i]['category_id'])
        """
        if self.transform:
          img_np = np.array(img)
          bboxes = boxes
          labels_list = labels

          transformed = self.transform(
                image=img_np,
                bboxes=bboxes,
                labels=labels_list
          )
          img = transformed['image']
          new_bboxes = transformed['bboxes']
          new_labels = transformed['labels']

          new_bboxes = torch.as_tensor(new_bboxes, dtype=torch.float32)
          new_labels = torch.as_tensor(new_labels, dtype=torch.int64)

          boxes = new_bboxes
          labels = new_labels
        else:
          raise NotImplementedError('Sl')
        """
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([img_id])}
        
        if self.transform and self.pass_img_and_target_to_transform:
            img, target = self.transform(img, target)
        
        elif self.transform:
            img = self.transform(img)
        
        return img, target




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
        img_path = os.path.join(self.img_dir, 'images',img_info['file_name'])
        
        img = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = []
        for ann in self.annotations['annotations']:
            if ann['image_id'] == img_info['id']:
                boxes.append(ann['bbox'])            # [x, y, width, height]
                labels.append(ann['category_id'])
        

        
        if self.transform:
            transformed = self.transform(image=np.array(img), bboxes=boxes, labels=labels)
            image_np = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        else:
            image_np = transforms.ToTensor()(img)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        image_id = int( img_info['id'] )
        target["boxes"] = boxes
        target["labels"] = labels
        target['image_id'] = torch.tensor([image_id], dtype=torch.int64)
        
        return image_np, target


def get_transform(train, img_size = 320, 
                  min_visibility_for_transform = 0.3 ,
                  tofloat = True, normalize=True,
                  horizontalflip=True , colorjitter  = True, randombrightnesscontrast = True ):
    
    transforms_list = []
    transforms_list.append ( A.Resize(width=img_size, height=img_size) )
    
    if train:
        
        if img_size is not None and isinstance(img_size, int):
            transforms_list.append ( A.Resize(width=img_size, height=img_size) )
        else:
            print('Set imgsz to defualt : 320*320')
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


#def collate_fn(batch):
#    return tuple(zip(*batch))

def collate_fn(batch):
  batch = [data for data in batch if data is not None and data[0] is not None]
  return tuple(zip(*batch))


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
