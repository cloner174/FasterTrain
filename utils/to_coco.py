# in the name of God
#
import os
import pandas as pd
import json
from PIL import Image


map_of_classes = {
  0: "Car",
  1: "Different-Traffic-Sign",
  2: "Green-Traffic-Light",
  3: "Motorcycle",
  4: "Pedestrian",
  5: "Pedestrian-Crossing",
  6: "Prohibition-Sign",
  7: "Red-Traffic-Light",
  8: "Speed-Limit-Sign",
  9: "Truck",
  10: "Warning-Sign"
}

def get_df(label_dir, image_dir, dir_of='train'):
    
    df = []
    for filename in os.listdir(label_dir):
        
        if filename.endswith('.txt'):
            
            filepath = os.path.join(label_dir, filename)
            image_path = os.path.join(image_dir, filename.replace('.txt', '.jpg'))
            
            if os.path.isfile(image_path):
                image_name = filename.replace('.txt', '.jpg')
            else:
                image_path = os.path.join(image_dir, filename.replace('.txt', '.png'))
                if os.path.isfile(image_path):
                    image_name = filename.replace('.txt', '.png')
                else:
                    raise Exception(f'No Image Found for : {filename}')
            
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size
            
            with open(filepath, 'r') as file:
                
                for line in file:
                    
                    parts = line.strip().split()
                    obj_class = parts[0]
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    box = [x_center, y_center, width, height]
                    
                    converted_box = convert_bbox_format([box], (image_width, image_height))[0]
                    
                    df.append({
                        'image_name': image_name,
                        'image_path': image_path,
                        'xmin': converted_box[0],
                        'ymin': converted_box[1],
                        'xmax': converted_box[2],
                        'ymax': converted_box[3],
                        'class': map_of_classes[int(obj_class)],
                        'width': image_width,
                        'height': image_height
                    })
    
    df = pd.DataFrame(df)
    
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    classes = sorted(df['class'].unique())
    category_mapping = {cls: idx + 1 for idx, cls in enumerate(classes)}  # COCO category IDs start at 1
    categories = [{"id": idx, "name": cls} for cls, idx in category_mapping.items()]
    coco["categories"] = categories
    
    image_id = 1
    annotation_id = 1
    images_dict = {}
    for _, row in df.iterrows():
        img_name = row['image_name']
        if img_name not in images_dict:
            image_info = {
                "id": image_id,
                "file_name": img_name,
                "width": int(row['width']),
                "height": int(row['height'])
            }
            coco["images"].append(image_info)
            images_dict[img_name] = image_id
            image_id += 1
        
        bbox_width = row['xmax'] - row['xmin']
        bbox_height = row['ymax'] - row['ymin']
        area = bbox_width * bbox_height
        
        annotation = {
            "id": annotation_id,
            "image_id": images_dict[img_name],
            "category_id": category_mapping[row['class']],
            "bbox": [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
            "area": area,
            "iscrowd": 0
        }
        coco["annotations"].append(annotation)
        annotation_id += 1
    
    with open(f'coco_{dir_of}.json', 'w') as f:
        json.dump(coco, f, indent=4)
    
    df.to_csv(f'df_{dir_of}.csv', index=False)
    
    return df



def get_df_coco(label_dir, image_dir, dir_of='train'):
    
    df = []
    for filename in os.listdir(label_dir):
        
        if filename.endswith('.txt'):
            
            filepath = os.path.join(label_dir, filename)
            image_path = os.path.join(image_dir, filename.replace('.txt', '.jpg'))
            
            if os.path.isfile(image_path):
                image_name = filename.replace('.txt', '.jpg')
            else:
                image_path = os.path.join(image_dir, filename.replace('.txt', '.png'))
                if os.path.isfile(image_path):
                    image_name = filename.replace('.txt', '.png')
                else:
                    raise Exception(f'No Image Found for : {filename}')
            
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size
            
            with open(filepath, 'r') as file:
                
                for line in file:
                    
                    parts = line.strip().split()
                    obj_class = parts[0]
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    df.append({
                        'image_name': image_name,
                        'image_path': image_path,
                        'xmin': x_center,
                        'ymin': y_center,
                        'xmax': width,
                        'ymax': height,
                        'class': map_of_classes[int(obj_class)],
                        'width': image_width,
                        'height': image_height
                    })
    
    df = pd.DataFrame(df)
    
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    classes = sorted(df['class'].unique())
    category_mapping = {cls: idx + 1 for idx, cls in enumerate(classes)}  # COCO category IDs start at 1
    categories = [{"id": idx, "name": cls} for cls, idx in category_mapping.items()]
    coco["categories"] = categories
    
    image_id = 1
    annotation_id = 1
    images_dict = {}
    for _, row in df.iterrows():
        img_name = row['image_name']
        if img_name not in images_dict:
            image_info = {
                "id": image_id,
                "file_name": img_name,
                "width": int(row['width']),
                "height": int(row['height'])
            }
            coco["images"].append(image_info)
            images_dict[img_name] = image_id
            image_id += 1
        
        #bbox_width = row['xmax'] - row['xmin']
        #bbox_height = row['ymax'] - row['ymin']
        area = row['xmax'] *  row['ymax']  # bbox_width * bbox_height
        
        annotation = {
            "id": annotation_id,
            "image_id": images_dict[img_name],
            "category_id": category_mapping[row['class']],
            "bbox": [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
            "area": area,
            "iscrowd": 0
        }
        coco["annotations"].append(annotation)
        annotation_id += 1
    
    with open(f'coco_{dir_of}.json', 'w') as f:
        json.dump(coco, f, indent=4)
    
    df.to_csv(f'df_{dir_of}.csv', index=False)
    
    return df



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

#cloner174