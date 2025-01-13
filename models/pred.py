# in the name of God
#
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F


def infer_and_draw(image_path, model, device, map_of_classes, threshold=0.8):
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).to(device)
    
    with torch.no_grad():
        prediction = model([img_tensor])
    
    draw = ImageDraw.Draw(img)
    
    found_boxes = False
    for element in range(len(prediction[0]['boxes'])):
        
        score = prediction[0]['scores'][element]
        if score > threshold:
            
            found_boxes = True
            
            box = prediction[0]['boxes'][element].cpu().numpy()
            label = prediction[0]['labels'][element].cpu().numpy()
            label_name = map_of_classes[int(label)]
            
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=4)
            
            text_x = box[0]
            text_y = box[1] - 20
            
            text_bg_color = "RED"
            draw.rectangle([(text_x, text_y), (text_x + 80, text_y + 15)], fill=text_bg_color)
            draw.text((text_x, text_y), f"{label_name}: {score:.2f}", fill="WHITE")
    
    if found_boxes:
        print("Bounding boxes found and drawn.")
    else:
        print("No bounding boxes found above the threshold.")
    
    return img


#cloner174