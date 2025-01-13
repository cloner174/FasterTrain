# in the name of God
#
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as F


def get_albumentations_transform(train=True):
    
    if train:
        transform = A.Compose([
            A.Resize(height=640, width=640),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(
                mean=(0.4539, 0.4772, 0.4796),
                std=(0.2488, 0.2556, 0.2704),
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',  # boxes are [x_min, y_min, x_max, y_max]
            label_fields=['labels'],
        )
        )
    
    else:
        transform = A.Compose([
            A.Resize(height=640, width=640),
            A.Normalize(
                mean=(0.4539, 0.4772, 0.4796),
                std=(0.2488, 0.2556, 0.2704),
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        )
        )
    
    return transform



class ResizeAndNormalize:
    
    def __init__(self, new_width, new_height,
                 mean=(0.4539, 0.4772, 0.4796),
                 std=(0.2488, 0.2556, 0.2704)):
        
        self.new_width = new_width
        self.new_height = new_height
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target):
        """
        image: PIL Image
        target: dict with 'boxes' (Nx4 Tensor [x_min, y_min, x_max, y_max])
        Returns: transformed image, updated target
        """
        # --- 1) Resize the image ---
        old_width, old_height = image.size  # PIL Image: size = (width, height)
        image = F.resize(image, (self.new_height, self.new_width))
        # compute scale factors
        scale_x = self.new_width  / float(old_width)
        scale_y = self.new_height / float(old_height)
        # --- 2) Resize bounding boxes ---
        boxes = target['boxes']
        # boxes is shape (N,4): [x_min, y_min, x_max, y_max]
        boxes[:, 0] = boxes[:, 0] * scale_x
        boxes[:, 2] = boxes[:, 2] * scale_x
        boxes[:, 1] = boxes[:, 1] * scale_y
        boxes[:, 3] = boxes[:, 3] * scale_y
        target['boxes'] = boxes
        image = F.to_tensor(image)  # => shape [C,H,W], floats in [0,1]
        # F.normalize(image, mean, std) -> uses channel-wise
        image = F.normalize(image, self.mean, self.std)

        return image, target



def get_custom_transform(train=True):
    transforms = []
    transforms.append(ResizeAndNormalize(640, 640))
    return Compose(transforms)


class Compose:
    """Simple class to apply multiple transforms in sequence."""
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    

#cloner174