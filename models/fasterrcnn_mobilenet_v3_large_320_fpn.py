#in the name of God
#
import torchvision
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


def get_model(pretrained=True, num_classes=91):
    
    if pretrained:
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    else:
        weights = None
    
    backbone = mobilenet_v3_large(weights=weights).features
    backbone.out_channels = 960
    
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    
    return model


#cloner174