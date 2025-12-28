import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def load_model(num_classes, weight_path, device):
    """
    Loads Faster R-CNN model with trained weights
    """
    model = fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    return model
