import torch
import cv2
import numpy as np

def predict(model, image, device, conf=0.5):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img / 255., dtype=torch.float32).permute(2, 0, 1)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)[0]

    keep = output["scores"] >= conf
    boxes = output["boxes"][keep].cpu().numpy()
    labels = output["labels"][keep].cpu().numpy()
    scores = output["scores"][keep].cpu().numpy()

    return boxes, labels, scores
