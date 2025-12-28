import json
import os
import cv2
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class VinDrCocoDataset(Dataset):
    """
    PyTorch Dataset for COCO-style VinDr-CXR annotations
    Compatible with Faster R-CNN
    """

    def __init__(self, coco_json, image_root, transforms=None):
        self.image_root = image_root
        self.transforms = transforms

        with open(coco_json, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.categories = coco["categories"]

        # image_id -> image info
        self.image_id_to_info = {img["id"]: img for img in self.images}

        # image_id -> annotations
        self.image_id_to_anns = defaultdict(list)
        for ann in self.annotations:
            self.image_id_to_anns[ann["image_id"]].append(ann)

        self.ids = list(self.image_id_to_info.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_info = self.image_id_to_info[image_id]

        img_path = os.path.join(self.image_root, img_info["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = self.image_id_to_anns[image_id]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]
            areas = [1.0]
            iscrowd = [0]

        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        areas = torch.tensor(areas, dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
