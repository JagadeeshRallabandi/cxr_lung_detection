import torch

def box_iou(box1, box2):
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-6)


def evaluate_map_50(model, dataloader, device, score_thresh=0.5):
    model.eval()
    TP = FP = FN = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for pred, tgt in zip(outputs, targets):
                gt_boxes = tgt["boxes"].to(device)
                gt_labels = tgt["labels"].to(device)

                keep = pred["scores"] >= score_thresh
                pb = pred["boxes"][keep]
                pl = pred["labels"][keep]

                if len(pb) == 0:
                    FN += len(gt_boxes)
                    continue

                ious = box_iou(pb, gt_boxes)
                matched = set()

                for i in range(len(pb)):
                    max_iou, idx = ious[i].max(0)
                    if max_iou >= 0.5 and pl[i] == gt_labels[idx] and idx.item() not in matched:
                        TP += 1
                        matched.add(idx.item())
                    else:
                        FP += 1

                FN += len(gt_boxes) - len(matched)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1
