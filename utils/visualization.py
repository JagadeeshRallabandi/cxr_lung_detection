import cv2
import numpy as np


def draw_boxes(
    image,
    boxes,
    labels=None,
    scores=None,
    label_map=None,
    color=(0, 255, 0)
):
    """
    Draw bounding boxes on image

    image: RGB image (H,W,3)
    boxes: Nx4 (x1,y1,x2,y2)
    labels: optional class ids
    scores: optional confidence scores
    label_map: dict {id: class_name}
    """

    img = image.copy()

    if isinstance(boxes, np.ndarray) is False:
        boxes = boxes.cpu().numpy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = ""
        if labels is not None:
            cls_id = int(labels[i])
            if label_map and cls_id in label_map:
                text += label_map[cls_id]
            else:
                text += str(cls_id)

        if scores is not None:
            text += f" {scores[i]:.2f}"

        if text:
            cv2.putText(
                img,
                text,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

    return img
