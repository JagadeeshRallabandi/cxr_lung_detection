import albumentations as A
import cv2


def get_train_transforms():
    """
    Safe medical augmentations for Chest X-rays
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=1024),
            A.PadIfNeeded(
                min_height=1024,
                min_width=1024,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.05,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.4
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_area=0.0,
            min_visibility=0.0
        )
    )


def get_val_transforms():
    """
    Validation / inference transforms (NO augmentation)
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=1024),
            A.PadIfNeeded(
                min_height=1024,
                min_width=1024,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_area=0.0,
            min_visibility=0.0
        )
    )
