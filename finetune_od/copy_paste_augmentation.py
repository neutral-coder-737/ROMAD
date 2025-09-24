import argparse
import cv2
import random
import os
import glob
from tqdm import tqdm


def copy_paste_augment(
    image,
    bboxes,
    rare_classes,
    max_paste=3,
    iou_thresh=0.01,
):
    H, W = image.shape[:2]
    new_bboxes = bboxes.copy()
    new_image = image.copy()

    def yolo_to_xyxy(box):
        cls, xc, yc, w, h = box
        x1 = int((xc - w / 2) * W)
        y1 = int((yc - h / 2) * H)
        x2 = int((xc + w / 2) * W)
        y2 = int((yc + h / 2) * H)
        return cls, x1, y1, x2, y2

    def xyxy_to_yolo(cls, x1, y1, x2, y2):
        xc = ((x1 + x2) / 2) / W
        yc = ((y1 + y2) / 2) / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        return [cls, xc, yc, w, h]

    def iou(box1, box2):
        # boxes in xyxy
        _, x1, y1, x2, y2 = box1
        _, xx1, yy1, xx2, yy2 = box2
        xi1, yi1 = max(x1, xx1), max(y1, yy1)
        xi2, yi2 = min(x2, xx2), min(y2, yy2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (xx2 - xx1) * (yy2 - yy1)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def rotate_crop(crop, angle):
        """Rotate a crop and return rotated crop + its enclosing box size."""
        h, w = crop.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = abs(rot_mat[0, 0]), abs(rot_mat[0, 1])
        # new bounding dimensions
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        # adjust rotation matrix
        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(
            crop, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
        )
        return rotated

    # Collect candidate rare boxes
    rare_boxes = [yolo_to_xyxy(b) for b in bboxes if int(b[0]) in rare_classes]
    if not rare_boxes:
        return new_image, new_bboxes

    for _ in range(max_paste):
        cls, x1, y1, x2, y2 = random.choice(rare_boxes)
        crop = image[y1:y2, x1:x2]
        h, w = crop.shape[:2]

        if h < 2 or w < 2:
            continue

        # random rotation
        angle = random.choice([0, 90, 180, 270])
        crop = rotate_crop(crop, angle)
        h, w = crop.shape[:2]

        for attempt in range(20):
            nx1 = random.randint(0, W - w)
            ny1 = random.randint(0, H - h)
            nx2, ny2 = nx1 + w, ny1 + h

            new_box = [cls, nx1, ny1, nx2, ny2]

            # Check IoU with all existing boxes
            if all(iou(new_box, yolo_to_xyxy(b)) < iou_thresh for b in new_bboxes):
                new_image[ny1:ny2, nx1:nx2] = crop
                new_bboxes.append(xyxy_to_yolo(cls, nx1, ny1, nx2, ny2))
                break

    return new_image, new_bboxes


def apply_copy_paste_aug(dataset_path):
    input_img_paths = glob.glob(os.path.join(f"{dataset_path}", "**/*.JPG"))
    if len(input_img_paths) == 0:
        print("No matched input images!")

    for input_img_path in tqdm(
        input_img_paths, desc="Applying copy-paste augmentation"
    ):
        if "_aug" in os.path.basename(input_img_path):
            continue
        for i, max_paste in enumerate([2, 3]):
            input_bbox_path = input_img_path.replace("images", "labels").replace(
                ".JPG", ".txt"
            )
            img = cv2.imread(input_img_path)
            with open(input_bbox_path) as f:
                bboxes = [list(map(float, line.strip().split())) for line in f]

            # twin_bracelets' rare classes
            rare_classes = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            aug_img, aug_bboxes = copy_paste_augment(
                img, bboxes, rare_classes, max_paste=max_paste
            )

            # save new image + bboxes
            output_img_path = input_img_path.replace(".JPG", f"_aug{i}.JPG")
            output_bbox_path = input_bbox_path.replace(".txt", f"_aug{i}.txt")

            os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
            os.makedirs(os.path.dirname(output_bbox_path), exist_ok=True)
            cv2.imwrite(output_img_path, aug_img)
            with open(output_bbox_path, "w") as f:
                for bbox in aug_bboxes:
                    line = bbox
                    line[0] = int(line[0])
                    f.write(" ".join(map(str, line)) + "\n")


def get_args():
    parser = argparse.ArgumentParser(description="Copy-paste augmentation")
    parser.add_argument(
        "--dataset_path",
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    apply_copy_paste_aug(args.dataset_path)
