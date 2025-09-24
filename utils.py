import os
import sys
from PIL import Image
import cv2
import torch
from tqdm import tqdm
import logging
from torchvision.ops import nms
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml
from types import SimpleNamespace
from contextlib import contextmanager

sys.path.append("./ultralytics")
from ultralytics.utils import LOGGER


device = "cuda:0"


@contextmanager
def silence_ultralytics(level=logging.ERROR):
    """Temporarily change Ultralytics LOGGER level inside the block."""
    old_level = LOGGER.level
    try:
        LOGGER.setLevel(level)
        yield
    finally:
        LOGGER.setLevel(old_level)


def get_dataset_args(dataset_cls):
    with open("dataset_args.yaml", "r") as f:
        dataset_configs = yaml.safe_load(f)
        
    if dataset_cls not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_cls}")
    return SimpleNamespace(**dataset_configs[dataset_cls])


def get_area_from_sam(
    sam_model,
    img,
    valid_obj_boxes,
    seg_imgsz,
):
    valid_obj_boxes_cp = valid_obj_boxes.detach().clone()
    if not len(valid_obj_boxes_cp):
        return None, None

    results = sam_model(img, bboxes=valid_obj_boxes_cp, imgsz=seg_imgsz)
    masks_preview = results[0].plot(show=False)
    masks = results[0].masks.data.cpu()

    return masks, masks_preview


def dump_visualizations(
    img,
    masks,
    valid_obj_classes,
    valid_obj_boxes,
    filename,
    id2label,
):
    def reorder_by_draw_order(
        valid_obj_boxes,
        masks,
        valid_obj_classes,
        draw_order,
    ):
        priority = {cls: i for i, cls in enumerate(draw_order)}
        combined = list(zip(valid_obj_boxes, masks, valid_obj_classes))
        combined_sorted = sorted(
            combined, key=lambda x: priority.get(int(x[2]), len(draw_order))
        )
        new_boxes, new_masks, new_classes = zip(*combined_sorted)
        return list(new_boxes), list(new_masks), list(new_classes)

    if "breakfast_box" in filename:
        color_palette = [
            (240, 214, 16),  # yellow
            (241, 243, 246),  # white
            (252, 34, 129),  # pink
            (0, 191, 45),  # green
            (100, 28, 255),  # purple
        ]
        draw_order = [0, 1, 2, 3, 4]
    elif "screw_bag" in filename:
        color_palette = [
            (210, 40, 4),  # brown
            (241, 243, 246),  # white
            (240, 150, 40),  # yellow
            (23, 190, 207),  # cyan
        ]
        draw_order = [3, 1, 2, 0]
    elif "pushpins" in filename:
        color_palette = [
            (23, 190, 207),  # cyan
            (210, 40, 4),  # brown
        ]
        draw_order = [0, 1]
    elif "juice_bottle" in filename:
        color_palette = [
            (252, 34, 129),  # pink
            (240, 150, 40),  # yellow
            (241, 243, 246),  # white
            (100, 28, 255),  # purple
            (250, 40, 4),  # red
            (23, 190, 207),  # cyan
            (0, 191, 45),  # green
            (160, 145, 140),  # gray
        ]
        draw_order = [0, 4, 6, 1, 3, 2, 5, 7]
    elif "splicing_connectors" in filename:
        color_palette = [
            (252, 34, 129),  # pink
            (0, 0, 255),  # pure blue
            (240, 150, 40),  # yellow
            (250, 40, 4),  # red
        ]
        draw_order = [0, 1]
    elif "twin_bracelets" in filename:
        color_palette = [
            (252, 34, 129),  # pink
            (240, 150, 40),  # yellow
            (255, 255, 0),  # pure yellow
            (241, 243, 246),  # white
            (100, 28, 255),  # purple
            (160, 145, 140),  # gray
            (23, 190, 207),  # cyan
            (181, 78, 1),  # brown
            (0, 255, 0),  # pure green
            (117, 142, 12),  # mountain green
            (255, 0, 0),  # pure red
            (255, 0, 255),  # magenta
            (0, 0, 255),  # pure blue
        ]
        draw_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    valid_obj_boxes_cp = valid_obj_boxes.detach().clone()
    masks_cp = masks.detach().clone()
    valid_obj_classes_cp = valid_obj_classes.detach().clone()

    valid_obj_boxes_cp, masks_cp, valid_obj_classes_cp = reorder_by_draw_order(
        valid_obj_boxes_cp, masks_cp, valid_obj_classes_cp, draw_order
    )
    if isinstance(img, Image.Image):
        img_overlay = np.array(img.convert("RGB"))
    else:
        img_overlay = img.copy()

    # Visualize object segmentation
    seg_img = np.zeros_like(img, dtype=np.uint8)
    # seg_img = img_overlay.copy()
    for mask, cls in zip(masks_cp, valid_obj_classes_cp):
        mask = mask.cpu().numpy().astype(bool)
        color = color_palette[int(cls) % len(color_palette)]
        seg_img[mask] = (1 * np.array(color) + 0 * seg_img[mask]).astype(np.uint8)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.imsave(filename.replace(".jpg", "_seg.jpg"), seg_img)

    # Visualize object detection
    for cls, box in zip(valid_obj_classes_cp, valid_obj_boxes_cp):
        color = color_palette[int(cls) % len(color_palette)]
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img_overlay, (x1, y1), (x2, y2), color, thickness=4)
        cv2.putText(
            img_overlay,
            id2label[int(cls)],
            (x1, y1 - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=color,
            thickness=2,
        )

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.imsave(filename.replace(".jpg", "_od.jpg"), img_overlay)


# Distance-based attention
def add_dba_to_nodes(representative_embs, valid_obj_boxes, context_count=3):
    k, _, d = representative_embs.shape
    assert d == 256

    # No object detected in the given image --> context = []
    if k == 0:
        context = torch.zeros([0, 1, d], device=device)
    # One object detected in the given image --> context = [0, ..., 0]
    elif k == 1:
        context = torch.zeros([1, 1, d], device=device)
    else:
        x_centers = (valid_obj_boxes[:, 0] + valid_obj_boxes[:, 2]) / 2  # (k,)
        y_centers = (valid_obj_boxes[:, 1] + valid_obj_boxes[:, 3]) / 2  # (k,)
        centers = torch.stack([x_centers, y_centers], dim=1)  # (k, 2)

        # Compute pairwise Euclidean distances
        centers1 = centers.unsqueeze(1).expand(k, k, 2)  # (k, k, 2)
        centers2 = centers.unsqueeze(0).expand(k, k, 2)  # (k, k, 2)
        distances = torch.norm(centers1 - centers2, dim=2, keepdim=True)  # (k, k, 1)

        # remove self loop
        distances = distances.masked_fill(
            torch.eye(k, device=distances.device).bool().unsqueeze(-1), float("inf")
        )  # (k, k, 1)

        # Normalize across rows
        inv_distances = 1.0 / (1.0 + distances)
        weights = inv_distances / inv_distances.sum(dim=1, keepdim=True)  # (k, k, 1)

        if context_count is not None:
            top_k = min(context_count, k)
            weights_2d = weights.squeeze(-1)  # (k, k)
            _, topk_indices = torch.topk(weights_2d, k=top_k, dim=1)
            mask = torch.zeros_like(weights_2d)  # (k, k)
            mask.scatter_(1, topk_indices, 1.0)
            top_weights = (weights_2d * mask).unsqueeze(-1)  # (k, k, 1)

            # Re-normalize across rows
            weights = top_weights / top_weights.sum(dim=1, keepdim=True)  # (k, k, 1)

        context = torch.sum(
            weights * representative_embs.transpose(0, 1), dim=1, keepdim=True
        )  # (k, 1, d)

    return torch.cat([representative_embs, context], dim=2)  # (k, 1, 2d)


# bsz=1
@torch.no_grad()
def extract_representative_embs(
    rtdetr_model,
    data_dirs,
    nms_threshold,
    obj_threshold=0.3,
    vis_path="./visualizations",
    include_area=False,
    sam_model=None,
    include_dba=False,
    context_count=3,
    visualize: bool = True,
    od_imgsz=640,
    seg_imgsz=1024,
):
    representative_embs_list = []
    valid_obj_indices_list = []
    valid_obj_classes_list = []
    img_count = 0
    for data_dir in sorted(data_dirs):
        img_count += len(
            [
                img_name
                for img_name in os.listdir(data_dir)
                if img_name.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
        )
    pbar = tqdm(total=img_count, desc="Processing images")
    for data_dir in sorted(data_dirs):
        for img_name in sorted(os.listdir(data_dir)):
            if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(data_dir, img_name)
                results = rtdetr_model(image_path, conf=obj_threshold, imgsz=od_imgsz)

                valid_obj_boxes = results[0].boxes.xyxy
                valid_obj_classes = results[0].boxes.cls
                valid_obj_scores = results[0].boxes.conf
                valid_obj_indices = results[0].extra["selected_indices"]
                raw_img = Image.open(results[0].path)
                id2label = results[0].names
                object_queries = results[0].extra["dec_embs"]

                if nms_threshold is not None:
                    nms_keep_indices = nms(
                        valid_obj_boxes, valid_obj_scores, nms_threshold
                    )
                    valid_obj_boxes = valid_obj_boxes[nms_keep_indices]
                    valid_obj_classes = valid_obj_classes[nms_keep_indices]
                    valid_obj_indices = valid_obj_indices[nms_keep_indices]
                    valid_obj_scores = valid_obj_scores[nms_keep_indices]
                    object_queries = object_queries[nms_keep_indices]

                valid_obj_indices_list.append(valid_obj_indices)
                valid_obj_classes_list.append(valid_obj_classes)

                if include_area:
                    os.makedirs(vis_path, exist_ok=True)
                    masks, masks_preview = get_area_from_sam(
                        sam_model=sam_model,
                        img=raw_img,
                        valid_obj_boxes=valid_obj_boxes,
                        seg_imgsz=seg_imgsz,
                    )

                    areas = (
                        masks.sum(dim=(1, 2))
                        if masks is not None
                        else torch.zeros([0], device=device)
                    )

                    ############################################
                    # Visualize object detection, segmentation #
                    ############################################
                    if visualize:
                        dataset_name = Path(data_dir).parts[1]  # e.g. screw_bag
                        img_path_parts = Path(results[0].path).parts
                        img_vis_path = (
                            str(
                                Path(
                                    *img_path_parts[
                                        img_path_parts.index(dataset_name) :
                                    ]
                                ).with_suffix("")
                            )
                            + ".jpg"
                        )
                        vis_filename = os.path.join(vis_path, img_vis_path)
                        os.makedirs(os.path.dirname(vis_filename), exist_ok=True)

                        if masks is not None:
                            plt.imsave(
                                vis_filename.replace(".jpg", "_ult.jpg"), masks_preview
                            )
                            binary_masks_list = []
                            for i, mask in enumerate(masks):
                                binary_mask = mask.numpy().astype("uint8") * 255
                                binary_masks_list.append(binary_mask)
                                plt.imsave(
                                    vis_filename.replace(".jpg", f"_part_{i}.jpg"),
                                    binary_mask,
                                    cmap="gray",
                                )

                            dump_visualizations(
                                img=raw_img,
                                masks=masks,
                                valid_obj_classes=valid_obj_classes,
                                valid_obj_boxes=valid_obj_boxes,
                                filename=vis_filename,
                                id2label=id2label,
                            )

                ############################################
                # Feature Enrichment                       #
                ############################################
                representative_embs = object_queries.unsqueeze(1)  # (k, 1, d)

                # (k, 1, d*=2)
                if include_dba:
                    representative_embs = add_dba_to_nodes(
                        representative_embs,
                        valid_obj_boxes,
                        context_count=context_count,
                    )

                # (k, 1, d+=1)
                if include_area:
                    areas = (
                        areas.unsqueeze(1)
                        .unsqueeze(2)
                        .expand(-1, representative_embs.shape[1], 1)
                        .to(device)
                    )  # (k, levels, 1)
                    representative_embs = torch.cat((representative_embs, areas), dim=2)

                representative_embs_list.append(
                    {
                        "embeddings": representative_embs,
                        "object_classes": valid_obj_classes,
                        "image_path": results[0].path,
                    }
                )

            pbar.update(1)

    pbar.close()
    return representative_embs_list, valid_obj_indices_list, valid_obj_classes_list
