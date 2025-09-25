import json
from pathlib import Path
from matplotlib import pyplot as plt
import os
import shutil
import glob
from collections import Counter
from torchvision.ops import nms
import math
import cv2
import logging
from contextlib import contextmanager
from tqdm import tqdm
import random
import sys
import warnings

sys.path.append("../ultralytics")
from ultralytics.models.rtdetr.model import RTDETR
from ultralytics.utils import LOGGER
from copy_paste_augmentation import apply_copy_paste_aug

warnings.filterwarnings("ignore")


@contextmanager
def silence_ultralytics(level=logging.ERROR):
    """Temporarily change Ultralytics LOGGER level inside the block."""
    old_level = LOGGER.level
    try:
        LOGGER.setLevel(level)
        yield
    finally:
        LOGGER.setLevel(old_level)


def dump_yolo_labels(
    pseudo_annotations_path,
    data_list,
    max_pseudo_samples,
    data_yaml_src_path,
    data_yaml_dst_path,
    img_ext="JPG",
):
    if "twin_bracelets" in data_list[0]["path"]:
        twin_a_data = [d for d in data_list if "twin_a/" in d["path"]]
        twin_b_data = [d for d in data_list if "twin_b/" in d["path"]]
        random.shuffle(twin_a_data)
        random.shuffle(twin_b_data)
        data_lists = [
            twin_a_data[: max_pseudo_samples // 2],
            twin_b_data[: max_pseudo_samples // 2],
        ]
    # Loco Datasets
    else:
        random.shuffle(data_list)
        data_lists = [data_list[:max_pseudo_samples]]

    for data_list_per_type in data_lists:
        for data in data_list_per_type:
            img_path = data["path"]
            classes = data["obj_classes"].tolist()
            boxes = data["obj_boxes"].tolist()
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            if "twin_a/" in img_path:
                label_path = os.path.join(
                    pseudo_annotations_path,
                    f"labels/train/twin_a_{base_name}_pseudo.txt",
                )
            elif "twin_b/" in img_path:
                label_path = os.path.join(
                    pseudo_annotations_path,
                    f"labels/train/twin_b_{base_name}_pseudo.txt",
                )
            # Loco Datasets
            else:
                label_path = os.path.join(
                    pseudo_annotations_path, f"labels/train/{base_name}_pseudo.txt"
                )

            cp_img_path = label_path.replace("labels", "images").replace(
                ".txt", f".{img_ext}"
            )

            lines = []
            for cls, box in zip(classes, boxes):
                x1, y1, x2, y2 = map(float, box)
                # convert to YOLO format (normalized cx, cy, w, h)
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

            os.makedirs(
                os.path.dirname(label_path.replace("train", "val")), exist_ok=True
            )
            with open(label_path.replace("train", "val"), "w") as f:
                f.write("\n".join(lines))

            os.makedirs(os.path.dirname(cp_img_path), exist_ok=True)
            os.makedirs(
                os.path.dirname(cp_img_path.replace("train", "val")), exist_ok=True
            )
            shutil.copy(img_path, cp_img_path)
            shutil.copy(img_path, cp_img_path.replace("train", "val"))

    os.makedirs(os.path.dirname(data_yaml_dst_path), exist_ok=True)
    shutil.copy(data_yaml_src_path, data_yaml_dst_path)


def add_manual_annotations_to_trainset(man_annotations_path, new_trainset_path):
    for label_path in glob.glob(
        os.path.join(man_annotations_path, "labels/train/*.txt")
    ):
        base_name = os.path.splitext(os.path.basename(label_path))[0]

        shutil.copy(
            label_path,
            os.path.join(new_trainset_path, f"labels/train/{base_name}_man.txt"),
        )
        shutil.copy(
            label_path,
            os.path.join(new_trainset_path, f"labels/val/{base_name}_man.txt"),
        )
        shutil.copy(
            label_path.replace("labels", "images").replace(
                ".txt", f".{config['img_ext']}"
            ),
            os.path.join(
                new_trainset_path,
                f"images/train/{base_name}_man.{config['img_ext']}",
            ),
        )
        shutil.copy(
            label_path.replace("labels", "images").replace(
                ".txt", f".{config['img_ext']}"
            ),
            os.path.join(
                new_trainset_path,
                f"images/val/{base_name}_man.{config['img_ext']}",
            ),
        )


def extract_gt_cls_hists(manual_bbox_root_dir):
    manual_bbox_paths = glob.glob(os.path.join(manual_bbox_root_dir, "*.txt"))

    gt_cls_hists = []
    for manual_bbox_path in manual_bbox_paths:
        with open(manual_bbox_path) as f:
            bboxes = [list(map(float, line.strip().split())) for line in f]
            gt_cls_list = [int(bbox[0]) for bbox in bboxes]

            # gt_cls_hist = {0: 12, 1: 7, 3: 4, 2: 1, 4: 1, 5: 1, 6: 1, 7: 1}
            gt_cls_hist = dict(Counter(gt_cls_list))
            gt_cls_hists.append(gt_cls_hist)

    return gt_cls_hists


def histogram_distance(h1, h2, metric="l2"):
    # Collect all possible keys
    keys = set(h1.keys()) | set(h2.keys())

    if metric == "l1":
        return sum(abs(h1.get(k, 0) - h2.get(k, 0)) for k in keys)
    elif metric == "l2":
        return math.sqrt(sum((h1.get(k, 0) - h2.get(k, 0)) ** 2 for k in keys))
    else:
        raise ValueError("Unknown metric: choose 'l1' or 'l2'")


def train_rtdetr(
    init_model,
    data_yaml_path,
    tmp_save_path,
    archive_path,
    num_epochs,
):
    if os.path.exists(tmp_save_path):
        shutil.rmtree(tmp_save_path)

    init_model.train(
        data=data_yaml_path,
        epochs=num_epochs,
        batch=8,
        project=tmp_save_path,
        name="large_rtdetr",
        exist_ok=True,  # overwrite the previous checkpoint
        verbose=False,
    )

    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    shutil.move(
        os.path.join(tmp_save_path, "large_rtdetr/weights/best.pt"), archive_path
    )
    shutil.rmtree(tmp_save_path)

    # reload the best model after training
    return RTDETR(archive_path)


def gen_pseudo_annotations(
    rtdetr_model,
    trainset_root_path,
    gt_cls_hists,
    img_ext="JPG",
    dump_vis=False,
):
    input_img_paths = glob.glob(
        os.path.join(trainset_root_path, f"**/*.{img_ext}"),
        recursive=True,
    )
    with silence_ultralytics():
        best_pseudo_bboxes = []
        for image_path in tqdm(input_img_paths, desc="Generating pseudo annotations"):
            results = rtdetr_model(image_path, conf=0.7)
            obj_classes = results[0].boxes.cls
            obj_boxes = results[0].boxes.xyxy
            obj_scores = results[0].boxes.conf
            nms_keep_indices = nms(obj_boxes, obj_scores, 0.95)
            obj_classes = obj_classes[nms_keep_indices]
            obj_boxes = obj_boxes[nms_keep_indices]

            query_cls_hist = dict(Counter([obj_cls.item() for obj_cls in obj_classes]))

            distances = [
                histogram_distance(query_cls_hist, gt, metric="l2")
                for gt in gt_cls_hists
            ]
            if min(distances) == 0:
                best_pseudo_bboxes.append(
                    {
                        "path": image_path,
                        "obj_classes": obj_classes,
                        "obj_boxes": obj_boxes,
                    }
                )

            if dump_vis:
                img_with_masks = results[0].plot(show=False)
                output_path = image_path.replace("data/ad_data/", "rtdetr_preview/")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.imsave(output_path, img_with_masks)

    return best_pseudo_bboxes


def get_max_run_idx(base_dir):
    run_indices = [
        int(d.name.split("_")[-1])
        for d in Path(base_dir).iterdir()
        if d.is_dir() and d.name.startswith("run_") and d.name.split("_")[-1].isdigit()
    ]
    return max(run_indices) if run_indices else None


if __name__ == "__main__":
    config_list = [
        # twin_bracelets
        {
            "ad_dataset": "data/ad_data/twin_bracelets",
            "od_dataset": "data/od_data/twin_bracelets_8shot",
            "checkpoint_path": "checkpoint_archives/twin_bracelets",
            "checkpoint_name": "rtdetr_large_twin_bracelets_manual8_pseudo*_adfullshot_pass*.pt",
            "max_pseudo_samples": 64,
            "pass1_aug": True,
            "pass2_aug": False,
            "pass1_num_epochs": 600,
            "pass2_num_epochs": 600,
            "img_ext": "JPG",
        },
        # screw_bag
        {
            "ad_dataset": "data/ad_data/screw_bag",
            "od_dataset": "data/od_data/screw_bag_6shot",
            "checkpoint_path": "checkpoint_archives/screw_bag",
            "checkpoint_name": "rtdetr_large_screw_bag_manual6_pseudo*_adfullshot_pass*.pt",
            "max_pseudo_samples": 64,
            "pass1_aug": False,
            "pass2_aug": False,
            "pass1_num_epochs": 600,
            "pass2_num_epochs": 600,
            "img_ext": "png",
        },
        # juice_bottle
        {
            "ad_dataset": "data/ad_data/juice_bottle",
            "od_dataset": "data/od_data/juice_bottle_6shot",
            "checkpoint_path": "checkpoint_archives/juice_bottle",
            "checkpoint_name": "rtdetr_large_juice_bottle_manual6_pseudo*_adfullshot_pass*.pt",
            "max_pseudo_samples": 64,
            "pass1_aug": False,
            "pass2_aug": False,
            "pass1_num_epochs": 600,
            "pass2_num_epochs": 600,
            "img_ext": "png",
        },
        # breakfast_box
        {
            "ad_dataset": "data/ad_data/breakfast_box",
            "od_dataset": "data/od_data/breakfast_box_6shot",
            "checkpoint_path": "checkpoint_archives/breakfast_box",
            "checkpoint_name": "rtdetr_large_breakfast_box_manual6_pseudo*_adfullshot_pass*.pt",
            "max_pseudo_samples": 64,
            "pass1_aug": False,
            "pass2_aug": False,
            "pass1_num_epochs": 600,
            "pass2_num_epochs": 600,
            "img_ext": "png",
        },
        # splicing_connectors
        {
            "ad_dataset": "data/ad_data/splicing_connectors",
            "od_dataset": "data/od_data/splicing_connectors_6shot",
            "checkpoint_path": "checkpoint_archives/splicing_connectors",
            "checkpoint_name": "rtdetr_large_splicing_connectors_manual6_pseudo*_adfullshot_pass*.pt",
            "max_pseudo_samples": 64,
            "pass1_aug": False,
            "pass2_aug": False,
            "pass1_num_epochs": 600,
            "pass2_num_epochs": 600,
            "img_ext": "png",
        },
        # pushpins
        {
            "ad_dataset": "data/ad_data/pushpins",
            "od_dataset": "data/od_data/pushpins_6shot",
            "checkpoint_path": "checkpoint_archives/pushpins",
            "checkpoint_name": "rtdetr_large_pushpins_manual6_pseudo*_adfullshot_pass*.pt",
            "max_pseudo_samples": 64,
            "pass1_aug": False,
            "pass2_aug": False,
            "pass1_num_epochs": 600,
            "pass2_num_epochs": 600,
            "img_ext": "png",
        },
    ]

    for config in config_list:
        os.makedirs(config["checkpoint_path"], exist_ok=True)
        base_run_idx = get_max_run_idx(config["checkpoint_path"])
        run_idx = 0 if base_run_idx is None else base_run_idx + 1
        config["checkpoint_path"] = os.path.join(
            config["checkpoint_path"], f"run_{run_idx}"
        )

        print(f"\nStart of finetuning process for {config['ad_dataset']}\n")
        gt_cls_hists = extract_gt_cls_hists(
            os.path.join(config["od_dataset"], "labels/train")
        )
        rtdetr_init_model = RTDETR("./rtdetr-l.pt")

        #############################################
        #   Pass 1                                  #
        #############################################
        od_data_pass1 = config["od_dataset"]
        if config["pass1_aug"]:
            if os.path.exists(od_data_pass1 + "_aug"):
                shutil.rmtree(od_data_pass1 + "_aug")
            shutil.copytree(od_data_pass1, od_data_pass1 + "_aug")
            apply_copy_paste_aug(
                dataset_path=os.path.join(od_data_pass1 + "_aug", "images")
            )
            od_data_pass1 = od_data_pass1 + "_aug"

        ckpt_archive_path = os.path.join(
            config["checkpoint_path"],
            config["checkpoint_name"]
            .replace("pseudo*", "pseudo0")
            .replace("pass*", "pass1"),
        )

        # Train the model
        rtdetr_model_pass1 = train_rtdetr(
            init_model=rtdetr_init_model,
            data_yaml_path=os.path.join(od_data_pass1, "data.yaml"),
            tmp_save_path="./rtdetr_pass1",
            archive_path=ckpt_archive_path,
            num_epochs=config["pass1_num_epochs"],
        )

        # Alternatively, if you already have a first pass checkpoint, load it:
        # rtdetr_model_pass1 = RTDETR("path_to_your_checkpoint.pt")

        best_pseudo_bboxes1 = gen_pseudo_annotations(
            rtdetr_model=rtdetr_model_pass1,
            trainset_root_path=os.path.join(config["ad_dataset"], "train"),
            gt_cls_hists=gt_cls_hists,
            img_ext=config["img_ext"],
            dump_vis=False,
        )
        config["pass1_matches"] = len(best_pseudo_bboxes1)
        print(f"Total of {config['pass1_matches']} matches found after pass 1")

        pseudo_annotations_path = config["ad_dataset"].replace(
            "data/ad_data/", "pseudo_annotations/"
        )
        if os.path.exists(pseudo_annotations_path):
            shutil.rmtree(pseudo_annotations_path)

        # Filter out pseudo annotations we have manual annotations for
        manual_annotations_names = [
            os.path.basename(label_path)
            for label_path in glob.glob(
                os.path.join(
                    config["od_dataset"], f"images/train/*.{config['img_ext']}"
                )
            )
        ]
        best_pseudo_bboxes1 = [
            pseudo_bbox
            for pseudo_bbox in best_pseudo_bboxes1
            if os.path.basename(pseudo_bbox["path"]) not in manual_annotations_names
        ]
        print(
            f"Number of pseudo annotations without a corresponding manual annotation: {len(best_pseudo_bboxes1)}"
        )

        pseudo_bbox_count = min(len(best_pseudo_bboxes1), config["max_pseudo_samples"])
        if pseudo_bbox_count:
            dump_yolo_labels(
                pseudo_annotations_path=pseudo_annotations_path,
                data_list=best_pseudo_bboxes1,
                max_pseudo_samples=config["max_pseudo_samples"],
                data_yaml_src_path=os.path.join(od_data_pass1, "data.yaml"),
                data_yaml_dst_path=os.path.join(
                    pseudo_annotations_path,
                    "data.yaml",
                ),
                img_ext=config["img_ext"],
            )

            # Add manual annotations to the new trainset
            add_manual_annotations_to_trainset(
                config["od_dataset"], pseudo_annotations_path
            )

            config["had_enough_pseudo_annotations"] = True
            od_data_pass2 = pseudo_annotations_path
        else:
            print(
                "Not enough pseudo annotations. Reverting back to manual annotations!"
            )
            config["had_enough_pseudo_annotations"] = False

            os.makedirs(config["checkpoint_path"], exist_ok=True)
            with open(
                os.path.join(config["checkpoint_path"], "run_config.json"), "w"
            ) as _f:
                json.dump(config, _f, indent=4)

            continue

        #############################################
        #   Pass 2                                  #
        #############################################
        if config["pass2_aug"]:
            if os.path.exists(od_data_pass2 + "_aug"):
                shutil.rmtree(od_data_pass2 + "_aug")
            shutil.copytree(od_data_pass2, od_data_pass2 + "_aug")

            apply_copy_paste_aug(
                dataset_path=os.path.join(od_data_pass2 + "_aug", "images")
            )
            od_data_pass2 = od_data_pass2 + "_aug"

        ckpt_archive_path = os.path.join(
            config["checkpoint_path"],
            config["checkpoint_name"]
            .replace("pseudo*", f"pseudo{pseudo_bbox_count}")
            .replace("pass*", "pass2"),
        )

        # Train the model
        rtdetr_model_pass2 = train_rtdetr(
            init_model=rtdetr_init_model,
            data_yaml_path=os.path.join(od_data_pass2, "data.yaml"),
            tmp_save_path="./rtdetr_pass2",
            archive_path=ckpt_archive_path,
            num_epochs=config["pass2_num_epochs"],
        )

        # Alternatively, if you already have a second pass checkpoint, load it:
        # rtdetr_model_pass2 = RTDETR("path_to_your_checkpoint.pt")

        best_pseudo_bboxes2 = gen_pseudo_annotations(
            rtdetr_model=rtdetr_model_pass2,
            trainset_root_path=os.path.join(config["ad_dataset"], "train"),
            gt_cls_hists=gt_cls_hists,
            img_ext=config["img_ext"],
            dump_vis=False,
        )
        config["pass2_matches"] = len(best_pseudo_bboxes2)
        print(f"Total of {config['pass2_matches']} matches found after pass 2")
        print(
            f"Improvement of pass2: {config['pass2_matches'] - config['pass1_matches']}"
        )

        with open(
            os.path.join(config["checkpoint_path"], "run_config.json"), "w"
        ) as _f:
            json.dump(config, _f, indent=4)
