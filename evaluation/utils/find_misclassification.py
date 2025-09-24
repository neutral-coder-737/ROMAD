import json
import numpy as np
from torch.utils.data import DataLoader
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())
from custom_datasets.custom_dataset import CustomDataset, DatasetSplit


def get_misclassified_samples(dataloader, output_scores_root_path, opt_thresh):
    normal_mis_list = []
    anomaly_mis_list = []
    for item in tqdm(dataloader):
        image_path = item["image_path"][0]  # string
        json_path = image_path.replace("data", output_scores_root_path)
        for img_ext in [".jpg", ".JPG", ".jpeg", ".png", ".bmp", ".tiff"]:
            json_path = json_path.replace(img_ext, "_scores.json")

        with open(json_path, "r") as _file:
            parsed_file = json.load(_file)
            pr_sp = float(np.array(parsed_file["img_level_score"]["__ndarray__"]))

        if item["is_anomaly"][0]:
            if pr_sp < opt_thresh:
                anomaly_mis_list.append(image_path)
        else:
            if pr_sp > opt_thresh:
                normal_mis_list.append(image_path)

    return normal_mis_list, anomaly_mis_list


def get_args():
    parser = argparse.ArgumentParser(description="AD utills")
    parser.add_argument("--data_path", type=str, default=None, help="dataset path")
    parser.add_argument("--results_path", type=str, default=None, help="results path")
    parser.add_argument("--class_name", type=str, default=None, help="category")
    parser.add_argument(
        "--output_scores_path",
        type=str,
        default="output_scores",
        help="json scores path",
    )
    parser.add_argument(
        "--output_log_path", type=str, default=None, help="output log path"
    )
    args = parser.parse_args()
    return args


def get_available_class_names(root_data_path):
    all_items = os.listdir(root_data_path)
    folder_names = [
        item for item in all_items if os.path.isdir(os.path.join(root_data_path, item))
    ]

    return folder_names


if __name__ == "__main__":
    args = get_args()
    print(args)

    with open(args.results_path, "r") as _file:
        results_file = json.load(_file)
    results_file = {res["Category"]: res for res in results_file}

    class_names = (
        get_available_class_names(args.data_path)
        if args.class_name.lower() == "all"
        else [args.class_name]
    )

    for class_name in class_names:
        print(f"Category: {class_name}")
        test_data = CustomDataset(
            source=args.data_path,
            input_size=256,
            split=DatasetSplit.TEST,
            classname=class_name,
        )
        test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

        normal_mis_list, anomaly_mis_list = get_misclassified_samples(
            test_loader,
            args.output_scores_path,
            results_file[class_name]["image_opt_thresh"],
        )

        # Write logs to file
        os.makedirs(Path(args.output_log_path), exist_ok=True)

        print(f"Number of normal misclassified samples: {len(normal_mis_list)}")
        with open(
            Path(args.output_log_path) / f"{class_name}_normal_misclassification.txt",
            "w",
        ) as _file:
            for line in normal_mis_list:
                _file.write(f"{line}\n")
        print(f"Number of anomaly misclassified samples: {len(anomaly_mis_list)}")
        with open(
            Path(args.output_log_path) / f"{class_name}_anomaly_misclassification.txt",
            "w",
        ) as _file:
            for line in anomaly_mis_list:
                _file.write(f"{line}\n")
