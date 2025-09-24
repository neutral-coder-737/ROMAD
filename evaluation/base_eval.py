import warnings
import os
from pathlib import Path
import csv
import json
import torch
import numpy as np
from tqdm import tqdm


from custom_datasets.custom_dataset import CustomDataset, DatasetSplit
from evaluation.utils.metrics import compute_metrics
from evaluation.utils.json_helpers import json_to_dict

warnings.filterwarnings("ignore")


class BaseEval:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            "cuda:{}".format(cfg["device"]) if torch.cuda.is_available() else "cpu"
        )

        self.path = cfg["datasets"]["data_path"]
        self.dataset = cfg["datasets"]["dataset_name"]
        self.save_csv = cfg["testing"]["save_csv"]
        self.save_json = cfg["testing"]["save_json"]
        self.categories = cfg["datasets"]["class_name"]
        if isinstance(self.categories, str):
            if self.categories.lower() == "all":
                if self.dataset == "custom_dataset":
                    self.categories = self.get_available_class_names(self.path)
            else:
                self.categories = [self.categories]
        self.output_dir = cfg["testing"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.scores_dir = cfg["testing"]["output_scores_dir"]
        self.resolution = cfg["resolution"]

    def get_available_class_names(self, root_data_path):
        all_items = os.listdir(root_data_path)
        folder_names = [
            item
            for item in all_items
            if os.path.isdir(os.path.join(root_data_path, item))
        ]

        return folder_names

    def load_datasets(self, category):
        dataset_classes = {
            "custom_dataset": CustomDataset,
        }

        dataset_splits = {
            "custom_dataset": DatasetSplit.TEST,
        }

        test_dataset = dataset_classes[self.dataset](
            source=self.path,
            input_size=self.resolution,
            split=dataset_splits[self.dataset],
            output_size=self.resolution,
            classname=category,
        )
        return test_dataset

    def get_category_metrics(
        self,
        category,
        included_types=None,
    ):
        print(f"Loading scores of '{category}'")
        gt_sp, pr_sp, gt_px, pr_px, _ = self.load_category_scores(
            category,
            included_types,
        )
        print(
            f"Loaded {len(pr_sp)} samples for anomlay types: {'ALL' if included_types is None else included_types}",
            end="\n",
        )

        print(f"Computing metrics for '{category}'")
        image_metric, pixel_metric = compute_metrics(gt_sp, pr_sp, gt_px, pr_px)

        return image_metric, pixel_metric

    def get_scores_for_image(self, image_path):
        image_scores_path = self.get_scores_path_for_image(image_path)
        image_scores = json_to_dict(image_scores_path)

        return image_scores

    def load_category_scores(
        self,
        category,
        included_types=None,
    ):
        cls_scores_list = []  # image level prediction
        anomaly_maps = []  # pixel level prediction
        gt_list = []  # image level ground truth
        img_masks = []  # pixel level ground truth

        image_path_list = []
        test_dataset = self.load_datasets(category)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        for image_info in tqdm(test_dataloader):
            if not isinstance(image_info, dict):
                raise ValueError("Encountered non-dict image in dataloader")

            del image_info["image"]
            if included_types is not None:
                if (
                    len(
                        set(image_info["anomaly_type_correspondence"][0])
                        & included_types
                    )
                    == 0
                ):
                    continue

            image_path = image_info["image_path"][0]
            image_path_list.extend(image_path)

            img_masks.append(image_info["mask"])
            gt_list.extend(list(image_info["is_anomaly"].numpy()))

            image_scores = self.get_scores_for_image(image_path)
            cls_scores = image_scores["img_level_score"]
            anomaly_maps_iter = image_scores["pix_level_score"]

            cls_scores_list.append(cls_scores)
            anomaly_maps.append(anomaly_maps_iter)

        pr_sp = np.array(cls_scores_list).astype(np.float32)
        gt_sp = np.array(gt_list).astype(np.int32)
        pr_px = np.array(anomaly_maps).astype(np.float32)
        gt_px = torch.cat(img_masks, dim=0).numpy().astype(np.int32)
        # assert pr_px.shape[1:] == (
            # self.resolution,
            # self.resolution,
        # ), "Predicted output scores do not meet the expected shape!"
        assert gt_px.shape[2:] == (
            self.resolution,
            self.resolution,
        ), "Loaded ground truth maps do not meet the expected shape!"

        return gt_sp, pr_sp, gt_px, pr_px, image_path_list

    def get_scores_path_for_image(self, image_path):
        """example image_path: './data/photovoltaic_module/test/good/037.png'"""
        path = Path(image_path)
        image_name = path.stem
        return_path = self.scores_dir
        for path_part in path.parts[1:-1]:
            return_path = os.path.join(return_path, path_part)

        return os.path.join(return_path, f"{image_name}_scores.json")

    def main(self):
        image_auroc_list = []
        image_f1_list = []
        image_ap_list = []
        image_opt_thresh_list = []
        pixel_auroc_list = []
        pixel_f1_list = []
        pixel_ap_list = []
        pixel_aupro_list = []
        pixel_opt_thresh_list = []
        for category in self.categories:
            image_metric, pixel_metric = self.get_category_metrics(
                category=category, included_types=None
            )
            image_auroc, image_f1, image_ap, image_opt_thresh = image_metric
            pixel_auroc, pixel_f1, pixel_ap, pixel_aupro, pixel_opt_thresh = (
                pixel_metric
            )

            image_auroc_list.append(image_auroc)
            image_f1_list.append(image_f1)
            image_ap_list.append(image_ap)
            image_opt_thresh_list.append(image_opt_thresh)
            pixel_auroc_list.append(pixel_auroc)
            pixel_f1_list.append(pixel_f1)
            pixel_ap_list.append(pixel_ap)
            pixel_aupro_list.append(pixel_aupro)
            pixel_opt_thresh_list.append(pixel_opt_thresh)

            print(category)
            print(
                "[image level] auroc:{}, f1:{}, ap:{}, opt_thresh:{}".format(
                    image_auroc,
                    image_f1,
                    image_ap,
                    image_opt_thresh,
                )
            )
            print(
                "[pixel level] auroc:{}, f1:{}, ap:{}, aupro:{}, opt_thresh:{}".format(
                    pixel_auroc,
                    pixel_f1,
                    pixel_ap,
                    pixel_aupro,
                    pixel_opt_thresh,
                )
            )

        image_auroc_mean = sum(image_auroc_list) / len(image_auroc_list)
        image_f1_mean = sum(image_f1_list) / len(image_f1_list)
        image_ap_mean = sum(image_ap_list) / len(image_ap_list)
        pixel_auroc_mean = sum(pixel_auroc_list) / len(pixel_auroc_list)
        pixel_f1_mean = sum(pixel_f1_list) / len(pixel_f1_list)
        pixel_ap_mean = sum(pixel_ap_list) / len(pixel_ap_list)
        pixel_aupro_mean = sum(pixel_aupro_list) / len(pixel_aupro_list)

        print("mean")
        print(
            "[image level] auroc:{}, f1:{}, ap:{}".format(
                image_auroc_mean, image_f1_mean, image_ap_mean
            )
        )
        print(
            "[pixel level] auroc:{}, f1:{}, ap:{}, aupro:{}".format(
                pixel_auroc_mean,
                pixel_f1_mean,
                pixel_ap_mean,
                pixel_aupro_mean,
            )
        )

        # Save the final results as a csv file
        if self.save_csv:
            csv_data = [
                [
                    "Category",
                    "pixel_auroc",
                    "pixel_f1",
                    "pixel_ap",
                    "pixel_aupro",
                    "pixel_opt_thresh",
                    "image_auroc",
                    "image_f1",
                    "image_ap",
                    "image_opt_thresh",
                ]
            ]
            for i, category in enumerate(self.categories):
                csv_data.append(
                    [
                        category,
                        pixel_auroc_list[i],
                        pixel_f1_list[i],
                        pixel_ap_list[i],
                        pixel_aupro_list[i],
                        pixel_opt_thresh_list[i],
                        image_auroc_list[i],
                        image_f1_list[i],
                        image_ap_list[i],
                        image_opt_thresh_list[i],
                    ]
                )
            csv_data.append(
                [
                    "mean",
                    pixel_auroc_mean,
                    pixel_f1_mean,
                    pixel_ap_mean,
                    pixel_aupro_mean,
                    None,
                    image_auroc_mean,
                    image_f1_mean,
                    image_ap_mean,
                    None,
                ]
            )

            csv_file_path = os.path.join(self.output_dir, "results.csv")
            with open(csv_file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(csv_data)

        # Save the final results as a json file
        if self.save_json:
            json_data = []
            for i, category in enumerate(self.categories):
                json_data.append(
                    {
                        "Category": category,
                        "pixel_auroc": pixel_auroc_list[i],
                        "pixel_f1": pixel_f1_list[i],
                        "pixel_ap": pixel_ap_list[i],
                        "pixel_aupro": pixel_aupro_list[i],
                        "pixel_opt_thresh": pixel_opt_thresh_list[i],
                        "image_auroc": image_auroc_list[i],
                        "image_f1": image_f1_list[i],
                        "image_ap": image_ap_list[i],
                        "image_opt_thresh": image_opt_thresh_list[i],
                    }
                )
            json_data.append(
                {
                    "Category": "mean",
                    "pixel_auroc": pixel_auroc_mean,
                    "pixel_f1": pixel_f1_mean,
                    "pixel_ap": pixel_ap_mean,
                    "pixel_aupro": pixel_aupro_mean,
                    "pixel_opt_thresh": None,
                    "image_auroc": image_auroc_mean,
                    "image_f1": image_f1_mean,
                    "image_ap": image_ap_mean,
                    "image_opt_thresh": None,
                }
            )

            json_file_path = os.path.join(self.output_dir, "results.json")
            with open(json_file_path, mode="w") as _file:
                json.dump(json_data, _file, indent=4)
