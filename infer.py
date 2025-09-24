import os
import sys
import torch
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
import argparse
from utils import (
    extract_representative_embs,
    get_dataset_args,
    silence_ultralytics,
)

sys.path.append("./ultralytics")
from ultralytics.models.sam.model import SAM
from ultralytics.models.rtdetr.model import RTDETR
from evaluation.utils.dump_scores import DumpScores

device = "cuda:0"


def matching_distance(
    nodes1,
    nodes2,
    metric="euclidean",
    embedding_size=256,
    area_coeff=0.5,
    include_area=True,
    include_dba=True,
    unmatched_policy="max",
    num_obj_coeff=0.5,
):
    k1, d1 = nodes1.shape
    k2, d2 = nodes2.shape
    assert (
        d1 == d2
    ), f"Shape mismatch between query : {d1} and memory bank dimension: {d2}"
    embedding_thresh = embedding_size * 2 if include_dba else embedding_size

    nodes_np1 = nodes1[:, :embedding_thresh].cpu().numpy()
    nodes_np2 = nodes2[:, :embedding_thresh].cpu().numpy()

    cost_matrix = (
        cdist(nodes_np1, nodes_np2, metric=metric) / embedding_thresh
    )  # (k1, k2)

    if include_area:
        area_np1 = nodes1[:, embedding_thresh].cpu().numpy().reshape(-1, 1)
        area_np2 = nodes2[:, embedding_thresh].cpu().numpy().reshape(-1, 1)

        cost_matrix_area = cdist(
            area_np1, area_np2, metric=lambda u, v: np.abs((u - v) / (u + v))
        )  # (k1, k2)
        cost_matrix += area_coeff * cost_matrix_area

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    g1_unmatched_elements = set(range(len(nodes_np1))) - set(row_ind)
    g2_unmatched_elements = set(range(len(nodes_np2))) - set(col_ind)
    matchings = {}
    matchings["matched"] = {
        row_idx: col_idx for row_idx, col_idx in zip(row_ind, col_ind)
    }
    matchings["g1_unmatched"] = list(g1_unmatched_elements)
    matchings["g2_unmatched"] = list(g2_unmatched_elements)

    matched_distance = cost_matrix[row_ind, col_ind].sum()

    unmatched_distance = 0
    # Extra object
    for unmatched_elem in g1_unmatched_elements:
        if not len(cost_matrix):
            self_cost_matrix = cdist(nodes_np1, nodes_np1, metric=metric)
            unmatched_distance += self_cost_matrix[unmatched_elem, :].max()
        else:
            if unmatched_policy == "max":
                unmatched_distance += cost_matrix[unmatched_elem, :].max()
            elif unmatched_policy == "min":
                unmatched_distance += cost_matrix[unmatched_elem, :].min()
            elif unmatched_policy == "min_max":
                unmatched_distance += cost_matrix[unmatched_elem, :].min()
            elif unmatched_policy == "normalized_max":
                unmatched_distance += cost_matrix[unmatched_elem, :].max() / (
                    k2 * num_obj_coeff
                )
            else:
                raise Exception(f"Unmatched policy not supported: {unmatched_policy}")

    # Missing object
    for unmatched_elem in g2_unmatched_elements:
        if not len(cost_matrix):
            self_cost_matrix = cdist(nodes_np2, nodes_np2, metric=metric)
            unmatched_distance += self_cost_matrix[:, unmatched_elem].max()
        else:
            if unmatched_policy == "max":
                unmatched_distance += cost_matrix[:, unmatched_elem].max()
            elif unmatched_policy == "min":
                unmatched_distance += cost_matrix[:, unmatched_elem].min()
            elif unmatched_policy == "min_max":
                unmatched_distance += cost_matrix[:, unmatched_elem].max()
            elif unmatched_policy == "normalized_max":
                unmatched_distance += cost_matrix[:, unmatched_elem].max() / (
                    k2 * num_obj_coeff
                )
            else:
                raise Exception(f"Unmatched policy not supported: {unmatched_policy}")

    return matched_distance, unmatched_distance, matchings


def load_memory_bank(load_path):
    memory_bank = torch.load(load_path)
    return memory_bank


def knn_search(
    query,
    memory_bank,
    k,
    embedding_size=256,
    area_coeff=0.5,
    include_area=True,
    include_dba=True,
    unmatched_policy="max",
    metric="euclidean",
):
    distances = []
    for idx, reference_set in enumerate(memory_bank):
        matched_distance, unmatched_distance, _ = matching_distance(
            query,
            reference_set["embeddings"],
            metric=metric,
            embedding_size=embedding_size,
            area_coeff=area_coeff,
            include_area=include_area,
            include_dba=include_dba,
            unmatched_policy=unmatched_policy,
            num_obj_coeff=0.2,
        )
        total_distance = matched_distance + unmatched_distance
        distances.append((idx, total_distance))

    distances.sort(key=lambda x: x[1])
    return distances[:k]


def get_real_paths(data_dirs):
    return [
        os.path.join(data_dir, img_name)
        for data_dir in sorted(data_dirs)
        for img_name in sorted(os.listdir(data_dir))
        if img_name.lower().endswith((".jpg", ".png", ".jpeg"))
    ]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def none_or_float(v):
    return None if v.lower() in {"none", "null"} else float(v)


def get_script_args():
    parser = argparse.ArgumentParser(description="Scores Generator")
    parser.add_argument(
        "--dataset_cls",
        type=str,
    )
    parser.add_argument(
        "--include_area",
        type=str2bool,
    )
    parser.add_argument(
        "--unmatched_policy",
        type=str,
    )
    parser.add_argument(
        "--vis_flag",
        type=str2bool,
    )
    parser.add_argument(
        "--include_dba",
        type=str2bool,
    )
    parser.add_argument(
        "--nms_threshold",
        type=none_or_float,
        help="NMS threshold (float). Use 'None' or 'null' to disable NMS.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    script_args = get_script_args()
    dataset_cls = script_args.dataset_cls
    args = get_dataset_args(dataset_cls=dataset_cls)
    level = 0
    num_nearest_neighbors = 3
    include_area = script_args.include_area
    include_dba = script_args.include_dba
    memory_bank_path = f"./memory_banks/{dataset_cls}_memory_bank_area_{include_area}_dba_{include_dba}.pth"
    test_data_dirs = [
        os.path.join(args.data_path, test_data_dir)
        for test_data_dir in args.test_data_dirs
    ]

    print(f"Loading checkpoint: {args.checkpoint_path}")
    rtdetr_model = RTDETR(args.checkpoint_path)
    sam_model = SAM("sam2.1_b.pt")

    with silence_ultralytics():
        representative_embs_list, valid_obj_indices_list, valid_obj_classes_list = (
            extract_representative_embs(
                rtdetr_model=rtdetr_model,
                data_dirs=test_data_dirs,
                nms_threshold=script_args.nms_threshold,
                obj_threshold=args.obj_threshold,
                vis_path=f"./visualizations/vis_{dataset_cls}",
                include_area=include_area,
                sam_model=sam_model,
                include_dba=include_dba,
                context_count=args.context_count,
                visualize=script_args.vis_flag,
            )
        )

    # Get memory bank
    memory_bank = load_memory_bank(memory_bank_path)
    real_paths = get_real_paths(test_data_dirs)

    pred_img_level = []
    pred_pix_level = []
    for representative_embs in tqdm(representative_embs_list):
        query = representative_embs["embeddings"][:, level, :]
        pred_pix_level.append(None)
        pred_img_level.append(
            np.mean(
                [
                    dist
                    for _, dist in knn_search(
                        query=query,
                        memory_bank=memory_bank,
                        k=num_nearest_neighbors,
                        embedding_size=args.embedding_size,
                        area_coeff=args.area_coeff,
                        include_area=include_area,
                        include_dba=include_dba,
                        unmatched_policy=script_args.unmatched_policy,
                    )
                ]
            )
        )

    assert len(real_paths) == len(pred_img_level)
    assert len(real_paths) == len(pred_pix_level)
    dump_scores = DumpScores(scores_dir="./output_scores")
    dump_scores.save_scores(real_paths, pred_img_level, pred_pix_level)
