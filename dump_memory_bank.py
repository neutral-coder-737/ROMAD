import os
import sys
import torch
import argparse
from utils import (
    extract_representative_embs,
    silence_ultralytics,
    get_dataset_args,
)

sys.path.append("./ultralytics")
from ultralytics.models.sam.model import SAM
from ultralytics.models.rtdetr.model import RTDETR


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
    parser = argparse.ArgumentParser(description="Memory Bank Generator")
    parser.add_argument(
        "--dataset_cls",
        type=str,
    )
    parser.add_argument(
        "--include_area",
        type=str2bool,
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


def dump_memory_bank(representative_embs_list, level, save_path):
    for representative_embs in representative_embs_list:
        representative_embs["embeddings"] = representative_embs["embeddings"][
            :, level, :
        ].cpu()
        representative_embs["object_classes"] = representative_embs["object_classes"].cpu()

    torch.save(representative_embs_list, save_path)

    d = representative_embs_list[0]["embeddings"][0, :].shape[0]
    print(
        f"Memory bank with shape ({len(representative_embs_list)}, {d}) saved to {save_path}"
    )


if __name__ == "__main__":
    script_args = get_script_args()
    dataset_cls = script_args.dataset_cls

    args = get_dataset_args(dataset_cls)
    include_area = script_args.include_area
    include_dba = script_args.include_dba
    memory_bank_path = f"./memory_banks/{dataset_cls}_memory_bank_area_{include_area}_dba_{include_dba}.pth"

    print(f"Loading checkpoint: {args.checkpoint_path}")
    rtdetr_model = RTDETR(args.checkpoint_path)
    sam_model = SAM("sam2.1_b.pt")

    with silence_ultralytics():
        representative_embs_list, valid_obj_indices_list, valid_obj_classes_list = (
            extract_representative_embs(
                rtdetr_model=rtdetr_model,
                data_dirs=[
                    os.path.join(args.data_path, train_data_dir)
                    for train_data_dir in args.train_data_dirs
                ],
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

    os.makedirs(os.path.dirname(memory_bank_path), exist_ok=True)
    dump_memory_bank(
        representative_embs_list,
        level=0,
        save_path=memory_bank_path,
    )
