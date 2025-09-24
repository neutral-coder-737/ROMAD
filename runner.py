import os

VIS_FLAG = True  # other options: False
INCLUDE_AREA = True  # other options: False
INCLUDE_DBA = True  # other options: False
UNMATCHED_POLICY = "max"  # other options: "min", "min_max", "normalized_max"
NMS_THRESHOLD = 0.95  # other options: "None"

DATASET_CLS_LIST = [
    "twin_bracelets",
    "screw_bag",
    "pushpins",
    "breakfast_box",
    "juice_bottle",
    "splicing_connectors",
]


for DATASET_CLS in DATASET_CLS_LIST:
    print("\n--------------------------------------------------------------")
    print(
        f"DATASET_CLS={DATASET_CLS} , INCLUDE_DBA={INCLUDE_DBA}, UNMATCHED_POLICY={UNMATCHED_POLICY}, INCLUDE_AREA={INCLUDE_AREA}, NMS_THRESHOLD={NMS_THRESHOLD}"
    )

    os.system(
        "python3 dump_memory_bank.py "
        + f"--dataset_cls {DATASET_CLS} "
        + f"--include_area {INCLUDE_AREA} "
        + f"--include_dba {INCLUDE_DBA} "
        + f"--vis_flag {VIS_FLAG} "
        + f"--nms_threshold {NMS_THRESHOLD} "
    )

    print("***")

    os.system(
        "python3 infer.py "
        + f"--dataset_cls {DATASET_CLS} "
        + f"--include_area {INCLUDE_AREA} "
        + f"--unmatched_policy {UNMATCHED_POLICY} "
        + f"--include_dba {INCLUDE_DBA} "
        + f"--vis_flag {VIS_FLAG} "
        + f"--nms_threshold {NMS_THRESHOLD} "
    )

    print("***")

    os.system(
        "python3 evaluation/eval_main.py "
        + f"--device 0 "
        + f"--data_path ./data/ "
        + f"--dataset_name custom_dataset "
        + f"--class_name {DATASET_CLS} "
        + f"--output_dir ./output/{DATASET_CLS}/INCLUDE_DBA={INCLUDE_DBA}_UNMATCHED_POLICY={UNMATCHED_POLICY}_INCLUDE_AREA={INCLUDE_AREA}_NMS_THRESHOLD={NMS_THRESHOLD} "
        + f"--output_scores_dir './output_scores/' "
        + f"--resolution 256 "
        + f"--save_csv False "
        + f"--save_json True "
    )

    print("--------------------------------------------------------------\n")
