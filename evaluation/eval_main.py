import warnings
import argparse
import os
import sys

sys.path.append(os.getcwd())
from evaluation.base_eval import BaseEval

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="AD utills")
    parser.add_argument("--data_path", type=str, default=None, help="dataset path")
    parser.add_argument("--dataset_name", type=str, default=None, help="dataset name")
    parser.add_argument("--class_name", type=str, default=None, help="category")
    parser.add_argument("--device", type=int, default=None, help="gpu id")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="save results path"
    )
    parser.add_argument(
        "--output_scores_dir", type=str, default=None, help="save scores path"
    )
    parser.add_argument("--save_csv", type=str, default=None, help="save csv")
    parser.add_argument("--save_json", type=str, default=None, help="save json")
    parser.add_argument("--resolution", type=int)
    args = parser.parse_args()
    return args


def load_args(cfg, args):
    cfg["datasets"]["data_path"] = args.data_path
    assert os.path.exists(
        cfg["datasets"]["data_path"]
    ), f"The dataset path {cfg['datasets']['data_path']} does not exist."
    cfg["datasets"]["dataset_name"] = args.dataset_name
    cfg["datasets"]["class_name"] = args.class_name
    cfg["device"] = args.device
    if isinstance(cfg["device"], int):
        cfg["device"] = str(cfg["device"])
    cfg["testing"]["output_dir"] = args.output_dir
    cfg["testing"]["output_scores_dir"] = args.output_scores_dir
    os.makedirs(cfg["testing"]["output_scores_dir"], exist_ok=True)
    cfg["resolution"] = args.resolution

    if args.save_csv.lower() == "true":
        cfg["testing"]["save_csv"] = True
    else:
        cfg["testing"]["save_csv"] = False

    if args.save_json.lower() == "true":
        cfg["testing"]["save_json"] = True
    else:
        cfg["testing"]["save_json"] = False

    return cfg


if __name__ == "__main__":
    args = get_args()
    cfg = load_args(cfg={"datasets": {}, "testing": {}, "models": {}}, args=args)
    print(cfg)
    model = BaseEval(cfg=cfg)
    model.main()
