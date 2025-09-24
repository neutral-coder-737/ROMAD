from tqdm import tqdm
import os
from pathlib import Path
import numpy as np

from evaluation.utils.json_helpers import dict_to_json


class DumpScores:
    def __init__(self, scores_dir="./output_scores"):
        self.scores_dir = scores_dir
        self.save_scores_precision = 8

    def save_scores(self, image_path_list, pred_img_level, pred_pix_level):
        print(
            f"Saving scores at '{self.scores_dir}' with precision: '{self.save_scores_precision}'"
        )
        for i in tqdm(range(len(image_path_list)), desc=f"Saving scores"):
            image_path = image_path_list[i]
            image_score_path = self.get_scores_path_for_image(image_path)
            os.makedirs(os.path.dirname(image_score_path), exist_ok=True)

            vectorized_enforce_precision = np.vectorize(self.enforce_precision)
            d = {
                "img_level_score": vectorized_enforce_precision(
                    pred_img_level[i], self.save_scores_precision
                ),
                "pix_level_score": vectorized_enforce_precision(
                    pred_pix_level[i], self.save_scores_precision
                ),
            }
            dict_to_json(d, image_score_path)

    def get_scores_path_for_image(self, image_path):
        """example image_path: './data/photovoltaic_module/test/good/037.png'"""
        path = Path(image_path)
        image_name = path.stem
        return_path = self.scores_dir
        for path_part in path.parts[1:-1]:
            return_path = os.path.join(return_path, path_part)
        return os.path.join(return_path, f"{image_name}_scores.json")

    def enforce_precision(self, x, precision):
        return None if x is None else float(f"{x:.{precision}f}")
