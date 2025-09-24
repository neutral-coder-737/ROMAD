# CELAD: Compositional Evaluation for Logical Anomaly Detection

This is the official repository for our paper:  
**"CELAD: Compositional Evaluation for Logical Anomaly Detection"**, containing the implementation of **ROMAD**.

---

## Environment Setup
We use `PyTorch==2.6.0` and `CUDA 12.6`.

To install dependencies:
```bash
pip install -r requirements.txt
```


## Datasets
### CELAD Dataset
The proposed CELAD dataset and full descriptions are available [here](https://huggingface.co/datasets/neutral-coder-737/materials/blob/main/CELAD_Dataset.zip).

For confidentiality, the dataset is provided as an encrypted file, with the password included in the supplementary materials for reviewers. The dataset will be made publicly available upon acceptance.

### MVTec LOCO AD Dataset
We also evaluate on the MVTec LOCO AD dataset, accessible [here](https://www.mvtec.com/company/research/datasets/mvtec-loco).

### Dataset Preprocessing
 - **Resizing CELAD:** For efficiency, we resize all CELAD images to 640×640:
    ``` python
    import os
    from PIL import Image

    def resize(directory, quality=90, target_size=640):
        for root, _, files in os.walk(directory):
            for _file in files:
                if _file.lower().endswith(".jpg"):
                    input_image = os.path.join(root, _file)

                    with Image.open(input_image) as img:
                        img = img.resize((target_size, target_size))
                        img.save(
                            input_image,
                            format="JPEG",
                            quality=quality,
                            subsampling=0,
                            optimize=True,
                        )

    resize(directory="path_to_dataset_root_dir")
    ```

 - **Removing structural anomalies from LOCO:** Our method does not address structural anomalies. Remove the `structural_anomalies` subset of MVTec LOCO AD:
    ```
    rm -rf ./path_to_loco_dataset_class/test/structural_anomalies/
    ```

    If this step is skipped, evaluation phase will fail with an error indicating missing anomaly scores for this subset.



## Self-Training
Follow these steps to fine-tune the object detector.
Alternatively, you can use our pre-trained checkpoints available [here](https://huggingface.co/datasets/neutral-coder-737/materials/blob/main/final_checkpoints.tar.gz) and skip this step.

### 1. Prepare Datasets
we provide the few manually annotated samples (in YOLO format) required for fine-tuning the detector. You may use these or prepare your own annotations.

Organize the annotations and their corresponding dataset images in the exact following format:
```
finetune_od/data/od_data
├── pushpins_6shot
│   ├── images
│   │   ├── train
│   │   │   ├── 007.png
│   │   │   ...
│   │   └── val
│   │       ├── 007.png
│   │       ...
│   ├── labels
│   │   ├── train
│   │   │   ├── 007.txt
│   │   │   ...
│   │   └── val
│   │       ├── 007.txt
│   │       ...
│   └── data.yaml
│
├── twin_bracelets_8shot
│   ├── images
│   │   ├── train
│   │   │   ├── twin_a_061.JPG
│   │   │   ...
│   │   └── val
│   │       ├── twin_a_061.JPG
│   │       ...
│   ├── labels
│   │   ├── train
│   │   │   ├── twin_a_061.txt
│   │   │   ...
│   │   └── val
│   │       ├── twin_a_061.txt
│   │       ...
│   └── data.yaml
│
...
```

Organize the complete training set in the following format:
```
finetune_od/data/ad_data
├── pushpins
│   └── train
│       └── good
├── twin_bracelets
│   └── train
│       └── good
│           ├── twin_a
│           └── twin_b
...
```

### 2. Fine-tune Object Detector
Run:
```
cd finetune_od/
python3 finetune_rtdetr.py
```
Checkpoints will be saved in: `finetune_od/checkpoint_archives/<class_name>/run_0/`

### Note
If you encounter an error from the Ultralytics trainer regarding the dataset directory, ensure that `datasets_dir` in
`~/.config/Ultralytics/settings.json` is set as:

```json
"datasets_dir": "/path_to_ROMAD/finetune_od/"
```

## Anomaly Detection Pipeline

### 1. Prepare Datasets
Organize the datasets in the following format:

```
data
├── twin_bracelets
│   ├── test
│   └── train
│
├── splicing_connectors
│   ├── test
│   ├── validation
│   └── train
│
...
```

### 2. Prepare Checkpoints
Set the paths to the fine-tuned object detector checkpoints acquired from [self-training](#self-training) in `dataset_args.yaml`.


### 3. Run the Matching Pipeline
Run:
```
python3 runner.py
```

<!-- ## Citation
```
placeholder
``` -->


## Acknowledgements
We build on the [ultralytics](https://github.com/ultralytics/ultralytics) implementations of RT-DETR and SAM.
Our modifications to the original code can be found in: `ultralytics/ultralytics-modifications.patch`