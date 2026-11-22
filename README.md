# Object Detection on BDD100K (10K) with Faster R-CNN

Fine-tuning a COCO-pretrained **Faster R-CNN (ResNet-50 FPN)** in PyTorch to detect road users in driving images from the **BDD100K 10K** subset.

Everything lives in a single notebook, `bdd100k-10k_updated.ipynb`, written to run on Kaggle with a GPU.

## Dataset

- **Source:** [BDD100K 10K on Dataset Ninja](https://datasetninja.com/bdd100k-10k), a subset of the [BDD100K](https://www.vis.xyz/bdd100k/) driving dataset from Berkeley DeepDrive.
- **Format:** Supervisely-style. Each split (`train/`, `val/`, `test/`) has an `img/` folder and an `ann/` folder with one `<image>.json` per image. A `meta.json` at the dataset root defines the classes.
- **Annotations:** bounding boxes (rectangles, with a fallback to the extent of polygon points).
- **Classes (10):** bicycle, bus, car, caravan, motorcycle, pedestrian, rider, trailer, train, truck.

Expected layout:

```
bdd100k_10k/
├── meta.json
├── train/{img,ann}
├── val/{img,ann}
└── test/{img,ann}
```

Please check the BDD100K license terms before using the data beyond research or coursework.

## Method

| Component | Choice |
|---|---|
| Model | `torchvision.models.detection.fasterrcnn_resnet50_fpn`, COCO-pretrained |
| Head | Box predictor replaced with `FastRCNNPredictor` (10 classes + background = 11) |
| Preprocessing | `ToTensor()` only, no augmentation |
| Batch size | 2 (train), 1 (val) |
| Optimizer | SGD, lr 0.005, momentum 0.9, weight decay 0.0005 |
| Scheduler | `StepLR`, step 3, gamma 0.1 |
| Epochs | 5 |

Notebook flow:

1. Set `DATA_ROOT` and verify the dataset folders.
2. Build the class-id to label mapping from `meta.json`.
3. Define `BDD100KDataset`, which parses annotations into `boxes`, `labels`, `area` and `iscrowd`, skipping unknown classes and degenerate boxes.
4. Load Faster R-CNN and swap in the new classification head.
5. Train for 5 epochs, tracking train and validation loss per epoch.
6. Plot the loss curves.
7. Run inference on the test images and save annotated images (score threshold 0.5) to `/kaggle/working/preds`.
8. Visualize random predictions with class-colored boxes and labels.
9. Compute accuracy, precision and recall on the validation set and plot a confusion matrix.

## Results

Loss per epoch (average over batches):

| Epoch | Train loss | Val loss |
|:---:|:---:|:---:|
| 1 | 0.8841 | 0.8123 |
| 2 | 0.7832 | 0.8040 |
| 3 | 0.7418 | 0.7867 |
| 4 | 0.6338 | 0.7373 |
| 5 | 0.6068 | 0.7396 |

Each epoch took about 42 minutes on Kaggle (~2,500 s). Validation loss was still falling at epoch 4 and flat at epoch 5, so the model would likely benefit from more epochs and augmentation.

Validation classification metrics (see the note below):

| Metric | Value |
|---|---|
| Accuracy | 0.728 |
| Precision (weighted) | 0.748 |
| Recall (weighted) | 0.728 |

The confusion matrix shows that `car` dominates the data and is predicted well, while rare classes (`caravan`, `trailer`, `train`, `rider`, `motorcycle`) are almost never predicted correctly. Many `pedestrian` and `truck` objects are confused with `car`.

### Note on the evaluation

The metrics above come from a simplified label-matching procedure: for each image, the top-N highest-scoring predicted labels are compared against the N ground-truth labels in order, with no IoU-based box matching. They show how class confusion behaves, but they are **not** standard detection metrics and are not comparable to published mAP numbers. For a rigorous evaluation, compute mAP@[0.5:0.95] (e.g. with `torchmetrics` or `pycocotools`).

## Requirements

```
torch
torchvision
pillow
matplotlib
seaborn
scikit-learn   # confusion matrix / metrics
tqdm
numpy
```

A CUDA GPU is strongly recommended. Training on CPU will be impractically slow.

## How to run

1. Open the notebook on Kaggle and add the BDD100K 10K dataset as input (or download it from Dataset Ninja).
2. Set `DATA_ROOT` in the first cell to the dataset folder, e.g. `/kaggle/input/bdd100k/bdd100k_10k`.
3. Enable a GPU accelerator and run all cells.
4. Annotated test predictions are written to `/kaggle/working/preds`.

To run locally, change `DATA_ROOT` and the `/kaggle/working/preds` output path.

## Known issues and notes

- The notebook contains the training loop twice. The first version has no validation, and the second adds a progress bar, validation loss and timing. Only the second is needed.
- `pretrained=True` is deprecated in recent torchvision. Use `weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT` instead.
- `draw.textsize` was removed in Pillow 10+. Use `draw.textbbox` in `draw_boxes_with_labels` if you hit an error.
- Validation loss is computed by temporarily switching the model to `train()` mode inside the validation loop. This works for detection models but also updates any train-mode behavior, so BatchNorm layers in torchvision's detection backbone are frozen and unaffected.

## Possible improvements

- Add augmentation (horizontal flip, color jitter, scale jitter).
- Handle class imbalance (class-weighted loss, oversampling rare classes).
- Train longer and use a learning-rate schedule with warmup.
- Evaluate with proper mAP and per-class AP.
- Try a stronger or faster backbone (e.g. Faster R-CNN v2, RetinaNet, YOLO).

## Acknowledgements

- BDD100K: Yu et al., *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning*, CVPR 2020.
- Dataset packaging by [Dataset Ninja](https://datasetninja.com/bdd100k-10k).
- Model and utilities from [torchvision](https://pytorch.org/vision/).

## Members

CSE465 (A group of two(2) members)
