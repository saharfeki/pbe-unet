# PBE-UNet project guide

## 1. What this project does

This is a PyTorch implementation for **binary segmentation of breast-ultrasound lesion regions**. Given an ultrasound image, the network predicts a pixel mask that marks lesion pixels as foreground and everything else as background.

The model is named PBE-UNet (Progressive Boundary-Enhanced U-Net). It combines a U-Net-style encoder/decoder with boundary prediction and feature-attention modules. The boundary heads are auxiliary training outputs: the model returns segmentation logits as its first output and four boundary-logit maps as its second output. The training loss supervises both.

The code's actual task is lesion segmentation, not classifying an image as benign or malignant. The BUSI case names/classes are used by the split generator for stratification; the model itself has one output channel (`num_classes=1`) and predicts lesion/background.

## 2. Project files

| Path | Purpose |
| --- | --- |
| `main4B.py` | Parses training options, constructs datasets/model/optimizer, runs training and validation, and writes checkpoints and metrics. |
| `dataloader/prepare_busi.py` | Converts the raw BUSI class folders into image files and combined binary masks. |
| `dataloader/makeSplit.py` | Creates reproducible, class-stratified train/validation/test and train/validation lists. |
| `dataloader/dataset.py` | Loads an image, its mask, applies transforms, derives a boundary target, and returns tensors as NumPy arrays. |
| `network/PBEUNet.py` | Defines the encoder-decoder network, attention blocks, boundary heads, and boundary-aware feature fusion. |
| `utils/losses_boundary.py` | Defines the segmentation BCE-plus-Dice loss and auxiliary boundary BCE loss. |
| `utils/metrics.py` | Computes overlap metrics, classification-style pixel metrics, HD95, and per-image summaries. |
| `utils/util.py` | Small helpers for parameter counts, booleans, and running averages. It is not used by the current `main4B.py` training path. |
| `README.md` | Brief project title/citation context; it does not provide setup or run instructions. |
| `chat.md` | Notes from the dataset preparation and split-generation runs in this workspace. |
| `data/busi/` | Generated workspace data: `images/`, `masks/0/`, and the split-list text files described below. |

There is no dependency manifest (`requirements.txt` or `pyproject.toml`) in the project root. The small `dataloader/tempCodeRunnerFile.py` file is an editor scratch file, not part of the training pipeline.

## 3. Data preparation and split workflow

### Raw BUSI data to training layout

`prepare_busi.py` expects a raw directory with `benign/` and `malignant/` subdirectories. It:

1. Finds PNG images in those two folders and skips filenames containing `_mask`.
2. Finds each image's matching mask files (`<image>_mask*.png`).
3. Combines multiple masks for one image by taking their pixel-wise union (maximum after binarizing each mask at 127).
4. Resizes a mask to the image dimensions with nearest-neighbor interpolation if they differ.
5. Writes images to `<out>/images/` and masks to `<out>/masks/0/`; a mask is named `<case>_mask.png`.

Normal cases are intentionally not processed. In the run recorded in `chat.md`, the script produced 647 cases: 437 benign and 210 malignant; it reported 17 cases with multiple masks and no resized masks.

Example, from the project root in PowerShell:

```powershell
& 'C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe' `
  'dataloader\prepare_busi.py' `
  --raw 'C:\path\to\Dataset_BUSI_with_GT' `
  --out '.\data\busi'
```

### Split lists

`makeSplit.py` scans `<base-dir>/images/*.png`, excludes case names beginning with `normal`, and labels names beginning with `benign` as benign; other included case names are assigned to the malignant class. It writes one case name per line:

- `busi_train.txt`, `busi_val.txt`, `busi_test.txt`: a stratified 70/10/20 split (Protocol B).
- `busi_train82_new.txt`, `busi_val82_new.txt`: a stratified 80/20 train/validation split (Protocol A).

The script uses a deterministic Python standard-library splitter and defaults to seed 42. Its default base directory is `data/busi`, and it writes the lists directly into that directory. For the prepared 647-case dataset, the recorded run yielded Protocol B counts of 452/65/130, and Protocol A counts of 517/130. Use either protocol consistently when comparing results.

```powershell
& 'C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe' `
  'dataloader\makeSplit.py' `
  --base-dir '.\data\busi' `
  --seed 42
```

The training entry point does **not** read the test list. A test list is generated for later use, but this project currently has no separate test-evaluation or inference command.

## 4. Dataset loading and preprocessing

`MedicalDataSets` in `dataloader/dataset.py` takes:

- `data_root`: a directory containing `images/` and `masks/0/`.
- `split_file`: a text file containing one image case name per line.
- `transform`: the Albumentations transform to apply to the image and mask together.

For each case it reads `<data_root>/images/<case>.png` and `<data_root>/masks/0/<case>_mask.png`. Missing image or mask files raise `FileNotFoundError`. OpenCV loads the image in BGR order; the loader converts it to RGB before applying the transforms. The current normalization is `A.Normalize()` (Albumentations' defaults), and the image is returned as a channel-first `float32` array. After transformation, the mask is thresholded at 127 and returned as a one-channel float array with values 0 or 1.

The boundary target is derived from the transformed mask, so its size matches the resized mask. It is a 3x3 morphological gradient:

```text
boundary = dilate(mask) - erode(mask)
```

The sample dictionary contains `image`, `label`, `boundary`, and `case`.

Training augmentation in `main4B.py` consists of random 90-degree rotations, horizontal and vertical flips, resize to 256x256, and normalization. Validation uses only the 256x256 resize and normalization (no random flips/rotations).

## 5. Network architecture

The network is an encoder-decoder with skip connections. The default channel widths are 16, 32, 128, 160, and 256. The encoder halves the spatial dimensions at each max-pooling step; the decoder upsamples and joins the matching encoder feature map.

| Stage | Resolution for 256x256 input | Channels |
| --- | ---: | ---: |
| Encoder 1 | 256x256 | 16 |
| Encoder 2 | 128x128 | 32 |
| Encoder 3 | 64x64 | 128 |
| Encoder 4 | 32x32 | 160 |
| Encoder 5 / bottleneck | 16x16 | 256 |
| Decoder outputs | 32x32, 64x64, 128x128, 256x256 | 160, 128, 32, 16 |

The input is three-channel RGB ultrasound; the final 1x1 convolution produces one full-resolution segmentation logit map. Each of the four decoder stages also has a one-channel boundary-logit head. `forward()` returns `(segmentation_logits, [b5, b4, b3, b2])`.

Important blocks in `network/PBEUNet.py`:

- **CMUNeXtBlock:** applies depthwise spatial convolution with a residual connection, followed by pointwise channel expansion/contraction and a convolution block that sets the stage's output width. The encoder uses kernel sizes 3, 3, 7, 7, 7 and depths 1, 1, 1, 3, 1 by default.
- **Skip connections / decoder:** each upsampled decoder tensor is concatenated with its same-resolution encoder feature. A `SAAM` block reduces the concatenated channels back to that stage's width.
- **SAAM (scale-aware aggregation):** splits channels into four groups and processes them through a cascade of regular and dilated convolutions (dilation rates 2, 3, and 4), then applies efficient channel attention (ECA) and a residual-style convolutional fusion. This lets the stage aggregate features with different receptive-field scales.
- **BoundaryDetection:** predicts an auxiliary boundary-logit map from each decoder feature.
- **BAFM (boundary-aware feature module):** concatenates the boundary logits with the decoder feature, fuses them, transforms the boundary map into a feature attention signal, and combines boundary-modulated features with a convolutional skip path.
- **CBAM_Attention:** applies channel attention (using global average and maximum pooling) and then spatial attention (using channel-wise average and maximum maps) to refine the decoder features.

Conceptually, the segmentation path estimates the lesion area, while boundary heads provide intermediate edge estimates that are both trained against a boundary target and fed back into decoder features. There is no second benign/malignant classifier head.

## 6. Loss and optimization

`utils/losses_boundary.py` combines the main segmentation objective with boundary supervision.

For segmentation, `BCEDiceLoss` computes:

```text
segmentation_loss = 0.5 * BCEWithLogits(segmentation_logits, mask)
                  + (1 - mean_batch_soft_Dice)
```

The Dice term applies sigmoid to logits and computes soft overlap per image, then averages over the batch. Smoothing constants prevent division by zero.

For each of the four boundary heads, the loss resizes the predicted logits to the ground-truth boundary size and computes binary cross-entropy with logits. The four losses are summed:

```text
total_loss = segmentation_loss + lambda2 * sum(boundary_BCE_at_each_decoder_stage)
```

`lambda2` is the command-line option `--lambda2` (default 0.7). Setting it to 0 disables boundary-loss contribution, although the boundary modules still exist in the network and forward pass.

Training defaults in `main4B.py`:

- 300 epochs; batch size 8.
- SGD, initial learning rate 0.001, momentum 0.9, weight decay 0.0001.
- Polynomial learning-rate decay each training iteration:
  `base_lr * (1 - iter_num / max_iter) ** 0.9`.
- Seed 41; deterministic cuDNN settings are enabled; CUDA is used if available, otherwise CPU.
- The model prints its trainable parameter count. The source notes that the paper reports 4.26 million parameters; the value actually printed is calculated from this code's instantiated model.
- The entry point creates `PBEUNet()` directly and does not load pretrained weights; training starts from the model's initialized parameters.

The training loop updates parameters from the combined segmentation and boundary loss. It also calculates fast thresholded IoU/Dice for training logs; these do not determine the gradients.

## 7. Validation and metrics

Validation runs after every training epoch. The model is switched to evaluation mode and validation is performed under `torch.no_grad()`. Validation uses its deterministic transform and does not update model weights. It computes the same combined loss as training and then calculates segmentation metrics from the final segmentation logits.

Probabilities are formed with sigmoid and thresholded at 0.5. The metrics are calculated per image and then averaged across images (a macro-average), rather than pooling all pixels across the full validation set.

| Metric | Meaning | Better direction |
| --- | --- | --- |
| IoU / Jaccard | `TP / (TP + FP + FN)`; overlap relative to the union. | Higher |
| Dice | `2 TP / (2 TP + FP + FN)`; overlap, also the F1 score for binary masks. | Higher |
| Recall / sensitivity | `TP / (TP + FN)`; fraction of lesion pixels found. | Higher |
| Precision | `TP / (TP + FP)`; fraction of predicted lesion pixels that are correct. | Higher |
| F1 | Harmonic mean of precision and recall. | Higher |
| Specificity | `TN / (TN + FP)`; fraction of background pixels correctly rejected. | Higher |
| Accuracy (`acc`) | `(TP + TN) / all_pixels`. | Higher, but can look high when background dominates |
| HD95 | 95th-percentile Hausdorff boundary distance from MedPy, in pixels at the 256x256 evaluation resolution. | Lower |

HD95 is expensive, so `--hd_every` (default 10) controls its frequency; it is computed on every tenth epoch and on the final epoch. If either the predicted mask or ground truth is empty, the per-image HD95 is NaN. The summary averages only non-NaN HD95 values and records the number of NaN images as `hd95_n_nan`. Other metrics are still computed for those images.

**Important evaluation limitation:** the loop only validates against the validation list. It does not evaluate `busi_test.txt`, and there is no final test-set report in this project. Validation metrics are model-selection diagnostics, not an unbiased held-out test result.

## 8. Checkpoints, logs, and resuming

The default output folder is `runs/pbeunet_seed41` (override with `--out_dir`). It contains:

- `config.json`: the parsed command-line configuration.
- `metrics.csv`: one row per epoch, including learning rate, train/validation loss, train/validation overlap scores, validation pixel metrics, HD95 when scheduled, NaN HD95 count, and elapsed seconds.
- `best_val_iou_model.pth`: model weights from the epoch with the best macro validation IoU so far.
- `last.pth`: latest model weights plus optimizer state, epoch, iteration count, and best IoU. Use `--resume` to load this file and continue from its recorded epoch.

The `last.pth` checkpoint is the final-epoch model when training completes; `best_val_iou_model.pth` is the best validation-IoU model. The program's final message points to `last.pth`. Resuming restores model and optimizer state and the iteration/epoch counters, but the checkpoint does not store RNG state. If you want the best-validation checkpoint for inference, load `best_val_iou_model.pth` explicitly; there is no inference helper in this repository.

## 9. Running this workspace

Run commands from the project root so relative paths such as `.\data\busi` and `.\runs\...` resolve as expected.

First prepare the images and masks, then make the split files:

```powershell
& 'C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe' `
  'dataloader\prepare_busi.py' `
  --raw 'C:\path\to\Dataset_BUSI_with_GT' `
  --out '.\data\busi'

& 'C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe' `
  'dataloader\makeSplit.py' `
  --base-dir '.\data\busi' `
  --seed 42
```

The `main4B.py` CLI accepts the BUSI folder through `--data_root`, and train and validation list **file paths** through `--train_list` and `--val_list`; it does not define a `--split_dir` option. The defaults are anchored to the project/data folder: `data/busi`, `busi_train.txt`, and `busi_val.txt` (the 70/10/20 split). To select paths explicitly, for example:

```powershell
& 'C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe' `
  'main4B.py' `
  --data_root '.\data\busi' `
  --train_list '.\data\busi\busi_train82_new.txt' `
  --val_list '.\data\busi\busi_val82_new.txt' `
  --epochs 300 `
  --batch_size 8 `
  --out_dir '.\runs\pbeunet_busi'
```

To use the 70/10/20 protocol for training and validation, use `busi_train.txt` and `busi_val.txt` instead. `busi_test.txt` is not consumed by `main4B.py`.

### Runtime packages

The imports in the current code require PyTorch, Albumentations, NumPy, OpenCV (`opencv-python`), and MedPy (`medpy`). The workspace's selected Python environment currently resolves only NumPy and OpenCV among those imports; Pylance reports PyTorch, Albumentations, and MedPy unresolved there. Install compatible versions in the interpreter/environment used to run training before attempting a full run. Training may use CUDA when available; CPU fallback exists in the code but will generally be much slower.

## 10. Practical caveats

- `README.md` does not document setup, dependencies, dataset download, or evaluation.
- Use the file-list paths emitted by `makeSplit.py`; the training parser's defaults (`splits/busi_train.txt` and `splits/busi_val.txt`) do not match the split generator's default output location (`data/busi/`). The example command above overrides them.
- The generated test set is not used by the current training script. Do not report validation scores as test scores.
- HD95 is reported in resized-image pixels, not physical millimeters; the code supplies no pixel-spacing metadata.
- Accuracy includes the large background area and should not be interpreted alone. Consider Dice/IoU and precision/recall together.
- The code does not include a prediction export/visualization script or a standalone checkpoint evaluation script.
- This guide describes the checked-in source and recorded data/split preparation, not a completed model-training run or measured segmentation performance.