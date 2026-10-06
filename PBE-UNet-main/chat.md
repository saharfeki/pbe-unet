# BUSI preparation run

## Steps

1. Located the raw BUSI dataset at `C:\Users\User\Downloads\archive\Dataset_BUSI_with_GT`. The script reads PNGs from its `benign` and `malignant` subfolders.
2. Configured the run to use Python 3.12 at `C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe`.
3. The first run reported `ModuleNotFoundError: No module named 'cv2'`. Installed the missing dependency into that interpreter:

   ```powershell
   & 'C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe' -m pip install opencv-python
   ```

4. Ran the preparation script (the project file is named `prepare_busi.py`):

   ```powershell
   & 'C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe' `
     'C:\Users\User\Downloads\PBE-UNet-main\PBE-UNet-main\dataloader\prepare_busi.py' `
     --raw 'C:\Users\User\Downloads\archive\Dataset_BUSI_with_GT' `
     --out 'C:\Users\User\Downloads\PBE-UNet-main\PBE-UNet-main\data\busi'
   ```

## Output

The script printed:

```text
647 cases, 17 with multiple masks, 0 masks resized
```

Prepared files were written to:

- `data\busi\images\` — 647 images (437 benign, 210 malignant)
- `data\busi\masks\0\` — 647 corresponding combined masks (437 benign, 210 malignant)

Counts were verified after the run. The script processes benign and malignant only; the raw dataset's `normal` folder is not included.

## BUSI split generation

1. Confirmed that `data\busi\images\` contained the 647 prepared benign and malignant images.
2. The split script imports `sklearn.model_selection`; installed `scikit-learn` into the configured Python 3.12 environment.
3. From the project root, ran:

   ```powershell
   Set-Location 'C:\Users\User\Downloads\PBE-UNet-main\PBE-UNet-main'
   & 'C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe' 'dataloader\makeSplit.py'
   ```

4. The script printed:

   ```text
   647 cases: 437 benign, 210 malignant
   ```

5. Verified the generated split files and class counts:

   | File | Total | Benign | Malignant |
   | --- | ---: | ---: | ---: |
   | `data\busi\busi_train.txt` | 452 | 305 | 147 |
   | `data\busi\busi_val.txt` | 65 | 44 | 21 |
   | `data\busi\busi_test.txt` | 130 | 88 | 42 |
   | `data\busi\busi_train82_new.txt` | 517 | 349 | 168 |
   | `data\busi\busi_val82_new.txt` | 130 | 88 | 42 |

   Protocol B files form a stratified 70/10/20 train/validation/test split. The `train82_new` and `val82_new` files form the stratified 80/20 split.

## Kaggle training and project-folder cleanup

The error `main4B.py: error: the following arguments are required: --data_root` means the copy of `main4B.py` being executed still declares `--data_root` as required. The updated script makes it optional: it looks for `data/busi` beside the script and defaults to the matching `busi_train.txt` and `busi_val.txt` split lists there. These defaults work regardless of the notebook's current directory. You may also pass all three paths explicitly.

The Kaggle file browser screenshot shows two nested project copies. Use exactly one project root: the deepest folder in the command below, which should contain `main4B.py`, `dataloader/`, `network/`, `utils/`, and `data/`. Do not run a script from one copy while using data or modules from the other. Keep the dataset before removing or renaming any duplicate folders.

1. In Kaggle, select the project folder that contains the code and the prepared dataset. The folder from the reported command is:

   ```text
   /kaggle/working/pbe-unet/PBE-UNet-main/pbe-unet/PBE-UNet-main
   ```

2. Replace that folder's `main4B.py` with the updated version from this project (or pull the updated file into that checkout). Restart the Kaggle kernel after replacing code so it cannot keep running a stale copy.
3. Confirm that the selected project root has the BUSI data layout:

   ```text
   data/busi/images/
   data/busi/masks/0/
   data/busi/busi_train.txt
   data/busi/busi_val.txt
   ```

   If the prepared dataset is mounted from Kaggle as an input dataset instead, use its BUSI directory for `--data_root` and pass absolute paths to its split lists. The expected layout under `--data_root` is still `images/` and `masks/0/`.
4. From a Kaggle notebook cell, run:

   ```python
   %cd /kaggle/working/pbe-unet/PBE-UNet-main/pbe-unet/PBE-UNet-main
   !pwd
   !python main4B.py --data_root data/busi --train_list data/busi/busi_train.txt --val_list data/busi/busi_val.txt --out_dir runs/pbeunet_busi
   ```

   If the split files are missing but `data/busi/images/` and `data/busi/masks/0/` exist, generate them first:

   ```python
   !python dataloader/makeSplit.py --base-dir data/busi
   ```

   For an input-mounted dataset, substitute its absolute paths in the training command. A missing dataset folder or split list now produces a specific path error rather than proceeding with a confusing later failure.
5. Keep only one active checkout for future runs. After verifying the selected root contains all source folders and the prepared data, archive or remove the unused duplicate through Kaggle's file browser. Do not delete the duplicate before confirming it does not contain the only copy of your dataset or checkpoints.
6. Kaggle already provides the notebook's Python runtime; do not copy this computer's `.venv` into Kaggle or install a second PyTorch build over Kaggle's CUDA-enabled one. If imports fail, inspect the missing package first and install only that package in the notebook, then restart the kernel.

The defaults use the stratified 70/10/20 `busi_train.txt` and `busi_val.txt` pair. The split generator also writes `busi_train82_new.txt` and `busi_val82_new.txt` for an 80/20 run; pass those filenames explicitly to train with that protocol.

The updated argument parser was runtime-checked for both its no-argument defaults and explicit path overrides. A full training run was not performed in this local environment because PyTorch and Albumentations are not available in the selected interpreter.

The split-generation script has since been changed to use a deterministic standard-library stratified splitter, so its current version no longer imports scikit-learn (the earlier successful run used the version present at that time).

## Project explanation

Created `explain.md` after reviewing the project files. It documents the dataset-preparation and split workflow, dataset transforms, PBE-UNet architecture, boundary-supervised loss, training loop and defaults, validation metrics, checkpoint/log output, run commands, and limitations such as the missing test-evaluation path.