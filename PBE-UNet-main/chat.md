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

## Current training data-path interface

On review of the current source, `main4B.py` requires `--data_root` and takes direct file paths in `--train_list` and `--val_list`; there is no `--split_dir` option. `MedicalDataSets` takes `data_root`, a `split_file`, and a transform. The split generator writes the lists into `data\busi\` by default, so this is the matching invocation from the project root:

```powershell
& 'C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe' `
  'main4B.py' `
  --data_root '.\data\busi' `
  --train_list '.\data\busi\busi_train82_new.txt' `
  --val_list '.\data\busi\busi_val82_new.txt'
```

Pylance syntax checks passed for `main4B.py`, `dataloader\dataset.py`, and `network\PBEUNet.py`. A runtime dataset-path check was not completed; PyTorch is unresolved in the selected workspace interpreter.

The split-generation script has since been changed to use a deterministic standard-library stratified splitter, so its current version no longer imports scikit-learn (the earlier successful run used the version present at that time).

## Project explanation

Created `explain.md` after reviewing the project files. It documents the dataset-preparation and split workflow, dataset transforms, PBE-UNet architecture, boundary-supervised loss, training loop and defaults, validation metrics, checkpoint/log output, run commands, and limitations such as the missing test-evaluation path.