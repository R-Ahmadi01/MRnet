# MRNet SliceCNN Baseline (PyTorch)

A simple **exam-level** knee MRI classifier for the **MRNet v1.0** dataset.

This baseline:
- Loads MRNet volumes saved as **`.npy`** files (one file per exam per plane)
- Selects a **fixed number of slices** deterministically (no randomness)
- Runs a small **2D CNN per slice**
- **Max-pools across slices** to output **one logit per exam**
- Trains with **BCEWithLogitsLoss** and reports **accuracy**

---

## What is MRNet?

**MRNet** is both:
1) a **public knee MRI dataset + competition**, and  
2) a deep learning model introduced in the MRNet paper.

The dataset contains **1,370 knee MRI exams** from Stanford University Medical Center, labeled for three binary tasks:
- `abnormal` — any abnormality
- `acl` — ACL tear
- `meniscus` — meniscal tear

Official pages:
- Stanford ML Group MRNet competition/dataset page: https://stanfordmlgroup.github.io/competitions/mrnet/
- Stanford AIMI dataset catalog entry: https://aimi.stanford.edu/datasets/mrnet-knee-mris
- MRNet paper (PLOS Medicine, 2018): https://doi.org/10.1371/journal.pmed.1002699

---

## What counts as “ground truth” in MRNet?

### 1) Public training/validation labels (what code is used)
For the publicly released MRNet splits, labels were **manually extracted from clinical reports** (report-based labels).  
So “truth” for training is stored in the CSVs in the dataset root.

### 2) Competition hidden test set (not downloadable)
The MRNet competition used a **hidden test set** for official scoring. Besidwe
- Models were evaluated by submitting code, which was run on a **test set that is not publicly readable**.
- The hidden test set “ground truth” was established using **majority vote of 3 MSK radiologists** (per the competition page / paper).

### 3) The actual imaging data
The MRI images come from DICOM originally, but MRNet v1.0 is commonly distributed in a preprocessed form that includes `.npy` volumes per exam and per plane (what this repo expects).

---

## How to get the MRNet data (officially)

MRNet v1.0 is available for **research use** under Stanford’s **MRNet Dataset Research Use Agreement (RUA)**.

You must register / agree to the RUA to receive the dataset download link by email.
**Do not share** the dataset or the private download link (that violates the agreement).

Start here:
- https://stanfordmlgroup.github.io/competitions/mrnet/
or via AIMI:
- https://aimi.stanford.edu/datasets/mrnet-knee-mris
