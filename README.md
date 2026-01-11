# MRNet SliceCNN Baseline (PyTorch)

A simple, deterministic baseline for the **MRNet v1.0** knee MRI dataset using PyTorch.

This project:
- Loads MRNet `.npy` volumes (per exam) for a chosen **plane** (`axial`, `coronal`, `sagittal`)
- Selects a **fixed number of slices** (deterministically, no randomness)
- Runs a small **2D CNN per slice**
- **Max-pools over slices** to produce one exam-level prediction (binary classification)
- Trains with `BCEWithLogitsLoss`

> Tasks supported: `abnormal`, `acl`, `meniscus`

---

## Repository Contents

- `MRnet.py`: training + validation loop
- Uses MRNet folder structure and label CSVs.

---

