# KV Candidate-Set Spine Alignment Project

A complete PyTorch reference project for validating the idea:

- candidate label set around `spine_gt`
- KV-candidate alignment scorer (`AlignScoreNet`)
- uncertainty gating from candidate-score entropy
- student segmentation network trained on soft candidate supervision
- optional DRR teacher branch for coordinate-free shape distillation

This project is designed for your setting:

- input: low-quality KV projection
- supervision: `drr_gt` and `spine_gt` are aligned with each other but may be a few millimeters misaligned to KV
- goal: train `KV -> pred_spine` without letting the network collapse to the fixed `spine_gt` position

## What this project contains

- `train_teacher.py`: trains a canonical DRR->spine teacher on aligned DRR/spine pairs
- `train_scorer.py`: trains `AlignScoreNet` on synthetic KV-like images generated from aligned DRR/bone/spine data
- `train_student.py`: trains the final KV student with:
  - candidate-set supervision
  - frozen scorer
  - uncertainty gating
  - frozen teacher shape distillation
- `infer.py`: runs final inference with only the student model

## Data layout

Create a data root like this:

```text
data_root/
  train/
    kv/
      case001.nii.gz
      case002.nii.gz
    drr/
      case001.nii.gz
      case002.nii.gz
    spine/
      case001.nii.gz
      case002.nii.gz
    bone/                # optional for scorer synthetic training
      case001.nii.gz
      case002.nii.gz
  val/
    kv/
    drr/
    spine/
    bone/
  test/
    kv/
```

Matching is done by basename, ignoring extension.

Supported file types:

- `.nii`
- `.nii.gz`
- `.npy`
- `.png`
- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`

## Installation

```bash
cd kv_align_candidate_project
pip install -r requirements.txt
```

## Stage 1: train teacher

```bash
python train_teacher.py --config configs/default.yaml
```

This trains on canonical crops from `drr` + `spine` and saves:

```text
outputs/teacher/best.pt
```

## Stage 2: train scorer

```bash
python train_scorer.py --config configs/default.yaml
```

This trains `AlignScoreNet` on synthetic KV-like images and saves:

```text
outputs/scorer/best.pt
```

## Stage 3: train student

```bash
python train_student.py --config configs/default.yaml
```

This loads the frozen teacher and scorer and saves:

```text
outputs/student/best.pt
```

## Inference

Single file:

```bash
python infer.py \
  --config configs/default.yaml \
  --checkpoint outputs/student/best.pt \
  --input /path/to/one_kv.nii.gz \
  --output_dir outputs/infer_one
```

Directory:

```bash
python infer.py \
  --config configs/default.yaml \
  --checkpoint outputs/student/best.pt \
  --input /path/to/test/kv \
  --output_dir outputs/infer_test
```

Inference saves:

- mask probability map
- binary mask
- overlay png if the input is 2D image-like
- `pred_conf` as a text log

## Key design choices

### Why candidate sets

`spine_gt` is not treated as a single unquestionable position. Instead, a small set of horizontally shifted candidates is generated around it.

### Why scorer

The scorer is a dedicated KV-candidate compatibility judge. It is trained separately to rank the correct aligned candidate above small misaligned candidates.

### Why uncertainty gate

When the scorer cannot clearly distinguish candidates, the position loss is automatically down-weighted.

### Why teacher

The teacher only supplies canonical shape information from DRR space. It does **not** supervise full-image absolute x-position.

## Important notes

1. First validate the **global horizontal candidate set** version. Do not jump immediately to multi-segment piecewise shifts.
2. Keep the teacher and scorer **frozen** during student training.
3. This is meant to be a practical validation baseline for your idea, not a final publication-quality system.
4. Because your KV may be very blurry, the model cannot mathematically guarantee perfect alignment on every case. The `pred_conf` head is included so you can inspect confidence instead of forcing false certainty.

## Suggested first experiment

Use the default config unchanged first:

- candidate shifts: `[-8,-6,-4,-2,0,2,4,6,8]`
- full-resolution student training
- scorer ROI around candidate band
- frozen teacher and scorer

Then inspect whether:

- predictions still collapse to fixed `spine_gt`
- `pred_conf` correlates with your visual trust
- candidate entropy is lower on clearer KV cases

