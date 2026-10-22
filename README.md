# Plastic Injection Moulding Quality Detection

6th Semester Mini Project: multi-sensor fusion for automated quality inspection of injection-moulded plastic parts.

The system combines four kinds of data recorded for every injection cycle (thermal images, surface images, process time series and machine parameters) and predicts which defects a part has: sink marks, sprue circle defects, underfilled parts, streaks (levels 1-3), old-granulate contamination, and an overall OK / NOK verdict.

## Table of contents

- [Contributors](#contributors)
- [Repository layout](#repository-layout)
- [Data](#data)
- [How it works](#how-it-works)
- [Training and evaluation](#training-and-evaluation)
- [Results](#results)
- [Demo application](#demo-application)
- [Getting started](#getting-started)
- [Key takeaways](#key-takeaways)
- [Acknowledgements](#acknowledgements)

## Contributors

| Contributor | GitHub | Work |
|---|---|---|
| Sandy | [@crazysandy1](https://github.com/crazysandy1) | Project owner and integration; computer-vision model ([`computervision-model3`](https://github.com/crazysandy1/minipro)) |
| Aditya Raj | commits as `Aditya Raj` | Fusion system (`fusion/`), training loop, demo script, web UI, analysis report |
| LelouchelawlietLIEBERT | [@LelouchelawlietLIEBERT](https://github.com/LelouchelawlietLIEBERT) | Thermal CNN model ([`thermal-cnn-model4`](https://github.com/LelouchelawlietLIEBERT/thermal-cnn)) |
| DarshanCHMSR | [@DarshanCHMSR](https://github.com/DarshanCHMSR) | Sequence / TCN model ([`tcn-model2`](https://github.com/DarshanCHMSR/mini-project-algo)) |

The model folders are Git submodules, each owned by the contributor listed. Roles are taken from the repository history and submodule ownership; edit this table if it does not match who did what.

## Repository layout

```
.
├── fusion/                  # Fusion model package
│   ├── config.py            # paths, hyper-parameters, label list
│   ├── data.py              # loading, CV-image index, multimodal Dataset
│   ├── models.py            # encoders, fusion transformer, DANN head
│   ├── train.py             # 5-fold group CV training loop
│   └── analyze.py           # metrics, plots, report generation
├── run_demo.py              # fast demo pipeline (cached embeddings)
├── server.py                # Flask dashboard
├── templates/index.html     # dashboard UI
├── run.sh                   # one-command launcher
├── data/                    # dataset (see Data)
├── artifacts/               # metrics, plots, REPORT.html, complexity.md
├── computervision-model3/   # submodule: computer-vision model
├── thermal-cnn-model4/      # submodule: thermal CNN model (also holds dataset_V2.parquet)
├── tcn-model2/              # submodule: TCN / time-series model
└── INFO.md                  # full technical write-up
```

After cloning, fetch the model folders with `git submodule update --init --recursive`.

## Data

- Tabular data: [data/dataset.csv](data/dataset.csv) - 564 samples, 334 features from 9 data sources; use `MachineCycleID` to link with images
- Data variables and labels: [data/data_description.pdf](data/data_description.pdf)
- Computer-vision images [6.9 GB]: `data/Rohbilder` - raw surface images per cycle
- Thermal images [1.2 GB]: `data/Thermographie` - infrared images per cycle

> The CV and thermal images are not pushed to this repository because they were obtained through requested access. Download them from the source below and place them in `data/`.

### Dataset info

- 564 injection-moulded parts from 47 experimental points (12 parts each)
- 2 materials: PP (polypropylene) and ABS (70 % recyclate)
- Manually labelled quality (OK / NOK and sub-labels)
- Design of Experiment: 4 parameters varied (cylinder temperature, mould temperature, injection speed, holding pressure)

### Source

- Original dataset: https://b2share.eudat.eu/records/k0v7s-jf859
- Project: ProBayes (SKZ / Fraunhofer IPA, 2021/2022)
- Converted from Parquet (`dataset_v2.parquet`) to CSV

### Defect labels

The 8 binary targets are heavily imbalanced, which is the main difficulty of the task.

| Label | Positives | Rate |
|---|---|---|
| `LBL_NOK` | 155 | 27.5 % |
| `LBL_SinkMarks` | 143 | 25.4 % |
| `LBL_SprueCircle` | 72 | 12.8 % |
| `LBL_Underfilled` | 60 | 10.6 % |
| `LBL_StreaksLevel3` | 48 | 8.5 % |
| `LBL_StreaksLevel1` | 30 | 5.3 % |
| `LBL_StreaksLevel2` | 18 | 3.2 % |
| `LBL_OldGranulate` | 9 | 1.6 % |

## How it works

### The four sensor modalities

| Modality | Input per cycle | Encoder | What it captures |
|---|---|---|---|
| Thermal | 2 infrared frames, 480 x 640 temperature matrices (CSV) + 10 ROI statistics | EfficientNet-B0 + small physics-informed MLP | Surface temperature of the fresh part: uneven cooling (sink marks), cold regions (underfill), variance along walls (streaks) |
| Computer vision | 3 greyscale BMP surface photos (positions 1, 2, 3) | Shared ResNet-50 + cross-section attention | Visible surface condition. In this dataset most defects are thermal rather than visual, so this modality contributes least |
| DXP time series | 8 channels x ~4096 time points | Causal TCN (receptive field 85, squeeze-excitation) | Injection, holding and clamping behaviour during the cycle |
| Tabular | 40 scalar machine parameters | 3-layer MLP (256-256-256) | Setpoints, dosing parameters, environment: the baseline process context |

The 8 DXP channels are injection pressure, injection position, injection velocity, holding pressure, clamping force, mould temperature (ejector side), mould temperature (nozzle side) and dosing volume. The TCN uses causal dilated convolutions, so it only looks backward in time and can run in real time on a live machine.

**Leakage prevention.** The raw dataset contains 102 pre-extracted columns derived from the images (prefixed `SIM_*` and `IR_Img*`). They are removed from the tabular features because they already encode the image content, which would make the CNN encoders redundant and inflate the metrics. The CNNs must learn from raw pixels.

### Architecture

```
Thermal IR frames ──▶ EfficientNet-B0 + ROI MLP ──┐
CV surface images ──▶ ResNet-50 x3 + cross attn ──┤
DXP time series   ──▶ Causal TCN (RF 85, SE) ─────┼──▶ Cross-modal fusion transformer
Tabular params    ──▶ 3-layer MLP ────────────────┘    (4 tokens, 2 layers, 4 heads)
                                                              │
                                        ┌─────────────────────┴───────────────┐
                                        ▼                                     ▼
                              Defect head (MLP)                      DANN head (MLP)
                              8 binary predictions               47-way experiment classifier
                                                                (gradient reversal, training only)
```

1. **Modality-specific encoders.** Each encoder is matched to its data type (see table above) and outputs one embedding.
2. **Cross-modal fusion transformer.** Each embedding is projected to a common token space and given a learned modality embedding. Four tokens pass through a 2-layer, 4-head transformer. A learned attention head pools the valid tokens into one fused vector.
3. **Dual heads.**
   - *Defect head* outputs 8 logits (one per label), trained with focal loss (gamma = 2.5) to focus on rare, hard examples.
   - *DANN head* (domain-adversarial) tries to guess which of the 47 experiments a sample came from. A gradient-reversal layer penalises the feature extractor when that is easy, so the fused representation becomes independent of the specific experiment and generalises better to new process settings.
4. **Missing-modality handling.** Every sample carries a valid flag per modality. A missing modality is replaced by a learned `[MASK]` token and its attention score is set to `-inf`, so the system degrades gracefully, for example if the thermal camera fails.

## Training and evaluation

### Data splitting

Samples from the same experiment share process settings. If they appear in both train and test sets, the model memorises experiment-specific patterns and reported performance is inflated (group leakage). The project therefore uses **5-fold stratified group cross-validation**:

- Group: experiment ID (`MET_ExperimentNumber`); all 12 parts of an experiment stay together
- Stratification: on `LBL_NOK`, so each fold has a similar share of defective parts
- Each fold has about 450 training and 114 test samples from completely different experiments

### Class imbalance

- `WeightedRandomSampler` oversamples rare classes (`OldGranulate` gets about 36x the weight of common classes)
- Focal loss down-weights easy examples
- Per-label F1 is reported, because predicting "no defect" everywhere would already score about 98 % accuracy

### Schedule

- 100 epochs per fold, AdamW (lr 5e-5 for backbones, 1e-3 for heads)
- DANN lambda ramps from 0 to 0.8 over the first 15 epochs, so the model learns the defects first and only then is pushed to be experiment-invariant

### Metrics

Macro F1 (all classes equal), micro F1 (frequency weighted), ROC-AUC, PR-AUC (better for imbalanced data) and per-label F1, reported as mean +/- std across the 5 folds. An ablation study removes one modality at a time (and also runs tabular-only) to measure each sensor's contribution.

## Results

5-fold cross-validated performance on unseen experiments:

| Metric | Value |
|---|---|
| Macro F1 | 0.59 +/- 0.02 |
| Micro F1 | 0.78 +/- 0.02 |
| Mean ROC-AUC | 0.87 +/- 0.02 |
| Mean PR-AUC | 0.71 +/- 0.01 |
| `LBL_NOK` F1 | 0.75 (precision 88.5 %, recall 74.8 %) |

**Ablation findings**

- Tabular is the strongest single modality (F1 = 0.52 alone)
- Full four-modality fusion gives +13 % over tabular alone
- Removing tabular drops performance by 30.5 %, so process context is essential
- Visual contributes least (+3.4 %), because the defects are mostly thermal phenomena

**Practical impact (per 1000 parts):** about 275 are defective; the model catches about 206 (74.8 % recall) with about 8 false alarms (96.9 % specificity), a 73 % reduction in manual QC workload.

**Model size:** 34.5 M parameters (about 130 MB in FP32), 15.76 GFLOPs, about 2.5 min training per fold on GPU. Details in [artifacts/complexity.md](artifacts/complexity.md).

> `artifacts/metrics.json` and the plots in `artifacts/plots/` come from the fast `run_demo.py` pipeline, which trains only the fusion transformer and heads on cached encoder embeddings. Treat them as a demonstration, not as the held-out benchmark above.

## Demo application

`server.py` is a Flask dashboard that:

- lists all 564 samples with ground-truth labels, with filters by defect type and material
- shows thermal heatmaps and gradient maps, CV surface images and DXP time-series plots
- shows modality attention weights
- runs a physics-based threshold predictor (separate from the learned model) as an explainable baseline

## Getting started

Requirements: Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/crazysandy1/plastic-injection-moulding-quality-detection.git
cd plastic-injection-moulding-quality-detection
# place the downloaded images in data/Rohbilder and data/Thermographie
bash run.sh
```

`run.sh` runs `run_demo.py` first if `artifacts/fusion_model.pt` is missing (about 60 s), then starts the dashboard at http://127.0.0.1:5000.

`server.py` and `fusion/config.py` read `dataset_V2.parquet` from `thermal-cnn-model4/`, so that submodule must be checked out.

## Key takeaways

1. Multi-modal fusion beats any single modality.
2. Pre-extracted image features in tabular data cause leakage and must be removed.
3. Group-stratified cross-validation is essential when experiments share process conditions.
4. Class imbalance is the main bottleneck: 4 of 8 labels have fewer than 50 positives.
5. The design handles missing modalities, uses causal convolutions for real-time use, and adds domain adaptation for unseen process conditions.

More detail: [INFO.md](INFO.md) and [artifacts/REPORT.html](artifacts/REPORT.html).

## Acknowledgements

Dataset: ProBayes project, SKZ / Fraunhofer IPA (2021/2022), published at https://b2share.eudat.eu/records/k0v7s-jf859.
