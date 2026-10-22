"""
config.py — single source of truth for all hyperparameters and paths.
All paths point into project_root/data/ (which contains Thermographie/ and Rohbilder/).
"""

from pathlib import Path

# =========================================================================
# Paths — everything is relative to project root (miniproj/)
# =========================================================================
ROOT = Path(__file__).parent.parent
DATA_DIR       = ROOT / "data"
PARQUET_PATH   = ROOT / "thermal-cnn-model4" / "dataset_V2.parquet"
THERMAL_CSV_DIR = DATA_DIR / "Thermographie"
CV_IMAGE_DIR   = DATA_DIR / "Rohbilder"
ARTIFACTS_DIR  = ROOT / "artifacts"
PLOTS_DIR      = ARTIFACTS_DIR / "plots"

for _d in [ARTIFACTS_DIR, PLOTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# =========================================================================
# Dataset columns
# =========================================================================
LABEL_COLS = [
    "LBL_SinkMarks", "LBL_SprueCircle", "LBL_Underfilled",
    "LBL_OldGranulate", "LBL_StreaksLevel1", "LBL_StreaksLevel2",
    "LBL_StreaksLevel3", "LBL_NOK",
]
N_LABELS = len(LABEL_COLS)
PRIMARY_TARGET = "LBL_NOK"

# IMPROVED: Only raw process parameters (no pre-extracted features)
TABULAR_PREFIXES = ("SET_", "QUA_", "ENV_", "CALC_", "DOS_", "DRY_")
# IR_Img* and SIM_* columns are excluded to avoid feature leakage
SEQUENCE_PREFIX  = "DXP_"
GROUP_COL    = "MET_ExperimentNumber"
CYCLE_ID_COL = "MET_MachineCycleID"
MATERIAL_COL = "MET_MaterialName"

IR_IMG1_COL = "IR_Image1Name"
IR_IMG2_COL = "IR_Image2Name"
CV_IMG1_COL = "CV_Image1Name"

THERMAL_ROI_COLS = [
    "IR_Img1TempSprue", "IR_Img1TempDome",
    "IR_Img1TempEdgeHor", "IR_Img1TempEdgeVer", "IR_Img1TempFull",
    "IR_Img2TempSprue", "IR_Img2TempDome",
    "IR_Img2TempEdgeHor", "IR_Img2TempEdgeVer", "IR_Img2TempFull",
]

# =========================================================================
# Image sizes
# =========================================================================
THERMAL_H, THERMAL_W    = 480, 640
THERMAL_INPUT_SIZE      = 224       # resize target for EfficientNet
CV_INPUT_SIZE           = 224       # resize target per section

# =========================================================================
# Embedding dimensions (IMPROVED: larger for better expressiveness)
# =========================================================================
THERMAL_EMB_DIM  = 512
VISUAL_EMB_DIM   = 512
TCN_EMB_DIM      = 256
TABULAR_EMB_DIM  = 256  # Increased from 192
FUSION_TOKEN_DIM = 384  # Increased from 256
FUSION_N_HEADS   = 4
FUSION_N_LAYERS  = 2
FUSION_DROPOUT   = 0.1

# =========================================================================
# Training (IMPROVED: better hyperparameters for class imbalance)
# =========================================================================
SEED         = 42
BATCH_SIZE   = 24  # Increased from 16
NUM_EPOCHS   = 100  # Increased from 60
LR_BACKBONE  = 5e-5  # Increased from 1e-5 (more learning)
LR_HEAD      = 1e-3  # Increased from 1e-4
WEIGHT_DECAY = 1e-5  # Decreased from 1e-4 (less regularization for small dataset)
PATIENCE     = 15
GRAD_CLIP    = 1.0

# DANN
DANN_LAMBDA_MAX = 0.8  # Increased from 0.5 for stronger domain adaptation
DANN_WARMUP_EP  = 15   # Increased from 10

# Focal loss (IMPROVED: more aggressive for rare classes)
FOCAL_ALPHA = 0.85  # Increased from 0.75
FOCAL_GAMMA = 2.5   # Increased from 2.0

# =========================================================================
# Sequence (TCN)
# =========================================================================
TCN_TARGET_LEN   = 4096
TCN_IN_CHANNELS  = 8
TCN_CHANNEL_LIST = [64, 128, 256]
TCN_KERNEL_SIZE  = 3
TCN_DILATIONS    = [1, 2, 4]
TCN_DROPOUT      = 0.2

KEY_DXP_CHANNELS = [
    "DXP_Inj1PrsAct", "DXP_Inj1PosAct", "DXP_Inj1VelAct",
    "DXP_HldPrsAct", "DXP_ClpFceAct",
    "DXP_MldTmpEjt", "DXP_MldTmpNzl", "DXP_DosVlmAct",
]

# =========================================================================
# Validation / evaluation
# =========================================================================
N_GROUP_FOLDS = 5
VAL_FRACTION  = 0.15
TEST_FRACTION = 0.15
THRESHOLD_GRID = [round(x * 0.05, 2) for x in range(2, 19)]
