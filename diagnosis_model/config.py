from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

DIM = 768

# Detection label space: 0=HEALTHY, 1=ABNORMAL
LABELS = ["HEALTHY", "ABNORMAL"]

# Path to RFDETR checkpoint (resolved from package root)
CHECKPOINT_PATH = str((PACKAGE_ROOT.parent / "data" / "coco" / "_merged" / "outputs" / "rfdetr" / "checkpoint_best_total.pth").resolve())

# Path to annotations.jsonl (resolved from package root)
ANNOTATIONS_PATH = str((PACKAGE_ROOT.parent / "data" / "coco" / "annotations.jsonl").resolve())

# Path to raw images (resolved from package root)
IMAGE_ROOT = str((PACKAGE_ROOT.parent / "data" / "raw").resolve())

# Path to CrossAlignFormer checkpoint (saved by train_cross_align.py)
CROSS_ALIGN_CHECKPOINT_DIR = str((PACKAGE_ROOT.parent / "data" / "diagnosis" / "checkpoints").resolve())
CROSS_ALIGN_CHECKPOINT = str((Path(CROSS_ALIGN_CHECKPOINT_DIR) / "epoch_20.pt").resolve())

KNOWLEDGE_BASE_DIR = str((PACKAGE_ROOT.parent / "data" / "diagnosis" / "knowledge_base").resolve())

# Number of slots for cause/treatment
N_CAUSE = 8
N_TREAT = 8

# Default fallback text when user provides none
DEFAULT_TEXT = "魚體外觀異常，似乎生病了。"

# Standard disclaimer used in reports
DISCLAIMER = (
    "本系統所提供的檢索式建議僅供決策參考，最終處置需由具備專業資格的人員依照現場情況與法規判斷後執行。"
)

# Knowledge base default version string
KB_VERSION = "v2025-10-TW"
