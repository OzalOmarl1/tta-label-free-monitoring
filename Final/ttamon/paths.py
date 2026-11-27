"""Depo kökü ve paylaşılan veri/kontrol noktası yolları (veri ve checkpoint'ler depo kökünde tutulur)."""
import os

FINAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(FINAL_DIR)
DATA_DIR = os.environ.get("TTAMON_DATA", os.path.join(REPO_ROOT, "data"))
CKPT_DIR = os.environ.get("TTAMON_CKPT", os.path.join(REPO_ROOT, "src", "models", "checkpoints"))
RESULTS_DIR = os.path.join(FINAL_DIR, "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
FIG_DIR = os.path.join(FINAL_DIR, "figures")

for _d in (RESULTS_DIR, RAW_DIR, TABLES_DIR, LOGS_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)


def ckpt_path(seed: int) -> str:
    return os.path.join(CKPT_DIR, f"resnet18_cifar10_seed{seed}.pth")
