"""
Package huấn luyện mô hình (src/training).
"""

from src.training.losses import MulticlassHingeLoss, LabelSmoothingCrossEntropy
from src.training.calibration import optimize_temperature, platt_scaling_svm, compute_all_thresholds
from src.training.trainer import TorchTrainer, plot_curves
from src.training.grid_search import grid_search_svm, grid_search_cnn

__all__ = [
    "MulticlassHingeLoss",
    "LabelSmoothingCrossEntropy",
    "optimize_temperature",
    "platt_scaling_svm",
    "compute_all_thresholds",
    "TorchTrainer",
    "plot_curves",
    "grid_search_svm",
    "grid_search_cnn"
]
