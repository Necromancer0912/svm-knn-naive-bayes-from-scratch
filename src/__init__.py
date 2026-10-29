"""
Classical Classifiers From Scratch (`src/`).

Pure NumPy implementation of Support Vector Machines (Linear & Dual RBF Kernel),
k-Nearest Neighbors (loop-free vectorized distance), and Naive Bayes (Gaussian & Multinomial).
"""

from .classifiers_engine import (
    GaussianNaiveBayesScratch,
    KernelSVMScratch,
    KNNScratch,
    LinearSVMScratch,
    MultinomialNaiveBayesScratch,
    compute_kernel_matrix,
    pairwise_sq_dists_vectorized,
)
from .plot_theme import (
    AMBER,
    BG_AXES,
    BG_CANVAS,
    CLASS_COLORS,
    CMAP_DECISION,
    CMAP_MULTICLASS,
    CMAP_SURFACE,
    CORAL,
    CYAN,
    EMERALD,
    MAGENTA,
    PALETTE,
    PURPLE,
    SLATE_BLUE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    apply_github_dark_style,
    plot_decision_boundary,
    plot_loss_landscape,
    save_fig,
    style_3d_axis,
)

__all__ = [
    "pairwise_sq_dists_vectorized",
    "compute_kernel_matrix",
    "LinearSVMScratch",
    "KernelSVMScratch",
    "KNNScratch",
    "GaussianNaiveBayesScratch",
    "MultinomialNaiveBayesScratch",
    "apply_github_dark_style",
    "style_3d_axis",
    "plot_decision_boundary",
    "plot_loss_landscape",
    "save_fig",
    "BG_CANVAS",
    "BG_AXES",
    "TEXT_PRIMARY",
    "TEXT_MUTED",
    "CYAN",
    "CORAL",
    "AMBER",
    "PURPLE",
    "SLATE_BLUE",
    "EMERALD",
    "MAGENTA",
    "PALETTE",
    "CLASS_COLORS",
    "CMAP_DECISION",
    "CMAP_SURFACE",
    "CMAP_MULTICLASS",
]

