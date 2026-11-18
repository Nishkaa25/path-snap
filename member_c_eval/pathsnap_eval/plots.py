from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def make_comparison_plots(results: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = results.copy()

    # CV Macro-F1 with fold standard deviation.
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        data["model"],
        data["cv_macro_f1_mean"],
        yerr=data["cv_macro_f1_std"],
        capsize=4,
    )
    ax.set_ylabel("Cross-validation candidate Macro-F1")
    ax.set_title("Group-aware cross-validation comparison")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "cv_macro_f1_comparison.png", dpi=180)
    plt.close(fig)

    # Student-test Top-1 plus Macro-F1. Macro-F1 is per observation only when
    # true_arc_id was supplied; otherwise it is candidate-row Macro-F1.
    f1_col = (
        "student_observation_arc_macro_f1"
        if "student_observation_arc_macro_f1" in data.columns
        else "observation_arc_macro_f1"
    )
    f1_label = "Student-test observation-level Macro-F1"
    if f1_col not in data.columns or data[f1_col].isna().all():
        f1_col = "student_candidate_macro_f1" if "student_candidate_macro_f1" in data.columns else "candidate_macro_f1"
        f1_label = "Student-test candidate-row Macro-F1"

    top1_col = "student_observation_top1_accuracy" if "student_observation_top1_accuracy" in data.columns else "observation_top1_accuracy"
    plot_df = data[["model", top1_col, f1_col]].copy()
    x = np.arange(len(plot_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, plot_df[top1_col], width, label="Top-1 Accuracy")
    ax.bar(x + width / 2, plot_df[f1_col], width, label=f1_label)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Student-testing performance")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "student_test_accuracy_macro_f1.png", dpi=180)
    plt.close(fig)

    loss_col = "student_log_loss" if "student_log_loss" in data.columns else "log_loss"
    valid_loss = data[np.isfinite(data[loss_col])]
    if not valid_loss.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(valid_loss["model"], valid_loss[loss_col])
        ax.set_ylabel("Candidate-level Log Loss (lower is better)")
        ax.set_title("Student-testing Log Loss")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(output_dir / "student_test_log_loss.png", dpi=180)
        plt.close(fig)

    continuity_col = "student_route_continuity" if "student_route_continuity" in data.columns else "route_continuity"
    if continuity_col in data.columns and data[continuity_col].notna().any():
        valid_continuity = data[data[continuity_col].notna()]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(valid_continuity["model"], valid_continuity[continuity_col])
        ax.set_ylabel("Route continuity")
        ax.set_title("Student-testing route continuity")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(output_dir / "student_test_route_continuity.png", dpi=180)
        plt.close(fig)

    candidate_miss_col = "student_candidate_miss_rate" if "student_candidate_miss_rate" in data.columns else "candidate_miss_rate"
    if candidate_miss_col in data.columns and data[candidate_miss_col].notna().any():
        valid_miss = data[data[candidate_miss_col].notna()]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(valid_miss["model"], valid_miss[candidate_miss_col])
        ax.set_ylabel("Candidate miss rate (lower is better)")
        ax.set_title("Student-testing candidate miss rate")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(output_dir / "student_test_candidate_miss_rate.png", dpi=180)
        plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Candidate classification: {model_name}")
    fig.tight_layout()
    safe_name = "".join(c if c.isalnum() else "_" for c in model_name).strip("_").lower()
    fig.savefig(output_dir / f"confusion_matrix_{safe_name}.png", dpi=180)
    plt.close(fig)
