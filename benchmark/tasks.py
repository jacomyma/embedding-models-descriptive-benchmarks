"""
tasks.py — Benchmark task definitions.

Each Task subclass knows how to:
  1. Load its dataset (HuggingFace datasets or synthetic).
  2. Encode the relevant text fields using a given model (via cache).
  3. Evaluate and return a flat dict of metric_name -> float.

This mirrors the MTEB design where tasks are self-contained units.
"""

from __future__ import annotations

import abc
import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .cache import encode_with_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Task
# ---------------------------------------------------------------------------

class Task(abc.ABC):
    name: str
    description: str = ""

    @abc.abstractmethod
    def run(self, model, cache_dir: Path, **kwargs) -> dict[str, float]:
        """Run the task and return {metric_name: score}."""


# ---------------------------------------------------------------------------
# Likert Continuum (WVS)
# ---------------------------------------------------------------------------

@dataclass
class LikertContinuumWVSTask(Task):
    """
    WVS Likert continuum benchmark.

    For each WVS-derived statement, projects them onto the
    first PCA axis, and computes the absolute Spearman correlation with the
    corresponding numeric codes. Reports the mean correlation across questions.

    WVS: World Values Survey, a large cross-cultural survey with many questions
    measured on a Likert scale.

    Dataset format: a CSV with columns "WVS question", "Statement", "Code".
    """

    name: str = "likert-wvs"
    description: str = (
        "Per-question Spearman correlation between PCA-1 projected statement "
        "embeddings and WVS codes, averaged across questions"
    )
    data_path: Path = Path("data/WVS Statements.csv")

    def run(self, model, cache_dir: Path, **kwargs) -> dict[str, float]:
        from collections import defaultdict

        from scipy.stats import spearmanr
        from sklearn.decomposition import PCA

        csv_path = self.data_path
        if not csv_path.is_file():
            raise FileNotFoundError(f"WVS statements CSV not found: {csv_path}")

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["WVS question"]].append(row)

        # Collect all texts and build mapping of indices to questions
        all_texts = []
        question_indices: dict[str, list[int]] = defaultdict(list)
        question_codes: dict[str, np.ndarray] = {}

        for question, q_rows in grouped.items():
            start_idx = len(all_texts)
            texts = [r["Statement"] for r in q_rows]
            codes = np.array([float(r["Code"]) for r in q_rows], dtype=np.float32)

            all_texts.extend(texts)
            question_indices[question] = list(range(start_idx, len(all_texts)))
            question_codes[question] = codes

        # Encode all texts at once
        embs = encode_with_cache(
            model,
            all_texts,
            dataset_name=self.name,
            cache_dir=cache_dir,
            **kwargs,
        )

        spearman_scores: list[float] = []
        for question, indices in question_indices.items():
            # Extract embeddings for this question
            q_embs = embs[indices]
            codes = question_codes[question]

            pca = PCA(n_components=1, random_state=42)
            axis_values = pca.fit_transform(q_embs).reshape(-1)

            rho = spearmanr(axis_values, codes).statistic
            if np.isnan(rho):
                continue
            spearman_scores.append(float(abs(rho)))

        mean_spearman = float(np.mean(spearman_scores)) if spearman_scores else 0.0

        return {
            "num_rows": float(len(rows)),
            "num_questions": float(len(spearman_scores)),
            "spearman": mean_spearman,
            "main_score": mean_spearman,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, Task] = {
    "likert-wvs": LikertContinuumWVSTask(),
}


def get_tasks(names: list[str]) -> list[Task]:
    missing = set(names) - set(TASK_REGISTRY)
    if missing:
        raise ValueError(f"Unknown tasks: {missing}. Available: {set(TASK_REGISTRY)}")
    return [TASK_REGISTRY[n] for n in names]
