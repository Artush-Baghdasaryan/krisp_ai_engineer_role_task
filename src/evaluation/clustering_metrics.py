import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

logger = logging.getLogger(__name__)


def evaluate_clustering(
    true_labels: pd.Series,
    predicted_cluster_ids: List[str],
) -> dict[str, float]:
    true_labels = true_labels.astype(str)
    aligned = __align_lengths(true_labels, predicted_cluster_ids)
    if aligned is None:
        return __nan_metrics()

    aligned_true, aligned_pred = aligned
    labels_true = __to_numeric_labels(aligned_true)
    labels_pred = __to_numeric_labels(pd.Series(aligned_pred))

    if __has_insufficient_classes(labels_true, labels_pred):
        return __nan_metrics()

    return __compute_metrics(labels_true, labels_pred)


def __align_lengths(
    true_labels: pd.Series,
    predicted_cluster_ids: List[str],
) -> Tuple[pd.Series, List[str]] | None:
    n_true = len(true_labels)
    n_pred = len(predicted_cluster_ids)

    if n_pred == 0:
        logger.warning("evaluate_clustering: empty predicted_cluster_ids, returning NaN metrics")
        return None

    if n_true != n_pred:
        logger.warning(
            "evaluate_clustering: length mismatch true_labels=%d predicted_cluster_ids=%d, trimming to min",
            n_true,
            n_pred,
        )
        min_len = min(n_true, n_pred)
        true_labels = true_labels.iloc[:min_len]
        predicted_cluster_ids = predicted_cluster_ids[:min_len]

    return true_labels, predicted_cluster_ids


def __to_numeric_labels(series: pd.Series) -> np.ndarray:
    return pd.Categorical(series).codes.astype(np.int32)


def __has_insufficient_classes(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> bool:
    n_unique_true = len(np.unique(labels_true))
    n_unique_pred = len(np.unique(labels_pred))
    if n_unique_true <= 1 and n_unique_pred <= 1:
        logger.warning(
            "evaluate_clustering: at most one unique label in both partitionings, returning NaN metrics",
        )
        return True
    return False


def __compute_metrics(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> dict[str, float]:
    ari = __compute_adjusted_rand_index(labels_true, labels_pred)
    nmi = __compute_normalized_mutual_information(labels_true, labels_pred)
    homogeneity, completeness, v_measure = __compute_v_measure(labels_true, labels_pred)

    return {
        "adjusted_rand_index": float(ari),
        "normalized_mutual_information": float(nmi),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
    }


def __compute_adjusted_rand_index(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> float:
    try:
        return float(adjusted_rand_score(labels_true, labels_pred))
    except Exception as error:
        logger.warning("evaluate_clustering: ARI failed: %s", error)
        return float("nan")


def __compute_normalized_mutual_information(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> float:
    try:
        return float(
            normalized_mutual_info_score(labels_true, labels_pred, average_method="arithmetic"),
        )
    except Exception as error:
        logger.warning("evaluate_clustering: NMI failed: %s", error)
        return float("nan")


def __compute_v_measure(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> Tuple[float, float, float]:
    try:
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
            labels_true,
            labels_pred,
        )
        return float(homogeneity), float(completeness), float(v_measure)
    except Exception as error:
        logger.warning("evaluate_clustering: V-measure failed: %s", error)
        return float("nan"), float("nan"), float("nan")


def __nan_metrics() -> dict[str, float]:
    return {
        "adjusted_rand_index": float("nan"),
        "normalized_mutual_information": float("nan"),
        "homogeneity": float("nan"),
        "completeness": float("nan"),
        "v_measure": float("nan"),
    }
