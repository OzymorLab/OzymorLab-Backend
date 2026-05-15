"""
Drift detection service — KL divergence computation, mean shift detection, alert generation.
This is the observability layer that makes AIOS different from a grading tool.
"""
import logging
import numpy as np
from scipy.stats import entropy

logger = logging.getLogger(__name__)


def compute_run_distribution(grade_results: list[dict], max_grade: int) -> np.ndarray:
    """Aggregate all submission grade_distributions into a run-level distribution."""
    agg = np.zeros(max_grade + 1)
    for result in grade_results:
        dist = result.get("grade_distribution", [])
        # Pad or truncate to match expected length
        if len(dist) <= max_grade + 1:
            agg[:len(dist)] += np.array(dist)
        else:
            agg += np.array(dist[:max_grade + 1])

    total = agg.sum()
    if total > 0:
        return agg / total
    return agg


def detect_drift(
    current_results: list[dict],
    baseline_results: list[dict],
    max_grade: int = 20,
) -> dict:
    """
    Compare two grading runs using KL divergence and mean shift.

    Args:
        current_results: Grade results from the current run
        baseline_results: Grade results from the baseline run
        max_grade: Maximum possible grade

    Returns:
        Drift report dict with KL divergence, mean shift, severity, etc.
    """
    p = compute_run_distribution(current_results, max_grade)
    q = compute_run_distribution(baseline_results, max_grade)

    # Add epsilon to avoid log(0)
    eps = 1e-10
    kl = float(entropy(p + eps, q + eps))

    grade_range = np.arange(len(p))
    mean_current = float(np.sum(p * grade_range))
    mean_baseline = float(np.sum(q * grade_range))
    mean_shift = mean_current - mean_baseline

    entropy_current = float(entropy(p + eps))
    entropy_baseline = float(entropy(q + eps))

    drift_detected = kl > 0.15 or abs(mean_shift) > 0.5

    if kl > 0.30:
        severity = "HIGH"
    elif kl > 0.15:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {
        "kl_divergence": round(kl, 6),
        "mean_shift": round(mean_shift, 4),
        "entropy_current": round(entropy_current, 4),
        "entropy_baseline": round(entropy_baseline, 4),
        "drift_detected": drift_detected,
        "severity": severity,
        "details": {
            "current_distribution": p.tolist(),
            "baseline_distribution": q.tolist(),
            "current_mean": round(mean_current, 4),
            "baseline_mean": round(mean_baseline, 4),
        },
    }


def compute_run_statistics(grade_results: list[dict], max_grade: int = 20) -> dict:
    """
    Compute aggregated statistics for a grading run.
    Used by the /runs/{id}/statistics endpoint.
    """
    if not grade_results:
        return {
            "submission_count": 0, "graded_count": 0, "failed_count": 0,
            "mean_grade": 0, "median_grade": 0, "p25_grade": 0, "p75_grade": 0,
            "mean_confidence": 0, "mean_latency_ms": 0,
            "aggregate_distribution": [0.0] * (max_grade + 1),
            "most_common_error": None,
        }

    grades = [r["grade"] for r in grade_results if r.get("grade") is not None]
    confidences = [r.get("confidence", 0) for r in grade_results]
    latencies = [r.get("latency_ms", 0) for r in grade_results if r.get("latency_ms")]

    # Error frequency
    error_counts: dict[str, int] = {}
    for r in grade_results:
        for sg in r.get("step_grades", []):
            et = sg.get("error_type")
            if et:
                error_counts[et] = error_counts.get(et, 0) + 1

    most_common_error = max(error_counts, key=error_counts.get) if error_counts else None

    agg_dist = compute_run_distribution(grade_results, max_grade)

    return {
        "submission_count": len(grade_results),
        "graded_count": len(grades),
        "failed_count": len(grade_results) - len(grades),
        "mean_grade": round(float(np.mean(grades)), 2) if grades else 0,
        "median_grade": round(float(np.median(grades)), 2) if grades else 0,
        "p25_grade": round(float(np.percentile(grades, 25)), 2) if grades else 0,
        "p75_grade": round(float(np.percentile(grades, 75)), 2) if grades else 0,
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0,
        "mean_latency_ms": round(float(np.mean(latencies)), 1) if latencies else 0,
        "aggregate_distribution": agg_dist.tolist(),
        "most_common_error": most_common_error,
    }


def generate_alerts(drift_report: dict, run_stats: dict, run_id: str) -> list[dict]:
    """Generate alerts based on drift report and run statistics."""
    alerts = []

    # Drift alert
    if drift_report.get("drift_detected"):
        alerts.append({
            "alert_type": "DRIFT",
            "severity": drift_report["severity"],
            "message": (
                f"Grade distribution drift detected (KL={drift_report['kl_divergence']:.4f}, "
                f"mean shift={drift_report['mean_shift']:+.2f})"
            ),
            "metadata_json": {"kl_divergence": drift_report["kl_divergence"], "mean_shift": drift_report["mean_shift"]},
        })

    # Failure rate alert
    total = run_stats.get("submission_count", 0)
    failed = run_stats.get("failed_count", 0)
    if total > 0 and (failed / total) > 0.05:
        alerts.append({
            "alert_type": "FAILURE_RATE",
            "severity": "HIGH" if (failed / total) > 0.15 else "MEDIUM",
            "message": f"High failure rate: {failed}/{total} submissions ({failed/total*100:.1f}%)",
            "metadata_json": {"failed_count": failed, "total_count": total},
        })

    return alerts
