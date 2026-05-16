"""
Confidence Validator — Confidence Validation & Human Review Flagging.

After the Score Fusion Engine produces a final grade, this module determines
whether the result is reliable enough for auto-approval or requires mandatory
human review.

Rules:
  - Overall confidence < CONFIDENCE_AUTO_APPROVE → flag as NEEDS_REVIEW.
  - Any component confidence < CONFIDENCE_COMPONENT_FLAG → flag that component.
  - Diagram steps with no detections → always flag.
"""
import logging

from app.config import settings

logger = logging.getLogger(__name__)


def validate_confidence(fused_result: dict) -> dict:
    """
    Validate the confidence of a fused grade result and determine review status.

    Args:
        fused_result: Output from score_fusion.fuse_component_scores().

    Returns:
        Updated fused_result dict with added fields:
            - review_status: "AUTO_GRADED" | "NEEDS_REVIEW"
            - review_reasons: list of reason strings
            - flagged_components: list of component types needing review
    """
    overall_confidence = fused_result.get("confidence", 0.0)
    component_grades = fused_result.get("component_grades", [])

    review_reasons = []
    flagged_components = []

    # Check overall confidence
    if overall_confidence < settings.CONFIDENCE_AUTO_APPROVE:
        review_reasons.append(
            f"Overall confidence ({overall_confidence:.2f}) is below "
            f"auto-approve threshold ({settings.CONFIDENCE_AUTO_APPROVE})"
        )

    # Check per-component confidence
    for cg in component_grades:
        comp_type = cg.get("type", "unknown")
        comp_confidence = cg.get("confidence", 0.0)

        if comp_confidence < settings.CONFIDENCE_COMPONENT_FLAG:
            flagged_components.append(comp_type)
            review_reasons.append(
                f"{comp_type.upper()} component confidence ({comp_confidence:.2f}) "
                f"is below threshold ({settings.CONFIDENCE_COMPONENT_FLAG})"
            )

        # Special case: diagram with zero marks and low confidence
        if comp_type == "diagram" and cg.get("marks_awarded", 0) == 0:
            if "diagram" not in flagged_components:
                flagged_components.append("diagram")
            if "Diagram received 0 marks — verify image was processed correctly" not in review_reasons:
                review_reasons.append(
                    "Diagram received 0 marks — verify image was processed correctly"
                )

        # Special case: labels with many missing
        if comp_type == "labels":
            label_details = cg.get("label_details", [])
            missing = [ld for ld in label_details if ld.get("status") == "missing"]
            if len(missing) > len(label_details) / 2 and label_details:
                if "labels" not in flagged_components:
                    flagged_components.append("labels")
                review_reasons.append(
                    f"More than half of expected labels are missing ({len(missing)}/{len(label_details)})"
                )

    # Determine review status
    if review_reasons:
        review_status = "NEEDS_REVIEW"
        logger.info(
            f"Submission flagged for review: {len(review_reasons)} reason(s). "
            f"Flagged components: {flagged_components}"
        )
    else:
        review_status = "AUTO_GRADED"

    # Add to fused result
    fused_result["review_status"] = review_status
    fused_result["review_reasons"] = review_reasons
    fused_result["flagged_components"] = flagged_components

    return fused_result
