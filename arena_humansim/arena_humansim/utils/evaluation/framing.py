import numpy as np
import pandas as pd

ABSTRACT_TEMPLATES = {
    "trifurcated": (
        "Our primary calibration: drivers of one family (force, geometric, learned) reproduce each other within "
        "{X_nav:.2f} m mean trajectory Hausdorff under pure navigation, while drivers "
        "of different families diverge by {K_nav:.1f}x; the same comparison gives {K_bt:.1f}x under "
        "behavior-tree load and {K_het:.1f}x under heterogeneous-agent (mixed human + robot) load -- "
        "driver family is a measurable axis of evaluation, and its effect amplifies through both "
        "interaction-layer cascades and kinematic heterogeneity."
    ),
    "nav+bt": (
        "Our primary calibration: drivers of one family reproduce each other within {X_nav:.2f} m mean trajectory Hausdorff under pure navigation, while drivers of different families diverge by {K_nav:.1f}x; this multiplier rises to {K_bt:.1f}x under behavior-tree load -- driver family cascades through the interaction layer."
    ),
    "nav+het": (
        "Our primary calibration: drivers of one family reproduce each other within {X_nav:.2f} m mean trajectory Hausdorff under pure navigation, while drivers of different families diverge by {K_nav:.1f}x; this multiplier rises to {K_het:.1f}x under heterogeneous-agent load -- driver family amplifies under kinematic heterogeneity."
    ),
    "pooled": ("Our primary calibration: drivers of one family (force, geometric, learned) reproduce each other within {X_pooled:.2f} m mean trajectory Hausdorff, while drivers of different families diverge by {K_pooled:.1f}x -- driver family is a measurable axis of evaluation."),
}


def pick_framing(head_df: pd.DataFrame, threshold: float = 1.2) -> dict:
    by_bucket = {row["bucket"]: row for _, row in head_df.iterrows()}
    nav = by_bucket.get("nav")
    if nav is None:
        return {"recommended": "incomplete", "error": "no nav bucket data"}

    bt = by_bucket.get("bt")
    het = by_bucket.get("het")
    pooled = by_bucket.get("all")

    K_nav = float(nav["ratio_K"])  # pyright: ignore[reportArgumentType]
    K_bt = float(bt["ratio_K"]) if bt is not None else float("nan")  # pyright: ignore[reportArgumentType]
    K_het = float(het["ratio_K"]) if het is not None else float("nan")  # pyright: ignore[reportArgumentType]
    K_pooled = float(pooled["ratio_K"]) if pooled is not None else float("nan")  # pyright: ignore[reportArgumentType]

    bt_clears = not np.isnan(K_bt) and K_bt >= K_nav * threshold
    het_clears = not np.isnan(K_het) and K_het >= K_nav * threshold

    if bt_clears and het_clears:
        rec = "trifurcated"
    elif bt_clears:
        rec = "nav+bt"
    elif het_clears:
        rec = "nav+het"
    else:
        rec = "pooled"

    return {
        "recommended": rec,
        "threshold": threshold,
        "X_nav": float(nav["within_class_mean"]),  # pyright: ignore[reportArgumentType]
        "K_nav": K_nav,
        "K_bt": K_bt,
        "K_het": K_het,
        "X_pooled": float(pooled["within_class_mean"]) if pooled is not None else float("nan"),  # pyright: ignore[reportArgumentType]
        "K_pooled": K_pooled,
        "bt_clears": bt_clears,
        "het_clears": het_clears,
    }


def render_framing(framing: dict) -> str:
    rec = framing["recommended"]
    if rec == "incomplete":
        return f"# Abstract framing\n\nINCOMPLETE: {framing.get('error', 'unknown')}\n"

    lines = [
        "# Abstract framing recommendation",
        "",
        f"**Recommended:** `{rec}` (threshold: K_bucket >= {framing['threshold']}x K_nav)",
        "",
        "## Numbers",
        "",
        "| metric | value |",
        "|---|---|",
        f"| X_nav | {framing['X_nav']:.3f} m |",
        f"| K_nav | {framing['K_nav']:.2f}x |",
        f"| K_bt | {framing['K_bt']:.2f}x ({'clears' if framing['bt_clears'] else 'collapses to'} K_nav) |",
        f"| K_het | {framing['K_het']:.2f}x ({'clears' if framing['het_clears'] else 'collapses to'} K_nav) |",
        f"| X_pooled | {framing['X_pooled']:.3f} m |",
        f"| K_pooled | {framing['K_pooled']:.2f}x |",
        "",
        "## Candidate sentences (drop into abstract sentence 5)",
        "",
    ]

    for key, template in ABSTRACT_TEMPLATES.items():
        marker = " - RECOMMENDED" if key == rec else ""
        lines.append(f"### {key}{marker}")
        lines.append("")
        try:
            sentence = template.format(**framing)
        except (KeyError, ValueError) as e:
            sentence = f"_(missing data: {e})_"
        lines.append(f"> {sentence}")
        lines.append("")

    return "\n".join(lines)
