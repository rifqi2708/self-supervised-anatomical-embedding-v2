"""Streamlit UI for the local Quadra anatomical mask review."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from tools.quadra.mask_review.core import (
    DECISIONS,
    FLAGGED_DECISIONS,
    ReviewError,
    display_to_voxel,
    export_review_state,
    first_pending_item_id,
    flatten_items,
    latest_checkpoint,
    load_decisions,
    load_manifest,
    mask_bbox,
    plane_projection,
    plane_slice,
    projection_bbox,
    save_decision,
    stepped_slice,
    subject_complete,
    verify_checkpoint,
    verify_identity,
    voxel_to_display,
)


WINDOWS = {
    "soft_tissue": {"label": "Soft tissue", "center": 40, "width": 400},
    "lung": {"label": "Lung", "center": -600, "width": 1500},
    "bone": {"label": "Bone", "center": 400, "width": 1800},
}
OVERLAY_MODES = {
    "fill_and_contour": "Fill + contour",
    "fill": "Fill",
    "contour": "Contour",
    "ct_only": "CT only",
}
VIEW_FRAMINGS = {
    "cropped": "Cropped (default)",
    "full_body": "Full body",
}
BOUNDARY_MARGIN_VOXELS = 5
DEFAULT_CROP_MARGIN_VOXELS = 20
DECISION_LABELS = {
    "pending": "Pending",
    "acceptable": "Acceptable",
    "requires_correction": "Requires correction",
    "requires_resegmentation": "Requires re-segmentation",
}
PLANE_AXES = {"sagittal": 0, "coronal": 1, "axial": 2}
AXIS_NAMES = {0: "x", 1: "y", 2: "z"}
PLANE_COLORS = {
    "axial": "#ff4b4b",
    "coronal": "#00d26a",
    "sagittal": "#ffd43b",
}


_clickable_image = components.declare_component(
    "quadra_clickable_image",
    path=str(Path(__file__).with_name("clickable_image_component")),
)
_keyboard_shortcut = components.declare_component(
    "quadra_keyboard_shortcut",
    path=str(Path(__file__).with_name("keyboard_shortcut_component")),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--review-root", type=Path, required=True)
    args, _unknown = parser.parse_known_args()
    return args


@st.cache_data(max_entries=2, show_spinner="Loading CT volume…")
def _load_ct(identity_json: str) -> np.ndarray:
    identity = json.loads(identity_json)
    verify_identity(identity)
    image = nib.as_closest_canonical(nib.load(identity["path"]))
    return np.asanyarray(image.dataobj)


@st.cache_data(max_entries=4, show_spinner="Loading organ mask…")
def _load_mask(identity_json: str) -> np.ndarray:
    identity = json.loads(identity_json)
    verify_identity(identity)
    image = nib.as_closest_canonical(nib.load(identity["path"]))
    return np.asanyarray(image.dataobj).astype(bool)


def _identity_json(identity: dict) -> str:
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _goto(item: dict) -> None:
    st.session_state["_goto_item"] = item["item_id"]
    st.rerun()


def _apply_pending_navigation(item_map: dict[str, dict]) -> None:
    target = st.session_state.pop("_goto_item", None)
    if not target:
        return
    if target not in item_map:
        st.error(f"Cannot navigate to unknown review item: {target}")
        return
    item = item_map[target]
    st.session_state["nav_subject"] = item["subject_id"]
    st.session_state["nav_session"] = item["session"]
    st.session_state["nav_organ"] = item["organ"]


def _render_plane(
    ct: np.ndarray,
    mask: np.ndarray,
    plane: str,
    index: int,
    window: dict[str, float],
    overlay_mode: str,
    opacity: float,
    boundary_margin: int,
    view_framing: str,
    crop_margin: int,
    crosshair_voxel: tuple[int, int, int] | None = None,
):
    ct_slice = plane_slice(ct, plane, index)
    mask_slice = plane_slice(mask, plane, index)
    projection = plane_projection(mask, plane)
    x0, y0, x1, y1, clipped = projection_bbox(projection, boundary_margin)
    vmin = float(window["center"] - window["width"] / 2)
    vmax = float(window["center"] + window["width"] / 2)
    figure, axis = plt.subplots(figsize=(6, 6), dpi=120)
    axis.imshow(ct_slice, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    if overlay_mode in {"fill", "fill_and_contour"} and mask_slice.any():
        axis.imshow(
            np.ma.masked_where(~mask_slice, mask_slice),
            cmap="autumn",
            alpha=opacity,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
    if overlay_mode in {"contour", "fill_and_contour"} and mask_slice.any():
        axis.contour(mask_slice.astype(np.uint8), levels=[0.5], colors=["#00ffff"], linewidths=0.8)
    if overlay_mode != "ct_only":
        axis.add_patch(
            Rectangle(
                (x0, y0),
                max(x1 - x0 - 1, 1),
                max(y1 - y0 - 1, 1),
                fill=False,
                edgecolor="#ff00ff" if clipped else "#00ff7f",
                linewidth=1.2,
                linestyle="--",
            )
        )
    if view_framing == "cropped":
        crop_x0, crop_y0, crop_x1, crop_y1, _ = projection_bbox(
            projection, crop_margin
        )
        axis.set_xlim(crop_x0 - 0.5, crop_x1 - 0.5)
        axis.set_ylim(crop_y1 - 0.5, crop_y0 - 0.5)
    if crosshair_voxel is not None:
        crosshair_x, crosshair_y = voxel_to_display(
            crosshair_voxel, plane, ct.shape
        )
        if plane == "axial":
            vertical_plane, horizontal_plane = "sagittal", "coronal"
        elif plane == "coronal":
            vertical_plane, horizontal_plane = "sagittal", "axial"
        else:
            vertical_plane, horizontal_plane = "coronal", "axial"
        axis.axvline(
            crosshair_x,
            color=PLANE_COLORS[vertical_plane],
            linewidth=1.0,
            alpha=0.95,
        )
        axis.axhline(
            crosshair_y,
            color=PLANE_COLORS[horizontal_plane],
            linewidth=1.0,
            alpha=0.95,
        )
        axis.plot(
            crosshair_x,
            crosshair_y,
            marker="o",
            markerfacecolor="none",
            markeredgecolor="#ffffff",
            markersize=6,
            markeredgewidth=0.9,
        )
    axis.set_title(
        f"{plane.title()} · slice {index}"
        + (" · cropped" if view_framing == "cropped" else " · full body")
        + (" · no mask on this slice" if not mask_slice.any() else "")
    )
    axis.axis("off")
    figure.tight_layout(pad=0.3)
    return figure


def _figure_payload(figure) -> tuple[str, dict[str, float]]:
    """Encode a Matplotlib figure and its axes geometry for click mapping."""
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    axis = figure.axes[0]
    bounds = axis.get_window_extent()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=figure.dpi)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", {
        "image_width": float(width),
        "image_height": float(height),
        "left": float(bounds.x0),
        "top": float(height - bounds.y1),
        "width": float(bounds.width),
        "height": float(bounds.height),
        "x0": float(axis.get_xlim()[0]),
        "x1": float(axis.get_xlim()[1]),
        "y0": float(axis.get_ylim()[0]),
        "y1": float(axis.get_ylim()[1]),
    }


def _show_clickable_figure(
    figure, *, plane: str, current_item_id: str, key: str
) -> dict | None:
    image_data_url, geometry = _figure_payload(figure)
    return _clickable_image(
        image_data_url=image_data_url,
        geometry=geometry,
        current_item_id=current_item_id,
        alt=f"{plane.title()} CT slice. Click to synchronize all three planes.",
        key=key,
        default=None,
    )


def _shortcut_event_id(
    payload: dict | None,
    *,
    current_item_id: str,
    processed_event_id: str | None,
) -> str | None:
    """Return a new shortcut event ID only when it targets the current item."""
    if not payload or payload.get("shortcut") != "accept_next":
        return None
    if payload.get("item_id") != current_item_id:
        return None
    event_id = str(payload.get("event_id", ""))
    if not event_id or event_id == processed_event_id:
        return None
    return event_id


def _next_item(items: list[dict], current_id: str) -> dict | None:
    index = next(i for i, value in enumerate(items) if value["item_id"] == current_id)
    return items[index + 1] if index + 1 < len(items) else None


def _first_item_for_subject(items: list[dict], subject_id: str) -> dict | None:
    return next((item for item in items if item["subject_id"] == subject_id), None)


def main() -> None:
    args = _arguments()
    review_root = args.review_root.expanduser().resolve()
    st.set_page_config(page_title="Quadra mask review", layout="wide")
    st.title("Quadra anatomical mask review")
    st.caption(
        "Direct local NIfTI review. Source CTs and masks are read-only; decisions are "
        "visual QA records, not manual delineations or independent ground truth."
    )

    try:
        manifest = load_manifest(review_root)
        items = flatten_items(manifest)
        item_map = {item["item_id"]: item for item in items}
        _apply_pending_navigation(item_map)
        state = load_decisions(review_root)
    except ReviewError as exc:
        st.error(str(exc))
        st.stop()

    # Reconcile the materialized CSV/summary files from the append-only event
    # log on every app start/rerun.  This makes a process interruption between
    # event fsync and export recover without losing the review action.
    summary = export_review_state(review_root)
    sidebar = st.sidebar
    sidebar.header("Review progress")
    sidebar.progress(
        (len(items) - summary["counts"]["pending"]) / len(items),
        text=f"{len(items) - summary['counts']['pending']:,}/{len(items):,} decided",
    )
    sidebar.caption(
        f"Subjects complete: {summary['subjects_complete']}/{len(manifest['subject_order'])}"
    )
    sidebar.caption(
        "Flagged: "
        f"{summary['counts']['requires_correction'] + summary['counts']['requires_resegmentation']}"
    )

    subjects = manifest["subject_order"]
    if "nav_subject" not in st.session_state:
        st.session_state["nav_subject"] = subjects[0]
    subject = sidebar.selectbox("Subject", subjects, key="nav_subject")
    subject_sessions = [
        value
        for value in manifest["session_order"]
        if any(item["subject_id"] == subject and item["session"] == value for item in items)
    ]
    if st.session_state.get("nav_session") not in subject_sessions:
        st.session_state["nav_session"] = subject_sessions[0]
    session = sidebar.selectbox("Session", subject_sessions, key="nav_session")
    scan_items = [
        item for item in items if item["subject_id"] == subject and item["session"] == session
    ]
    organs = [item["organ"] for item in scan_items]
    if st.session_state.get("nav_organ") not in organs:
        st.session_state["nav_organ"] = organs[0]
    organ_labels = {item["organ"]: item["display_name"] for item in scan_items}
    organ = sidebar.selectbox(
        "Organ",
        organs,
        key="nav_organ",
        format_func=lambda value: organ_labels[value],
    )
    current_id = f"{subject}|{session}|{organ}"
    item = item_map[current_id]
    current = state[current_id]

    queue = sidebar.radio("Quick queue", ("Pending", "Flagged", "All"), horizontal=True)
    if queue == "Pending":
        queue_items = [value for value in items if state[value["item_id"]]["decision"] == "pending"]
    elif queue == "Flagged":
        queue_items = [
            value
            for value in items
            if state[value["item_id"]]["decision"] in FLAGGED_DECISIONS
        ]
    else:
        queue_items = items
    if sidebar.button(
        f"Jump to next {queue.lower()} item ({len(queue_items):,})",
        disabled=not queue_items,
        width="stretch",
    ):
        following = next(
            (value for value in queue_items if items.index(value) > items.index(item)),
            queue_items[0],
        )
        _goto(following)

    sidebar.divider()
    sidebar.subheader("Saved progress")
    checkpoint = latest_checkpoint(review_root)
    resume_item_id = first_pending_item_id(manifest, state)
    if checkpoint is None:
        sidebar.caption("No checksum checkpoint is available yet.")
    else:
        checkpoint_status = verify_checkpoint(review_root, checkpoint)
        sidebar.caption(f"Latest checkpoint: {checkpoint['checkpoint_id']}")
        if checkpoint_status["current"]:
            sidebar.success("Checkpoint matches the current autosaved decision log.")
        else:
            sidebar.info(
                "The review has advanced since this checkpoint. Continue uses the "
                "newer autosaved decisions and does not roll back your work."
            )
    if sidebar.button(
        "Continue from saved progress",
        disabled=checkpoint is None or resume_item_id is None,
        help=(
            "Open the first pending mask using the current autosaved state."
            if resume_item_id is not None
            else "All masks have a saved decision."
        ),
        width="stretch",
    ):
        _goto(item_map[resume_item_id])

    if sidebar.button("Create checksum checkpoint", width="stretch"):
        exported = export_review_state(review_root, create_checkpoint=True)
        sidebar.success(f"Checkpoint created: {exported['checkpoint']['checkpoint_id']}")
        st.rerun()

    try:
        ct = _load_ct(_identity_json(item["ct"]))
        mask = _load_mask(_identity_json(item["mask"]))
        if ct.shape != mask.shape:
            raise ReviewError(f"Canonical CT/mask shapes differ: {ct.shape} vs {mask.shape}")
        bbox = mask_bbox(mask, margin=BOUNDARY_MARGIN_VOXELS)
    except (ReviewError, OSError, ValueError) as exc:
        st.error(f"Source verification or loading failed: {exc}")
        st.stop()

    header_a, header_b, header_c, header_d = st.columns(4)
    header_a.metric("Subject", subject.replace("quadra_hc_", ""))
    header_b.metric("Session", session)
    header_c.metric("Organ", item["display_name"])
    header_d.metric("Current decision", DECISION_LABELS[current["decision"]])
    st.caption(
        f"Sex: {item['sex']} · source: {item['source_selection']} · "
        f"mask voxels: {int(mask.sum()):,}"
    )

    if bbox["touches_volume_boundary"] or bbox["margin_clipped"]:
        axes = sorted(
            set(bbox["boundary_axes"]) | set(bbox["margin_clipped_axes"])
        )
        st.error(
            "Boundary warning: the mask touches the CT boundary or the five-voxel "
            f"review margin is clipped on axis/axes {', '.join(AXIS_NAMES[a] for a in axes)}."
        )

    controls_a, controls_b, controls_c, controls_d = st.columns(4)
    with controls_a:
        window_name = st.selectbox(
            "CT window",
            list(WINDOWS),
            index=list(WINDOWS).index(current.get("window_preset") or "soft_tissue"),
            format_func=lambda value: WINDOWS[value]["label"],
            key=f"window|{current_id}",
        )
    with controls_b:
        overlay_mode = st.selectbox(
            "Overlay",
            list(OVERLAY_MODES),
            index=list(OVERLAY_MODES).index(
                current.get("overlay_mode") or "fill_and_contour"
            ),
            format_func=lambda value: OVERLAY_MODES[value],
            key=f"overlay|{current_id}",
        )
    with controls_c:
        opacity = st.slider(
            "Mask opacity",
            0.05,
            0.90,
            float(current.get("opacity") or 0.35),
            0.05,
            key=f"opacity|{current_id}",
        )
    with controls_d:
        stored_framing = current.get("view_framing") or "cropped"
        if stored_framing not in VIEW_FRAMINGS:
            stored_framing = "cropped"
        view_framing = st.selectbox(
            "View framing",
            list(VIEW_FRAMINGS),
            index=list(VIEW_FRAMINGS).index(stored_framing),
            format_func=lambda value: VIEW_FRAMINGS[value],
            key=f"framing|{current_id}",
            help=(
                "Cropped view uses the organ bounding box with 20 voxels of context. "
                "Full body preserves the complete CT field of view."
            ),
        )
    crop_margin = int(current.get("crop_margin") or DEFAULT_CROP_MARGIN_VOXELS)
    if view_framing == "cropped":
        st.caption(
            f"Cropped view: organ projection plus {crop_margin} voxels of context; "
            f"dashed box remains {BOUNDARY_MARGIN_VOXELS} voxels outside the mask."
        )

    plane_order = ("axial", "coronal", "sagittal")
    slice_keys: dict[str, str] = {}
    for plane in plane_order:
        axis = PLANE_AXES[plane]
        key = f"slice|{current_id}|{plane}"
        slice_keys[plane] = key
        if key not in st.session_state:
            stored = current.get(f"{plane}_slice")
            middle = int(bbox["center"][axis])
            st.session_state[key] = int(stored) if stored not in {None, ""} else middle

    pending_crosshair = st.session_state.pop("_pending_crosshair", None)
    if pending_crosshair and pending_crosshair.get("item_id") == current_id:
        voxel = tuple(int(value) for value in pending_crosshair["voxel"])
        st.session_state[slice_keys["sagittal"]] = voxel[0]
        st.session_state[slice_keys["coronal"]] = voxel[1]
        st.session_state[slice_keys["axial"]] = voxel[2]

    crosshair_voxel = (
        int(st.session_state[slice_keys["sagittal"]]),
        int(st.session_state[slice_keys["coronal"]]),
        int(st.session_state[slice_keys["axial"]]),
    )
    st.caption(
        "Click any image to synchronize the three planes. Selected canonical voxel: "
        f"x={crosshair_voxel[0]}, y={crosshair_voxel[1]}, z={crosshair_voxel[2]}."
    )

    shortcut_events: list[dict] = []
    slice_columns = st.columns(3)
    selected: dict[str, int] = {}
    for column, plane in zip(slice_columns, plane_order):
        with column:
            axis = PLANE_AXES[plane]
            first = int(bbox["start"][axis])
            middle = int(bbox["center"][axis])
            last = int(bbox["end"][axis]) - 1
            key = slice_keys[plane]
            first_col, middle_col, last_col = st.columns(3)
            if first_col.button("First", key=f"first|{current_id}|{plane}", width="stretch"):
                st.session_state[key] = first
                st.rerun()
            if middle_col.button("Middle", key=f"middle|{current_id}|{plane}", width="stretch"):
                st.session_state[key] = middle
                st.rerun()
            if last_col.button("Last", key=f"last|{current_id}|{plane}", width="stretch"):
                st.session_state[key] = last
                st.rerun()
            previous_col, next_slice_col = st.columns(2)
            maximum = int(ct.shape[axis]) - 1
            if previous_col.button(
                "◀ Prev",
                key=f"previous|{current_id}|{plane}",
                disabled=st.session_state[key] <= 0,
                help=f"Move the {plane} view back one slice.",
                width="stretch",
            ):
                st.session_state[key] = stepped_slice(st.session_state[key], -1, maximum)
                st.rerun()
            if next_slice_col.button(
                "Next ▶",
                key=f"next|{current_id}|{plane}",
                disabled=st.session_state[key] >= maximum,
                help=f"Move the {plane} view forward one slice.",
                width="stretch",
            ):
                st.session_state[key] = stepped_slice(st.session_state[key], 1, maximum)
                st.rerun()
            selected[plane] = st.slider(
                f"{plane.title()} slice",
                0,
                maximum,
                key=key,
            )
            figure = _render_plane(
                ct,
                mask,
                plane,
                selected[plane],
                WINDOWS[window_name],
                overlay_mode,
                opacity,
                boundary_margin=BOUNDARY_MARGIN_VOXELS,
                view_framing=view_framing,
                crop_margin=crop_margin,
                crosshair_voxel=crosshair_voxel,
            )
            click = _show_clickable_figure(
                figure,
                plane=plane,
                current_item_id=current_id,
                key=f"clickable|{current_id}|{plane}",
            )
            plt.close(figure)
            if click:
                if click.get("shortcut") == "accept_next":
                    shortcut_events.append(click)
                else:
                    event_id = str(click.get("event_id", ""))
                    processed_key = f"processed_click|{current_id}|{plane}"
                    if event_id and st.session_state.get(processed_key) != event_id:
                        st.session_state[processed_key] = event_id
                        clicked_voxel = display_to_voxel(
                            float(click["display_x"]),
                            float(click["display_y"]),
                            plane,
                            selected[plane],
                            ct.shape,
                        )
                        st.session_state["_pending_crosshair"] = {
                            "item_id": current_id,
                            "voxel": clicked_voxel,
                        }
                        st.rerun()

    st.caption(
        "Crosshair legend: 🟥 Axial plane · 🟩 Coronal plane · "
        "🟨 Sagittal plane"
    )

    page_shortcut = _keyboard_shortcut(
        current_item_id=current_id,
        key=f"keyboard_shortcut|{current_id}",
        default=None,
    )
    if page_shortcut:
        shortcut_events.append(page_shortcut)

    st.subheader("Decision")
    st.caption(
        "Keyboard shortcut: Right Arrow = Acceptable + save + next organ. "
        "It is disabled while a form control or note field has focus."
    )
    decision = st.radio(
        "Review status",
        DECISIONS,
        index=DECISIONS.index(current["decision"]),
        format_func=lambda value: DECISION_LABELS[value],
        horizontal=True,
        key=f"decision|{current_id}",
    )
    note = st.text_area(
        "Reviewer note (required for correction or re-segmentation)",
        value=current.get("note", ""),
        key=f"note|{current_id}",
    )
    view_state = {
        "axial_slice": selected["axial"],
        "coronal_slice": selected["coronal"],
        "sagittal_slice": selected["sagittal"],
        "window_preset": window_name,
        "overlay_mode": overlay_mode,
        "opacity": opacity,
        "view_framing": view_framing,
        "crop_margin": crop_margin,
    }
    following = _next_item(items, current_id)
    processed_shortcut = st.session_state.get("_processed_accept_shortcut")
    shortcut_event_id = next(
        (
            event_id
            for payload in shortcut_events
            if (
                event_id := _shortcut_event_id(
                    payload,
                    current_item_id=current_id,
                    processed_event_id=processed_shortcut,
                )
            )
        ),
        None,
    )
    if shortcut_event_id is not None:
        st.session_state["_processed_accept_shortcut"] = shortcut_event_id
        try:
            save_decision(
                review_root,
                current_id,
                "acceptable",
                note,
                view_state,
            )
        except ReviewError as exc:
            st.error(str(exc))
        else:
            if following is not None:
                _goto(following)
            st.success("Final review item saved as acceptable.")
            st.rerun()

    save_col, next_col, subject_col = st.columns(3)
    if save_col.button("Save decision", type="primary", width="stretch"):
        try:
            save_decision(review_root, current_id, decision, note, view_state)
        except ReviewError as exc:
            st.error(str(exc))
        else:
            st.success("Decision saved.")
            st.rerun()
    if next_col.button("Save + next organ", width="stretch"):
        if decision == "pending":
            st.error("Choose a non-pending decision before advancing.")
        else:
            try:
                save_decision(review_root, current_id, decision, note, view_state)
            except ReviewError as exc:
                st.error(str(exc))
            else:
                if following is not None:
                    _goto(following)
                st.success("Final review item saved.")
    refreshed = load_decisions(review_root)
    current_subject_complete = subject_complete(refreshed, subject)
    subject_index = subjects.index(subject)
    next_subject = subjects[subject_index + 1] if subject_index + 1 < len(subjects) else None
    if subject_col.button(
        "Next subject",
        disabled=not current_subject_complete or next_subject is None,
        help=(
            "Available after every organ in test and retest has a saved decision."
            if not current_subject_complete
            else None
        ),
        width="stretch",
    ):
        target = _first_item_for_subject(items, next_subject)
        if target is not None:
            _goto(target)


if __name__ == "__main__":
    main()
