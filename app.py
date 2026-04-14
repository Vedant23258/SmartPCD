"""Responsive Streamlit frontend for Smart Pothole and Crack Detection."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from classifier import CLASS_NAMES, evaluate_model, load_model, predict_class, save_model, train_svm
from main import LABEL_MAP, VALID_EXTENSIONS, iter_dataset_images, load_image, process_image

MODEL_PATH = ROOT_DIR / "models" / "svm_model.joblib"
OUTPUT_DIR = ROOT_DIR / "outputs"
TRAINING_HISTORY_PATH = OUTPUT_DIR / "training_history.json"
PREDICTION_HISTORY_PATH = OUTPUT_DIR / "prediction_history.json"
MIN_CONFIDENCE_THRESHOLD = 0.55


def get_dataset_summary(dataset_dir: Path) -> dict[str, int]:
    summary: dict[str, int] = {}
    for class_name in LABEL_MAP:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            summary[class_name] = 0
            continue
        summary[class_name] = sum(
            1 for path in class_dir.iterdir() if path.suffix.lower() in VALID_EXTENSIONS
        )
    return summary


def prepare_dataset_arrays(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    for image_path, label in iter_dataset_images(dataset_dir):
        image = load_image(image_path)
        outputs = process_image(image)
        feature_rows.append(outputs["feature_vector"])
        labels.append(label)

    if not feature_rows:
        raise ValueError(
            "Dataset is empty. Add images to dataset/normal, dataset/crack, pothole, and severe."
        )

    return np.vstack(feature_rows), np.array(labels, dtype=np.int32)


def train_from_ui(dataset_dir: Path) -> dict[str, object]:
    from sklearn.model_selection import train_test_split

    features, labels = prepare_dataset_arrays(dataset_dir)
    unique_labels = np.unique(labels)
    metrics: dict[str, object] = {
        "samples": int(len(labels)),
        "class_counts": {CLASS_NAMES[idx]: int(np.sum(labels == idx)) for idx in CLASS_NAMES},
        "validated": False,
    }

    if len(labels) >= 8 and len(unique_labels) > 1:
        bincount = np.bincount(labels, minlength=max(CLASS_NAMES.keys()) + 1)
        stratify = labels if np.min(bincount[bincount > 0]) >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )
        model = train_svm(x_train, y_train)
        evaluation = evaluate_model(model, x_test, y_test)
        predictions = model.predict(x_test)
        metrics["validated"] = True
        metrics["accuracy"] = float(evaluation["accuracy"])
        metrics["report"] = str(evaluation["report"])
        metrics["confusion_matrix"] = confusion_matrix(
            y_test,
            predictions,
            labels=sorted(CLASS_NAMES.keys()),
        ).tolist()
    else:
        model = train_svm(features, labels)

    save_model(model, MODEL_PATH)
    metrics["model_path"] = str(MODEL_PATH)
    return metrics


def load_json_history(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def append_json_history(path: Path, entry: dict[str, object]) -> None:
    history = load_json_history(path)
    history.append(entry)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def plot_confusion_matrix(matrix_values: list[list[int]]) -> plt.Figure:
    labels = [CLASS_NAMES[idx] for idx in sorted(CLASS_NAMES.keys())]
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix_values, cmap="YlOrRd")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Validation Confusion Matrix")

    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, str(matrix_values[row][col]), ha="center", va="center")

    fig.tight_layout()
    return fig


def create_overlay_image(original_bgr: np.ndarray, defect_mask: np.ndarray) -> np.ndarray:
    """Overlay segmented defect contours on the original image."""

    overlay = original_bgr.copy()
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (24, 230, 120), 2)

    highlighted = overlay.copy()
    highlighted[defect_mask > 0] = (0, 95, 255)
    blended = cv2.addWeighted(overlay, 0.72, highlighted, 0.28, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def assess_road_likelihood(image: np.ndarray) -> tuple[bool, dict[str, float | bool | str]]:
    """Estimate whether the input resembles a road image.

    This is a lightweight heuristic gate, not a perfect road detector.
    It helps reject obviously unrelated inputs before SVM classification.
    """

    height, width = image.shape[:2]
    if height < 80 or width < 80:
        return False, {
            "is_road_like": False,
            "reason": "Image is too small for reliable road analysis.",
        }

    roi = image[int(height * 0.35) :, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    mean_saturation = float(np.mean(hsv[:, :, 1]))
    gray_std = float(np.std(gray))
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    dark_ratio = float(np.mean(gray < 35))
    bright_ratio = float(np.mean(gray > 225))

    checks = {
        "saturation_ok": mean_saturation < 115,
        "texture_ok": 14 <= gray_std <= 105,
        "edge_density_ok": 0.01 <= edge_density <= 0.30,
        "exposure_ok": (dark_ratio < 0.40 and bright_ratio < 0.35),
    }
    passed_checks = sum(1 for value in checks.values() if value)
    is_road_like = passed_checks >= 3

    reason = "Road-like image detected." if is_road_like else (
        "Image does not appear road-like enough. Try a clearer road surface photo."
    )
    return is_road_like, {
        "is_road_like": is_road_like,
        "reason": reason,
        "mean_saturation": round(mean_saturation, 3),
        "gray_std": round(gray_std, 3),
        "edge_density": round(edge_density, 4),
        "dark_ratio": round(dark_ratio, 4),
        "bright_ratio": round(bright_ratio, 4),
        **checks,
    }


def save_prediction_assets(
    source_name: str,
    original_bgr: np.ndarray,
    processed_image: np.ndarray,
    defect_mask: np.ndarray,
    overlay_rgb: np.ndarray,
) -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(source_name).stem or "scan"

    original_path = OUTPUT_DIR / f"{stem}_original.png"
    processed_path = OUTPUT_DIR / f"{stem}_processed.png"
    mask_path = OUTPUT_DIR / f"{stem}_mask.png"
    overlay_path = OUTPUT_DIR / f"{stem}_overlay.png"

    cv2.imwrite(str(original_path), original_bgr)
    cv2.imwrite(str(processed_path), processed_image)
    cv2.imwrite(str(mask_path), defect_mask)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    return {
        "original": str(original_path),
        "processed": str(processed_path),
        "mask": str(mask_path),
        "overlay": str(overlay_path),
    }


def generate_prediction_report(
    source_name: str,
    predicted_id: int,
    predicted_name: str,
    outputs: dict[str, object],
    probabilities: np.ndarray | None,
) -> dict[str, object]:
    return {
        "source": source_name,
        "predicted_class_id": int(predicted_id),
        "predicted_class_name": predicted_name,
        "damage_percentage": round(float(outputs["damage_percentage"]), 4),
        "severity": str(outputs["severity"]),
        "features": {
            key: round(float(value), 6) for key, value in outputs["feature_map"].items()
        },
        "probabilities": (
            {
                CLASS_NAMES[class_id]: round(float(probabilities[class_id]), 6)
                for class_id in sorted(CLASS_NAMES.keys())
            }
            if probabilities is not None
            else {}
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def get_condition_note(predicted_name: str, damage_percentage: float) -> str:
    if predicted_name == "Normal":
        return "Road surface looks acceptable. Continue routine monitoring."
    if predicted_name == "Crack":
        return "Surface crack detected. Preventive repair is recommended before expansion."
    if predicted_name == "Pothole":
        return "Pothole detected. Schedule patch repair to avoid vehicle damage."
    if damage_percentage > 12:
        return "High-impact road damage detected. Immediate maintenance attention is recommended."
    return "Severe pothole detected. Prioritize this section for repair."


def get_health_score(predicted_name: str, damage_percentage: float) -> int:
    base = {
        "Normal": 92,
        "Crack": 68,
        "Pothole": 45,
        "Severe Pothole": 22,
    }.get(predicted_name, 50)
    penalty = min(int(damage_percentage * 1.5), 25)
    return max(base - penalty, 5)


def run_prediction_on_image(image: np.ndarray, source_name: str) -> dict[str, object]:
    road_ok, road_info = assess_road_likelihood(image)
    if not road_ok:
        report_dict = {
            "source": source_name,
            "status": "rejected",
            "reason": road_info["reason"],
            "road_check": road_info,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        return {
            "status": "rejected",
            "reason": str(road_info["reason"]),
            "road_check": road_info,
            "report_json": json.dumps(report_dict, indent=2),
        }

    model = load_model(MODEL_PATH)
    outputs = process_image(image)
    predicted_id, predicted_name, probabilities = predict_class(model, outputs["feature_vector"])
    max_confidence = float(np.max(probabilities)) if probabilities is not None else None

    if max_confidence is not None and max_confidence < MIN_CONFIDENCE_THRESHOLD:
        report_dict = {
            "source": source_name,
            "status": "uncertain",
            "reason": (
                f"Prediction confidence is too low ({max_confidence:.3f}). "
                "Use a clearer road image or retrain with more balanced data."
            ),
            "confidence": round(max_confidence, 6),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        return {
            "status": "uncertain",
            "reason": report_dict["reason"],
            "confidence": max_confidence,
            "report_json": json.dumps(report_dict, indent=2),
        }

    overlay_rgb = create_overlay_image(image, outputs["defect_mask"])
    saved_paths = save_prediction_assets(
        source_name=source_name,
        original_bgr=image,
        processed_image=outputs["processed_image"],
        defect_mask=outputs["defect_mask"],
        overlay_rgb=overlay_rgb,
    )
    report_dict = generate_prediction_report(
        source_name=source_name,
        predicted_id=predicted_id,
        predicted_name=predicted_name,
        outputs=outputs,
        probabilities=probabilities,
    )
    health_score = get_health_score(predicted_name, float(outputs["damage_percentage"]))
    recommendation = get_condition_note(predicted_name, float(outputs["damage_percentage"]))

    history_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source_name,
        "predicted_class": predicted_name,
        "damage_percentage": round(float(outputs["damage_percentage"]), 2),
        "severity": str(outputs["severity"]),
        "health_score": health_score,
        "confidence": round(max_confidence, 4) if max_confidence is not None else None,
    }
    append_json_history(PREDICTION_HISTORY_PATH, history_entry)

    return {
        "status": "ok",
        "predicted_id": predicted_id,
        "predicted_name": predicted_name,
        "probabilities": probabilities,
        "outputs": outputs,
        "overlay_rgb": overlay_rgb,
        "saved_paths": saved_paths,
        "report_json": json.dumps(report_dict, indent=2),
        "health_score": health_score,
        "recommendation": recommendation,
        "confidence": max_confidence,
        "road_check": road_info,
    }


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #09131f;
            --bg-soft: rgba(11, 22, 36, 0.82);
            --bg-card: rgba(15, 28, 45, 0.92);
            --accent: #ffb703;
            --accent-2: #39d98a;
            --accent-3: #5dade2;
            --text-main: #f6fbff;
            --text-soft: #b8cbe0;
            --border: rgba(255,255,255,0.08);
            --danger: #ff6b6b;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255,183,3,0.18), transparent 28%),
                radial-gradient(circle at bottom right, rgba(57,217,138,0.14), transparent 24%),
                linear-gradient(160deg, #06111d 0%, #0d1a29 42%, #101e30 100%);
            color: var(--text-main);
        }
        .block-container {
            max-width: 1200px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4, label, p, li, span, div {
            color: var(--text-main);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(6,17,29,0.98) 0%, rgba(11,21,34,0.98) 100%);
            border-right: 1px solid var(--border);
        }
        .hero {
            border-radius: 26px;
            padding: 1.5rem 1.5rem 1.3rem 1.5rem;
            background:
                radial-gradient(circle at top right, rgba(255,183,3,0.16), transparent 26%),
                linear-gradient(135deg, rgba(19,40,68,0.96) 0%, rgba(10,22,38,0.95) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 20px 50px rgba(0,0,0,0.28);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.35rem;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: var(--text-soft);
            max-width: 760px;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1rem;
        }
        .badge {
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.07);
            color: var(--text-main);
            font-size: 0.9rem;
        }
        .surface-card {
            padding: 1rem;
            border-radius: 22px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            box-shadow: 0 14px 36px rgba(0,0,0,0.22);
            margin-bottom: 1rem;
        }
        .metric-card {
            border-radius: 20px;
            padding: 0.95rem 0.95rem 0.85rem 0.95rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
            border: 1px solid var(--border);
            min-height: 112px;
        }
        .metric-label {
            color: var(--text-soft);
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: var(--accent);
        }
        .metric-note {
            color: var(--text-soft);
            font-size: 0.84rem;
            margin-top: 0.2rem;
        }
        .status-card {
            padding: 1rem;
            border-radius: 18px;
            background: linear-gradient(160deg, rgba(57,217,138,0.10) 0%, rgba(0,0,0,0.08) 100%);
            border: 1px solid rgba(57,217,138,0.28);
        }
        .warning-card {
            padding: 1rem;
            border-radius: 18px;
            background: linear-gradient(160deg, rgba(255,107,107,0.10) 0%, rgba(0,0,0,0.08) 100%);
            border: 1px solid rgba(255,107,107,0.24);
        }
        .stButton button, .stDownloadButton button {
            width: 100%;
            border-radius: 14px;
            font-weight: 700;
            border: none;
            padding: 0.7rem 1rem;
            background: linear-gradient(135deg, #ffb703 0%, #ff8f00 100%);
            color: #08121e;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            flex-wrap: wrap;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.04);
            border-radius: 12px;
            padding: 0.55rem 0.9rem;
            color: var(--text-main);
        }
        .stTextArea textarea, .stTextInput input {
            background: rgba(255,255,255,0.03);
            color: var(--text-main);
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.035);
            border: 1px solid var(--border);
            padding: 0.8rem 0.9rem;
            border-radius: 16px;
        }
        .small-kicker {
            font-size: 0.82rem;
            letter-spacing: 0.14rem;
            text-transform: uppercase;
            color: var(--accent-3);
            margin-bottom: 0.5rem;
            font-weight: 700;
        }
        @media (max-width: 768px) {
            .hero-title {
                font-size: 1.75rem;
            }
            .block-container {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }
            .surface-card {
                padding: 0.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_tile(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(dataset_summary: dict[str, int]) -> None:
    total_images = sum(dataset_summary.values())
    st.markdown(
        f"""
        <div class="hero">
            <div class="small-kicker">Road Surface Intelligence</div>
            <div class="hero-title">Smart Pothole & Crack Detection</div>
            <div class="hero-subtitle">
                A mobile-friendly inspection dashboard for dataset training, live camera scans,
                batch analysis, and explainable SVM-based road defect prediction.
            </div>
            <div class="badge-row">
                <div class="badge">Dataset Images: {total_images}</div>
                <div class="badge">Model Ready: {"Yes" if MODEL_PATH.exists() else "No"}</div>
                <div class="badge">Phone Camera Compatible</div>
                <div class="badge">Desktop + Mobile Responsive</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(dataset_dir: Path) -> None:
    summary = get_dataset_summary(dataset_dir)
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.markdown("### Dashboard")
    st.write("Check dataset readiness and overall project health before training or scanning.")

    cols = st.columns(5)
    render_targets = [
        ("Normal", str(summary["normal"]), "Clean road samples"),
        ("Crack", str(summary["crack"]), "Minor defect samples"),
        ("Pothole", str(summary["pothole"]), "Medium damage samples"),
        ("Severe", str(summary["severe"]), "High damage samples"),
        ("Model", "Ready" if MODEL_PATH.exists() else "Missing", "Saved SVM file"),
    ]
    for col, tile in zip(cols, render_targets):
        with col:
            render_metric_tile(*tile)

    total_images = sum(summary.values())
    if total_images < 20:
        st.markdown(
            '<div class="warning-card">Dataset is still small. Add more balanced images for better SVM performance.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-card">Dataset looks usable. You can retrain the model and start testing predictions.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_training_panel(dataset_dir: Path) -> None:
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.markdown("### Train / Retrain Model")
    st.write("Extract features from the dataset and fit the SVM model with optional validation.")

    if st.button("Start Training", type="primary", key="train_button"):
        try:
            with st.spinner("Extracting road features and training the classifier..."):
                metrics = train_from_ui(dataset_dir)

            append_json_history(
                TRAINING_HISTORY_PATH,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "samples": metrics["samples"],
                    "class_counts": metrics["class_counts"],
                    "validated": metrics["validated"],
                    "accuracy": round(float(metrics.get("accuracy", 0.0)), 6) if metrics["validated"] else None,
                },
            )

            st.success("Training completed successfully.")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Samples Used", metrics["samples"])
            metric_cols[1].metric("Model Saved", "Yes")
            metric_cols[2].metric(
                "Validation Accuracy",
                f"{metrics['accuracy'] * 100:.2f}%" if metrics["validated"] else "N/A",
            )

            counts_df = pd.DataFrame(
                [
                    {"Class": class_name, "Images": metrics["class_counts"][class_name]}
                    for class_name in ["Normal", "Crack", "Pothole", "Severe Pothole"]
                ]
            )
            st.dataframe(counts_df, use_container_width=True, hide_index=True)

            if metrics["validated"]:
                st.text_area("Classification Report", metrics["report"], height=210)
                fig = plot_confusion_matrix(metrics["confusion_matrix"])
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("Validation split skipped because the dataset is still too small or sparse.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    history = load_json_history(TRAINING_HISTORY_PATH)
    if history:
        st.markdown("#### Training History")
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Training History CSV",
            history_df.to_csv(index=False).encode("utf-8"),
            file_name="training_history.csv",
            mime="text/csv",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_results(result: dict[str, object], source_name: str) -> None:
    if result.get("status") == "rejected":
        st.error(result["reason"])
        road_check = result.get("road_check", {})
        if road_check:
            road_df = pd.DataFrame(
                [{"Check": key, "Value": value} for key, value in road_check.items()]
            )
            st.markdown("#### Road Input Check")
            st.dataframe(road_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Rejection Report JSON",
            data=result["report_json"],
            file_name=f"{Path(source_name).stem}_rejection_report.json",
            mime="application/json",
        )
        return

    if result.get("status") == "uncertain":
        st.warning(result["reason"])
        if result.get("confidence") is not None:
            st.metric("Model Confidence", f"{float(result['confidence']) * 100:.2f}%")
        st.download_button(
            "Download Uncertain Prediction Report",
            data=result["report_json"],
            file_name=f"{Path(source_name).stem}_uncertain_report.json",
            mime="application/json",
        )
        return

    outputs = result["outputs"]
    probabilities = result["probabilities"]

    metrics = st.columns(4)
    metrics[0].metric("Predicted Class", str(result["predicted_name"]))
    metrics[1].metric("Damage %", f"{outputs['damage_percentage']:.2f}")
    metrics[2].metric("Severity", str(outputs["severity"]))
    metrics[3].metric("Health Score", str(result["health_score"]))

    lower_metrics = st.columns(2)
    lower_metrics[0].metric(
        "Model Confidence",
        f"{float(result['confidence']) * 100:.2f}%" if result.get("confidence") is not None else "N/A",
    )
    lower_metrics[1].metric(
        "Road Check",
        "Passed" if result.get("road_check", {}).get("is_road_like") else "Failed",
    )

    st.markdown(
        f"""
        <div class="status-card">
            <strong>Inspection Note:</strong> {result["recommendation"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    image_tab1, image_tab2, image_tab3, image_tab4 = st.tabs(
        ["Overlay View", "Original", "Processed", "Defect Mask"]
    )
    with image_tab1:
        st.image(result["overlay_rgb"], caption="Defect overlay on road surface", use_container_width=True)
    with image_tab2:
        original_rgb = cv2.cvtColor(load_image(result["saved_paths"]["original"]), cv2.COLOR_BGR2RGB)
        st.image(original_rgb, caption="Original image", use_container_width=True)
    with image_tab3:
        st.image(result["saved_paths"]["processed"], caption="Processed grayscale image", use_container_width=True)
    with image_tab4:
        st.image(result["saved_paths"]["mask"], caption="Segmented defect mask", use_container_width=True)

    detail_col1, detail_col2 = st.columns([1.15, 0.85], gap="large")
    with detail_col1:
        feature_df = pd.DataFrame(
            [
                {"Feature": key.replace("_", " ").title(), "Value": round(float(value), 4)}
                for key, value in outputs["feature_map"].items()
            ]
        )
        st.markdown("#### Extracted Features")
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

    with detail_col2:
        if probabilities is not None:
            st.markdown("#### Class Probabilities")
            prob_data = pd.DataFrame(
                {
                    "Class": [CLASS_NAMES[class_id] for class_id in sorted(CLASS_NAMES.keys())],
                    "Probability": [float(probabilities[class_id]) for class_id in sorted(CLASS_NAMES.keys())],
                }
            )
            st.bar_chart(prob_data.set_index("Class"))

    path_df = pd.DataFrame(
        [{"Asset": key.title(), "Path": value} for key, value in result["saved_paths"].items()]
    )
    st.markdown("#### Saved Files")
    st.dataframe(path_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Prediction Report JSON",
        data=result["report_json"],
        file_name=f"{Path(source_name).stem}_prediction_report.json",
        mime="application/json",
    )


def render_scan_center() -> None:
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.markdown("### Scan Center")
    st.write("Run a single scan, batch inspection, or live camera capture from the same dashboard.")

    if not MODEL_PATH.exists():
        st.warning("Train the model first. Prediction tools unlock after the model file is available.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    tab_single, tab_batch, tab_camera = st.tabs(["Single Upload", "Batch Scan", "Camera Scan"])

    with tab_single:
        uploaded_file = st.file_uploader(
            "Upload a road image",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="single_upload",
        )
        if uploaded_file is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                st.error("Unable to read the uploaded image.")
            else:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Preview", use_container_width=True)
                if st.button("Run Single Scan", key="single_scan_button"):
                    try:
                        with st.spinner("Scanning road surface..."):
                            result = run_prediction_on_image(image, uploaded_file.name)
                        render_prediction_results(result, uploaded_file.name)
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

    with tab_batch:
        batch_files = st.file_uploader(
            "Upload multiple road images",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="batch_upload",
        )
        if batch_files and st.button("Run Batch Scan", key="batch_scan_button"):
            rows: list[dict[str, object]] = []
            progress = st.progress(0)
            try:
                for index, batch_file in enumerate(batch_files, start=1):
                    image = cv2.imdecode(np.frombuffer(batch_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        rows.append(
                            {
                                "Filename": batch_file.name,
                                "Predicted Class": "Unreadable",
                                "Damage %": None,
                                "Severity": "N/A",
                                "Health Score": None,
                            }
                        )
                    else:
                        result = run_prediction_on_image(image, batch_file.name)
                        rows.append(
                            {
                                "Filename": batch_file.name,
                                "Predicted Class": result.get("predicted_name", result.get("status", "Unknown")).title(),
                                "Damage %": (
                                    round(float(result["outputs"]["damage_percentage"]), 2)
                                    if result.get("status") == "ok"
                                    else None
                                ),
                                "Severity": (
                                    str(result["outputs"]["severity"])
                                    if result.get("status") == "ok"
                                    else result.get("status", "N/A").title()
                                ),
                                "Health Score": result.get("health_score"),
                            }
                        )
                    progress.progress(index / len(batch_files))

                batch_df = pd.DataFrame(rows)
                st.dataframe(batch_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Batch CSV",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_prediction_results.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")

    with tab_camera:
        st.write("Open your phone or laptop camera and capture a road image directly.")
        camera_image = st.camera_input("Camera Capture")
        if camera_image is not None:
            image = cv2.imdecode(np.frombuffer(camera_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Captured frame", use_container_width=True)
                if st.button("Run Camera Scan", key="camera_scan_button"):
                    try:
                        result = run_prediction_on_image(image, "camera_capture.jpg")
                        render_prediction_results(result, "camera_capture.jpg")
                    except Exception as exc:
                        st.error(f"Camera prediction failed: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_history_panel() -> None:
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.markdown("### Inspection History")
    history = load_json_history(PREDICTION_HISTORY_PATH)
    if not history:
        st.info("No prediction history yet. Run a scan and the results will appear here.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    history_df = pd.DataFrame(history).sort_values("timestamp", ascending=False)
    top_cols = st.columns(3)
    top_cols[0].metric("Total Scans", len(history_df))
    top_cols[1].metric(
        "Average Damage %",
        f"{history_df['damage_percentage'].astype(float).mean():.2f}",
    )
    top_cols[2].metric(
        "Latest Prediction",
        str(history_df.iloc[0]["predicted_class"]),
    )

    st.dataframe(history_df, use_container_width=True, hide_index=True)

    summary = history_df["predicted_class"].value_counts().reset_index()
    summary.columns = ["Class", "Count"]
    st.bar_chart(summary.set_index("Class"))

    st.download_button(
        "Download Prediction History CSV",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar(dataset_dir: Path) -> None:
    st.sidebar.markdown("## Smart Road Inspector")
    st.sidebar.write("Train, scan, and review road defects from one place.")
    st.sidebar.info(f"Dataset: `{dataset_dir}`")
    st.sidebar.info(f"Model: `{MODEL_PATH}`")
    st.sidebar.info(f"Outputs: `{OUTPUT_DIR}`")
    st.sidebar.markdown("### Best Workflow")
    st.sidebar.write("1. Add balanced dataset images")
    st.sidebar.write("2. Retrain the SVM model")
    st.sidebar.write("3. Use single, batch, or camera scan")
    st.sidebar.write("4. Download reports or review history")
    st.sidebar.markdown("### Mobile Tip")
    st.sidebar.write("When deployed, open the site on your phone and use the Camera Scan tab.")


def main() -> None:
    st.set_page_config(
        page_title="Smart Pothole Detection",
        page_icon="road",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    dataset_dir = ROOT_DIR / "dataset"
    render_sidebar(dataset_dir)
    dataset_summary = get_dataset_summary(dataset_dir)
    render_hero(dataset_summary)
    render_overview(dataset_dir)

    first_col, second_col = st.columns([1.02, 0.98], gap="large")
    with first_col:
        render_training_panel(dataset_dir)
    with second_col:
        render_scan_center()

    render_history_panel()


if __name__ == "__main__":
    main()
