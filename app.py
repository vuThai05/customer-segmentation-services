from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised only when Plotly is absent.
    go = None


APP_TITLE = "Customer Segmentation Services"
COLOR_PALETTE = [
    "#0F4C5C",
    "#E36414",
    "#6A994E",
    "#5E548E",
    "#BC4749",
    "#277DA1",
    "#F4A261",
    "#8D99AE",
    "#C08497",
    "#7A5C61",
]


@dataclass
class PreparedNumericFeatures:
    numeric_df: pd.DataFrame
    scaled_array: np.ndarray
    numeric_columns: list[str]
    missing_values_fixed: int
    imputer: SimpleImputer | None
    scaler: StandardScaler | None


@dataclass
class SegmentationResult:
    labels: np.ndarray
    model: KMeans
    silhouette_value: float | None
    wcss_by_group_count: pd.DataFrame | None
    pca_model: PCA | None
    pca_coordinates: np.ndarray | None
    representative_coordinates: np.ndarray | None


def prepare_numeric_features(df: pd.DataFrame) -> PreparedNumericFeatures:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_columns = numeric_df.columns.tolist()
    missing_values_fixed = int(numeric_df.isna().sum().sum())

    if not numeric_columns:
        return PreparedNumericFeatures(
            numeric_df=pd.DataFrame(index=df.index),
            scaled_array=np.empty((len(df.index), 0)),
            numeric_columns=[],
            missing_values_fixed=0,
            imputer=None,
            scaler=None,
        )

    imputer = SimpleImputer(strategy="median")
    cleaned_array = imputer.fit_transform(numeric_df)
    cleaned_numeric_df = pd.DataFrame(
        cleaned_array,
        columns=numeric_columns,
        index=df.index,
    )

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(cleaned_numeric_df)

    return PreparedNumericFeatures(
        numeric_df=cleaned_numeric_df,
        scaled_array=scaled_array,
        numeric_columns=numeric_columns,
        missing_values_fixed=missing_values_fixed,
        imputer=imputer,
        scaler=scaler,
    )


def run_segmentation(X_scaled: np.ndarray, n_groups: int) -> SegmentationResult:
    if X_scaled.ndim != 2:
        raise ValueError("Expected a two-dimensional feature matrix.")

    n_samples, n_features = X_scaled.shape
    if n_samples < 2:
        raise ValueError("At least two rows are required to form groups.")
    if n_groups < 2:
        raise ValueError("At least two groups are required.")
    if n_groups > n_samples:
        raise ValueError("Number of groups cannot exceed the number of rows.")

    model = KMeans(n_clusters=n_groups, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)

    silhouette_value = None
    unique_label_count = len(np.unique(labels))
    if 1 < unique_label_count < n_samples:
        silhouette_value = float(silhouette_score(X_scaled, labels))

    max_group_count = min(10, n_samples)
    elbow_rows: list[dict[str, float | int]] = []
    for group_count in range(2, max_group_count + 1):
        elbow_model = KMeans(n_clusters=group_count, n_init=10, random_state=42)
        elbow_model.fit(X_scaled)
        elbow_rows.append(
            {
                "Number of Groups": group_count,
                "WCSS": float(elbow_model.inertia_),
            }
        )
    wcss_by_group_count = pd.DataFrame(elbow_rows) if elbow_rows else None

    pca_model = None
    pca_coordinates = None
    representative_coordinates = None
    if n_features >= 2:
        pca_model = PCA(n_components=2)
        pca_coordinates = pca_model.fit_transform(X_scaled)
        representative_coordinates = pca_model.transform(model.cluster_centers_)

    return SegmentationResult(
        labels=labels,
        model=model,
        silhouette_value=silhouette_value,
        wcss_by_group_count=wcss_by_group_count,
        pca_model=pca_model,
        pca_coordinates=pca_coordinates,
        representative_coordinates=representative_coordinates,
    )


def build_group_name_map(labels: np.ndarray) -> dict[int, str]:
    unique_labels = sorted(int(label) for label in np.unique(labels))
    return {
        label: f"Group {position}"
        for position, label in enumerate(unique_labels, start=1)
    }


def build_color_map(labels: np.ndarray) -> dict[str, str]:
    group_name_map = build_group_name_map(labels)
    return {
        group_name_map[label]: COLOR_PALETTE[index % len(COLOR_PALETTE)]
        for index, label in enumerate(sorted(group_name_map))
    }


def build_export_df(df_original: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    export_df = df_original.copy()
    group_name_map = build_group_name_map(labels)
    export_df["Cluster_Label"] = [group_name_map[int(label)] for label in labels]
    return export_df


def describe_silhouette(score: float | None) -> tuple[str, str]:
    if score is None:
        return "Not enough variation to score", "#6C757D"
    if score > 0.5:
        return "Excellent Separation", "#2E7D32"
    if score >= 0.2:
        return "Fair Separation", "#F9A825"
    return "Weak Clusters", "#C62828"


def figure_support_available() -> bool:
    return go is not None


def build_pca_figure(
    pca_coordinates: np.ndarray,
    labels: np.ndarray,
    representative_coordinates: np.ndarray,
    color_map: dict[str, str],
) -> "go.Figure":
    if go is None:  # pragma: no cover - guarded in the UI.
        raise RuntimeError("Plotly is required to build figures.")

    group_name_map = build_group_name_map(labels)
    group_names = np.array([group_name_map[int(label)] for label in labels], dtype=object)
    group_order = list(color_map.keys())

    fig = go.Figure()
    for group_name in group_order:
        mask = group_names == group_name
        fig.add_trace(
            go.Scatter(
                x=pca_coordinates[mask, 0],
                y=pca_coordinates[mask, 1],
                mode="markers",
                name=group_name,
                marker={
                    "size": 10,
                    "opacity": 0.8,
                    "color": color_map[group_name],
                },
                hovertemplate="Group=%{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
                text=[group_name] * int(mask.sum()),
            )
        )

    representative_x = []
    representative_y = []
    representative_names = []
    representative_colors = []
    for label, group_name in group_name_map.items():
        representative_x.append(representative_coordinates[label, 0])
        representative_y.append(representative_coordinates[label, 1])
        representative_names.append(f"{group_name} Representative")
        representative_colors.append(color_map[group_name])

    fig.add_trace(
        go.Scatter(
            x=representative_x,
            y=representative_y,
            mode="markers+text",
            name="Group Representatives",
            marker={
                "symbol": "x",
                "size": 18,
                "color": representative_colors,
                "line": {"width": 2, "color": "#111111"},
            },
            text=representative_names,
            textposition="top center",
            hovertemplate="%{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Master Insight Map",
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend_title="Groups",
        template="plotly_white",
        height=560,
    )
    return fig


def build_variable_explorer_figure(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    labels: np.ndarray,
    color_map: dict[str, str],
) -> "go.Figure":
    if go is None:  # pragma: no cover - guarded in the UI.
        raise RuntimeError("Plotly is required to build figures.")

    group_name_map = build_group_name_map(labels)
    group_names = np.array([group_name_map[int(label)] for label in labels], dtype=object)

    fig = go.Figure()
    for group_name, color in color_map.items():
        mask = group_names == group_name
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, x_axis],
                y=df.loc[mask, y_axis],
                mode="markers",
                name=group_name,
                marker={"size": 10, "opacity": 0.8, "color": color},
                hovertemplate=(
                    f"{x_axis}=%{{x:.2f}}<br>{y_axis}=%{{y:.2f}}"
                    "<br>Group=%{text}<extra></extra>"
                ),
                text=[group_name] * int(mask.sum()),
            )
        )

    fig.update_layout(
        title="Variable Explorer",
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        legend_title="Groups",
        template="plotly_white",
        height=520,
    )
    return fig


def build_wcss_figure(wcss_by_group_count: pd.DataFrame) -> "go.Figure":
    if go is None:  # pragma: no cover - guarded in the UI.
        raise RuntimeError("Plotly is required to build figures.")

    fig = go.Figure(
        data=[
            go.Scatter(
                x=wcss_by_group_count["Number of Groups"],
                y=wcss_by_group_count["WCSS"],
                mode="lines+markers",
                line={"color": COLOR_PALETTE[0], "width": 3},
                marker={"size": 9},
            )
        ]
    )
    fig.update_layout(
        title="Elbow View of Group Quality",
        xaxis_title="Number of Groups",
        yaxis_title="WCSS",
        template="plotly_white",
        height=380,
    )
    return fig


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    buffer.write(df.to_csv(index=False).encode("utf-8"))
    return buffer.getvalue()


def show_plotly_missing_message() -> None:
    st.warning(
        "Interactive charts need Plotly. Install the packages in requirements.txt "
        "before launching the dashboard locally."
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Upload a CSV to automatically clean numeric fields, form customer groups, "
        "and explore the result in business-friendly visuals."
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Drag and drop any customer or operational CSV to begin.",
    )

    with st.sidebar:
        st.header("Controls")
        show_elbow = st.toggle(
            "Show elbow method",
            value=False,
            help="This compares different group counts so you can see when the model stops improving.",
        )

    st.header("1. Data Onboarding")
    if uploaded_file is None:
        st.info("Upload a CSV file to unlock the dashboard.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:  # pragma: no cover - depends on file input behavior.
        st.error(f"We could not read that CSV file: {exc}")
        return

    if df.empty:
        st.error("This CSV has no data rows. Upload a file with at least two rows.")
        return

    prepared = prepare_numeric_features(df)

    preview_col, health_col = st.columns([2.3, 1.2])
    with preview_col:
        st.subheader("Data Preview")
        st.dataframe(df.head(100), use_container_width=True)

    with health_col:
        st.subheader("Health Report")
        st.metric("Rows", len(df))
        st.metric("Numeric Features", len(prepared.numeric_columns))
        st.metric("Auto-fixed Missing Values", prepared.missing_values_fixed)
        if prepared.numeric_columns:
            st.write("Numeric columns in scope:")
            st.write(", ".join(prepared.numeric_columns))
        else:
            st.write("Numeric columns in scope: none")

    if not prepared.numeric_columns:
        st.error("This file has no numeric columns, so the grouping engine cannot run.")
        return

    row_count = len(df.index)
    if row_count < 2:
        st.error("At least two rows are required to form groups.")
        return

    max_groups = min(10, row_count)
    with st.sidebar:
        number_of_groups = st.slider(
            "Number of Groups",
            min_value=2,
            max_value=max_groups,
            value=min(4, max_groups),
            help="Choose how many customer groups the model should create.",
        )

    segmentation = run_segmentation(prepared.scaled_array, number_of_groups)
    export_df = build_export_df(df, segmentation.labels)
    color_map = build_color_map(segmentation.labels)
    silhouette_label, silhouette_color = describe_silhouette(segmentation.silhouette_value)

    st.header("2. The Master Insight")
    metric_col, explainer_col = st.columns([1, 2])
    with metric_col:
        metric_value = (
            f"{segmentation.silhouette_value:.3f}"
            if segmentation.silhouette_value is not None
            else "N/A"
        )
        st.metric("Separation Score", metric_value)
        st.markdown(
            (
                f"<div style='color:{silhouette_color}; font-weight:600;'>"
                f"{silhouette_label}</div>"
            ),
            unsafe_allow_html=True,
        )
    with explainer_col:
        st.info(
            "PCA creates a simplified map of complex customer data so you can see how "
            "groups separate at a glance. The separation score tells you how clearly "
            "those groups differ from each other."
        )

    if len(prepared.numeric_columns) < 2:
        st.info(
            "The Master Insight map needs at least two numeric columns. Grouping still runs, "
            "but the 2D PCA view is disabled for this file."
        )
    elif not figure_support_available():
        show_plotly_missing_message()
    else:
        pca_figure = build_pca_figure(
            pca_coordinates=segmentation.pca_coordinates,
            labels=segmentation.labels,
            representative_coordinates=segmentation.representative_coordinates,
            color_map=color_map,
        )
        st.plotly_chart(pca_figure, use_container_width=True)

    if show_elbow:
        if segmentation.wcss_by_group_count is not None and figure_support_available():
            st.plotly_chart(
                build_wcss_figure(segmentation.wcss_by_group_count),
                use_container_width=True,
            )
        elif not figure_support_available():
            show_plotly_missing_message()

    st.header("3. Variable Explorer")
    st.info(
        "This view keeps the original units, so business users can inspect how groups "
        "look in familiar measures such as income, spend, or age."
    )
    if len(prepared.numeric_columns) < 2:
        st.info(
            "The Variable Explorer needs at least two numeric columns to compare one measure "
            "against another."
        )
    elif not figure_support_available():
        show_plotly_missing_message()
    else:
        x_col, y_col = st.columns(2)
        with x_col:
            x_axis = st.selectbox("X-axis", prepared.numeric_columns, index=0)
        with y_col:
            default_y_index = 1 if len(prepared.numeric_columns) > 1 else 0
            y_axis = st.selectbox("Y-axis", prepared.numeric_columns, index=default_y_index)

        variable_figure = build_variable_explorer_figure(
            df=prepared.numeric_df,
            x_axis=x_axis,
            y_axis=y_axis,
            labels=segmentation.labels,
            color_map=color_map,
        )
        st.plotly_chart(variable_figure, use_container_width=True)

    st.header("4. Export & Action")
    st.write(
        "Download the original records enriched with the model's group assignment so the "
        "result can be shared or used in downstream tools."
    )
    st.download_button(
        "Download grouped CSV",
        data=to_csv_bytes(export_df),
        file_name="customer_segmentation_output.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
