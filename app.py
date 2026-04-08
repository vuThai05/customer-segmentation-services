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
except ImportError:  # pragma: no cover - only when Plotly is absent.
    go = None


APP_TITLE = "Dịch vụ Phân khúc Khách hàng"
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
PROFILE_LEVEL_COLORS = [
    "rgb(255,255,255)",
    "rgb(255,160,122)",
    "rgb(250,128,114)",
    "rgb(220,20,60)",
]

# ── ID-column detection ────────────────────────────────────────────────────────

_ID_KEYWORDS = (
    "id", "invoice", "order", "transaction",
    "receipt", "serial", "reference", "barcode", "sku",
)


@dataclass
class PreparedNumericFeatures:
    numeric_df: pd.DataFrame
    numeric_columns: list[str]
    missing_values_fixed: int
    imputation_details: pd.DataFrame
    # NOTE: scaled_array removed — scaling is done after ID-column removal


@dataclass
class SegmentationResult:
    labels: np.ndarray
    silhouette_value: float | None
    wcss_by_group_count: pd.DataFrame | None
    pca_coordinates: np.ndarray | None
    representative_coordinates: np.ndarray | None


# ── Data loading & preparation ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))


def detect_id_like_columns(numeric_df: pd.DataFrame) -> list[str]:
    row_count = len(numeric_df)
    dropped: list[str] = []

    for column in numeric_df.columns:
        col_lower = str(column).strip().lower()
        has_keyword = any(kw in col_lower for kw in _ID_KEYWORDS) or col_lower.endswith("no")
        if not has_keyword:
            continue

        series = numeric_df[column]
        unique_ratio = series.nunique(dropna=True) / row_count if row_count else 0.0
        if unique_ratio >= 0.5 or pd.api.types.is_integer_dtype(series):
            dropped.append(column)

    return dropped


def auto_drop_id_columns(numeric_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    id_cols = detect_id_like_columns(numeric_df)
    keep = [c for c in numeric_df.columns if c not in set(id_cols)]
    if not keep:
        return numeric_df.copy(), []
    return numeric_df[keep].copy(), id_cols


@st.cache_data(show_spinner=False)
def prepare_numeric_features(df: pd.DataFrame) -> PreparedNumericFeatures:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_columns = numeric_df.columns.tolist()

    if not numeric_columns:
        return PreparedNumericFeatures(
            numeric_df=pd.DataFrame(index=df.index),
            numeric_columns=[],
            missing_values_fixed=0,
            imputation_details=pd.DataFrame(
                columns=["Cột", "Số giá trị thiếu đã sửa", "Giá trị thay thế (median)"]
            ),
        )

    missing_by_col = numeric_df.isna().sum()
    missing_total = int(missing_by_col.sum())

    imputer = SimpleImputer(strategy="median")
    cleaned_array = imputer.fit_transform(numeric_df)
    cleaned_df = pd.DataFrame(cleaned_array, columns=numeric_columns, index=df.index)

    median_by_col = pd.Series(imputer.statistics_, index=numeric_columns)
    detail_rows = [
        {
            "Cột": col,
            "Số giá trị thiếu đã sửa": int(missing_by_col[col]),
            "Giá trị thay thế (median)": float(median_by_col[col]),
        }
        for col in numeric_columns
        if missing_by_col[col] > 0
    ]

    return PreparedNumericFeatures(
        numeric_df=cleaned_df,
        numeric_columns=numeric_columns,
        missing_values_fixed=missing_total,
        imputation_details=pd.DataFrame(detail_rows),
    )


@st.cache_data(show_spinner=False)
def scale_numeric_frame(numeric_df: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(numeric_df)


# ── Segmentation ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_segmentation(X_scaled: np.ndarray, n_groups: int, compute_elbow: bool) -> SegmentationResult:
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

    silhouette_value: float | None = None
    unique_count = len(np.unique(labels))
    if 1 < unique_count < n_samples:
        silhouette_value = float(
            silhouette_score(
                X_scaled, labels,
                sample_size=min(5000, n_samples),
                random_state=42,
            )
        )

    wcss_by_group_count: pd.DataFrame | None = None
    if compute_elbow:
        max_k = min(10, n_samples)
        rows = [
            {"Số lượng nhóm": k, "WCSS": float(KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled).inertia_)}
            for k in range(2, max_k + 1)
        ]
        wcss_by_group_count = pd.DataFrame(rows) if rows else None

    pca_coordinates = representative_coordinates = None
    if n_features >= 2:
        pca_model = PCA(n_components=2)
        pca_coordinates = pca_model.fit_transform(X_scaled)
        representative_coordinates = pca_model.transform(model.cluster_centers_)

    return SegmentationResult(
        labels=labels,
        silhouette_value=silhouette_value,
        wcss_by_group_count=wcss_by_group_count,
        pca_coordinates=pca_coordinates,
        representative_coordinates=representative_coordinates,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_group_name_map(labels: np.ndarray) -> dict[int, str]:
    unique_labels = sorted(int(lbl) for lbl in np.unique(labels))
    return {lbl: f"Nhóm {pos}" for pos, lbl in enumerate(unique_labels, start=1)}


def build_color_map(group_name_map: dict[int, str]) -> dict[str, str]:
    """Accepts a pre-built group_name_map to avoid redundant recomputation."""
    return {
        name: COLOR_PALETTE[i % len(COLOR_PALETTE)]
        for i, name in enumerate(sorted(group_name_map.values()))
    }


def build_export_df(df_original: pd.DataFrame, labels: np.ndarray, group_name_map: dict[int, str]) -> pd.DataFrame:
    export_df = df_original.copy()
    export_df["Cluster_Label"] = [group_name_map[int(lbl)] for lbl in labels]
    return export_df


def build_group_profile_table(numeric_df: pd.DataFrame, labels: np.ndarray):
    profile_df = numeric_df.copy()
    profile_df["Cluster"] = labels.astype(int)
    summary_table = profile_df.groupby("Cluster").mean().round(3)

    def style_four_levels(column: pd.Series) -> list[str]:
        if column.dropna().empty:
            return [f"background-color: {PROFILE_LEVEL_COLORS[0]}; color: #000000;"] * len(column)

        q1, q2, q3 = column.quantile([0.25, 0.5, 0.75]).tolist()
        styles: list[str] = []
        for value in column:
            if pd.isna(value) or value <= q1:
                color = PROFILE_LEVEL_COLORS[0]
            elif value <= q2:
                color = PROFILE_LEVEL_COLORS[1]
            elif value <= q3:
                color = PROFILE_LEVEL_COLORS[2]
            else:
                color = PROFILE_LEVEL_COLORS[3]
            styles.append(f"background-color: {color}; color: #000000;")
        return styles

    def format_max_3_decimals(value: float) -> str:
        return "" if pd.isna(value) else f"{value:.3f}".rstrip("0").rstrip(".")

    styled = (
        summary_table.style.apply(style_four_levels, axis=0)
        .format(format_max_3_decimals)
        .set_properties(**{"color": "#000000", "font-size": "1.02rem", "font-weight": "600"})
        .set_table_styles(
            [{"selector": "th", "props": [("color", "#000000"), ("font-size", "1.02rem"), ("font-weight", "700")]}]
        )
    )
    return summary_table, styled


def describe_silhouette(score: float | None) -> tuple[str, str]:
    if score is None:
        return "Không đủ khác biệt để chấm điểm", "#6C757D"
    if score > 0.5:
        return "Tách biệt rất tốt", "#2E7D32"
    if score >= 0.2:
        return "Tách biệt trung bình", "#F9A825"
    return "Cụm yếu", "#C62828"


# ── Rendering helpers ──────────────────────────────────────────────────────────

def render_explainer(lines: str | list[str]) -> None:
    if isinstance(lines, str):
        normalized = [lines.strip()]
    else:
        normalized = [line.strip() for line in lines if line.strip()]
    if not normalized:
        return

    inner = "".join(f"<div style='margin:0.10rem 0;'>✨ {line}</div>" for line in normalized)
    st.markdown(
        f"<div style='font-size:1.45rem; color:#000000; line-height:1.5; "
        f"font-weight:600; text-align:left; margin:0.25rem 0 0.5rem 0;'>{inner}</div>",
        unsafe_allow_html=True,
    )


def render_section_divider() -> None:
    st.markdown(
        "<hr style='border:0; border-top:1px solid #4B5563; margin:1rem 0 1.2rem 0;'>",
        unsafe_allow_html=True,
    )


def show_plotly_missing_message() -> None:
    st.warning(
        "Biểu đồ tương tác cần Plotly. Hãy cài đủ thư viện trong requirements.txt "
        "trước khi chạy dashboard ở máy local."
    )


# ── Plotly figures ─────────────────────────────────────────────────────────────

def build_pca_figure(
    pca_coordinates: np.ndarray,
    labels: np.ndarray,
    representative_coordinates: np.ndarray,
    color_map: dict[str, str],
    group_name_map: dict[int, str],
) -> "go.Figure":
    if go is None:  # pragma: no cover
        raise RuntimeError("Plotly is required to build figures.")

    group_names = np.array([group_name_map[int(lbl)] for lbl in labels], dtype=object)

    fig = go.Figure()
    for group_name in color_map:
        mask = group_names == group_name
        fig.add_trace(
            go.Scatter(
                x=pca_coordinates[mask, 0],
                y=pca_coordinates[mask, 1],
                mode="markers",
                name=group_name,
                marker={"size": 10, "opacity": 0.8, "color": color_map[group_name]},
                hovertemplate="Nhóm=%{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
                text=[group_name] * int(mask.sum()),
            )
        )

    rep_x, rep_y, rep_names, rep_colors = [], [], [], []
    for lbl, name in group_name_map.items():
        rep_x.append(representative_coordinates[lbl, 0])
        rep_y.append(representative_coordinates[lbl, 1])
        rep_names.append(f"{name} đại diện")
        rep_colors.append(color_map[name])

    fig.add_trace(
        go.Scatter(
            x=rep_x, y=rep_y,
            mode="markers+text",
            name="Đại diện nhóm",
            marker={"symbol": "x", "size": 18, "color": rep_colors, "line": {"width": 2, "color": "#111111"}},
            text=rep_names,
            textposition="top center",
            hovertemplate="%{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Bản đồ Insight tổng quan",
        xaxis_title="PC1", yaxis_title="PC2",
        legend_title="Nhóm", template="plotly_white", height=560,
    )
    return fig


def build_variable_explorer_figure(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    labels: np.ndarray,
    color_map: dict[str, str],
    group_name_map: dict[int, str],
) -> "go.Figure":
    if go is None:  # pragma: no cover
        raise RuntimeError("Plotly is required to build figures.")

    group_names = np.array([group_name_map[int(lbl)] for lbl in labels], dtype=object)

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
                    "<br>Nhóm=%{text}<extra></extra>"
                ),
                text=[group_name] * int(mask.sum()),
            )
        )

    fig.update_layout(
        title="Khám phá biến",
        xaxis_title=x_axis, yaxis_title=y_axis,
        legend_title="Nhóm", template="plotly_white", height=520,
    )
    return fig


def build_wcss_figure(wcss_by_group_count: pd.DataFrame) -> "go.Figure":
    if go is None:  # pragma: no cover
        raise RuntimeError("Plotly is required to build figures.")

    fig = go.Figure(
        data=[
            go.Scatter(
                x=wcss_by_group_count["Số lượng nhóm"],
                y=wcss_by_group_count["WCSS"],
                mode="lines+markers",
                line={"color": COLOR_PALETTE[0], "width": 3},
                marker={"size": 9},
            )
        ]
    )
    fig.update_layout(
        title="Đường Elbow đánh giá chất lượng nhóm",
        xaxis_title="Số lượng nhóm", yaxis_title="WCSS",
        template="plotly_white", height=380,
    )
    return fig


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ── Main app ───────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    render_explainer(
        "Tải CSV lên để tự động làm sạch dữ liệu số, tạo nhóm khách hàng, và khám phá kết quả bằng biểu đồ dễ hiểu cho nghiệp vụ."
    )

    st.header("1. Nạp dữ liệu")
    uploaded_file = st.file_uploader(
        "Tải lên tệp CSV",
        type=["csv"],
        help="Kéo thả bất kỳ tệp CSV khách hàng hoặc vận hành để bắt đầu.",
    )

    if uploaded_file is None:
        st.info("Hãy tải lên tệp CSV để bắt đầu phân tích.")
        return

    try:
        df = load_csv_from_bytes(uploaded_file.getvalue())
    except Exception as exc:  # pragma: no cover
        st.error(f"Không thể đọc tệp CSV này: {exc}")
        return

    if df.empty:
        st.error("CSV không có dòng dữ liệu. Hãy tải tệp có ít nhất hai dòng.")
        return

    prepared = prepare_numeric_features(df)
    cluster_numeric_df, auto_dropped_id_cols = auto_drop_id_columns(prepared.numeric_df)
    cluster_numeric_columns = cluster_numeric_df.columns.tolist()

    preview_col, health_col = st.columns([2.3, 1.2])
    with preview_col:
        st.subheader("Xem trước dữ liệu")
        st.dataframe(df.head(100), use_container_width=True)

    with health_col:
        st.subheader("Báo cáo dữ liệu")
        st.metric("Số dòng", len(df))
        st.metric("Số cột số", len(prepared.numeric_columns))
        st.metric("Tổng giá trị thiếu đã tự sửa", prepared.missing_values_fixed)
        if prepared.numeric_columns:
            st.write("Các cột số được dùng:")
            st.write(", ".join(prepared.numeric_columns))
        else:
            st.write("Các cột số được dùng: không có")

    if auto_dropped_id_cols:
        st.info("Tự động bỏ cột ID khỏi phân cụm: " + ", ".join(auto_dropped_id_cols))

    if prepared.missing_values_fixed > 0 and not prepared.imputation_details.empty:
        st.write("Chi tiết các giá trị thiếu đã tự sửa:")
        st.dataframe(prepared.imputation_details, hide_index=True, use_container_width=True)
        render_explainer([
            "Mỗi giá trị thiếu sẽ được thay bằng trung vị của chính cột đó.",
            "Công thức điền thiếu: x_thiếu = median(cột).",
        ])
    else:
        render_explainer(["Không có giá trị thiếu cần tự sửa trong các cột số."])

    if not prepared.numeric_columns:
        st.error("Tệp này không có cột số nên không thể chạy phân nhóm.")
        return

    if len(df.index) < 2:
        st.error("Cần ít nhất hai dòng dữ liệu để phân nhóm.")
        return

    st.subheader("Thiết lập mô hình")
    row_count = len(df.index)
    max_groups = min(10, row_count)
    control_left, control_right = st.columns([1.5, 1.0])
    with control_left:
        number_of_groups = st.slider(
            "Số lượng nhóm", min_value=2, max_value=max_groups,
            value=min(4, max_groups),
            help="Chọn số nhóm khách hàng mà mô hình sẽ tạo.",
        )
    with control_right:
        show_elbow = st.toggle("Hiện biểu đồ Elbow", value=False, help="Bật để hiển thị biểu đồ Elbow.")

    scaled_for_clustering = scale_numeric_frame(cluster_numeric_df)
    segmentation = run_segmentation(scaled_for_clustering, number_of_groups, compute_elbow=show_elbow)

    # Compute once, reuse everywhere
    group_name_map = build_group_name_map(segmentation.labels)
    color_map = build_color_map(group_name_map)
    has_plotly = go is not None

    export_df = build_export_df(df, segmentation.labels, group_name_map)
    _, cluster_profile_styler = build_group_profile_table(cluster_numeric_df, segmentation.labels)
    silhouette_label, silhouette_color = describe_silhouette(segmentation.silhouette_value)

    render_section_divider()
    st.header("2. Insight tổng quan")
    metric_value = f"{segmentation.silhouette_value:.3f}" if segmentation.silhouette_value is not None else "N/A"
    st.metric("Điểm tách biệt", metric_value)
    st.markdown(
        f"<div style='color:{silhouette_color}; font-weight:600;'>{silhouette_label}</div>",
        unsafe_allow_html=True,
    )

    if len(cluster_numeric_columns) < 2:
        st.info(
            "Bản đồ Insight cần ít nhất hai cột số. Hệ thống vẫn phân nhóm, "
            "nhưng sẽ tắt biểu đồ PCA 2D cho tệp này."
        )
    elif not has_plotly:
        show_plotly_missing_message()
    else:
        render_explainer("PCA (Principal Component Analysis) giúp biểu diễn dữ liệu nhiều chiều lên mặt phẳng.")
        st.plotly_chart(
            build_pca_figure(
                pca_coordinates=segmentation.pca_coordinates,
                labels=segmentation.labels,
                representative_coordinates=segmentation.representative_coordinates,
                color_map=color_map,
                group_name_map=group_name_map,
            ),
            use_container_width=True,
        )
        render_explainer([
            "PC1 = w11*x1 + w12*x2 + ... + w1p*xp.",
            "PC2 = w21*x1 + w22*x2 + ... + w2p*xp.",
        ])

    if show_elbow:
        if segmentation.wcss_by_group_count is not None and has_plotly:
            st.plotly_chart(build_wcss_figure(segmentation.wcss_by_group_count), use_container_width=True)
            render_explainer([
                "WCSS là tổng bình phương khoảng cách giữa các điểm dữ liệu và tâm cụm của chúng.",
                "WCSS càng thấp thì các điểm trong cụm càng chặt.",
            ])
        elif not has_plotly:
            show_plotly_missing_message()

    st.subheader("Bảng hồ sơ nhóm (Group Profiling Table)")
    st.dataframe(cluster_profile_styler, use_container_width=True)
    render_explainer([
        "Bảng này dùng công thức df.groupby('Cluster').mean() để tính trung bình theo cụm.",
        "Công thức: mean(c,j) = (1/nc) * Σ(xᵢⱼ) với i thuộc cụm c.",
    ])

    render_section_divider()
    st.header("3. Khám phá biến")
    render_explainer(
        "Biểu đồ này giữ nguyên đơn vị gốc để người dùng nghiệp vụ xem các nhóm theo những thước đo nhất định."
    )
    if len(cluster_numeric_columns) < 2:
        st.info("Khám phá biến cần ít nhất hai cột số để so sánh giữa các thước đo.")
    elif not has_plotly:
        show_plotly_missing_message()
    else:
        x_col, y_col = st.columns(2)
        with x_col:
            x_axis = st.selectbox("Trục X", cluster_numeric_columns, index=0)
        with y_col:
            default_y = 1 if len(cluster_numeric_columns) > 1 else 0
            y_axis = st.selectbox("Trục Y", cluster_numeric_columns, index=default_y)

        st.plotly_chart(
            build_variable_explorer_figure(
                df=cluster_numeric_df,
                x_axis=x_axis, y_axis=y_axis,
                labels=segmentation.labels,
                color_map=color_map,
                group_name_map=group_name_map,
            ),
            use_container_width=True,
        )

    render_section_divider()
    st.header("4. Xuất dữ liệu")
    st.markdown(
        "<div style='font-size:1.1rem; color:#000000; font-weight:600; "
        "line-height:1.5; margin:0.2rem 0 0.6rem 0;'>"
        "Tải xuống dữ liệu gốc kèm nhãn nhóm do mô hình gán.</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Xem trước dữ liệu xuất")
    st.dataframe(export_df.head(100), use_container_width=True)
    st.download_button(
        "Tải xuống CSV đã gán nhóm",
        data=to_csv_bytes(export_df),
        file_name="customer_segmentation_output.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
