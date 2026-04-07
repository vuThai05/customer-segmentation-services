from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "exam_no1_output"
PLOT_DIR = OUTPUT_DIR / "plots"

DATA_CANDIDATES = [
    Path(r"D:\gradedata_v4.csv"),
    Path(r"D:\Data4csv"),
    Path(r"D:\Data4.csv"),
    BASE_DIR / "gradedata_v4.csv",
    BASE_DIR / "Data4csv",
    BASE_DIR / "Data4.csv",
]

LABELS = ["Low", "Medium", "Good", "Excellent"]


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def find_data_file() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Cannot find the dataset. Checked: "
        + ", ".join(str(path) for path in DATA_CANDIDATES)
    )


def load_data() -> pd.DataFrame:
    data_path = find_data_file()
    print_section("1) LOAD DATA")
    print(f"Loaded data from: {data_path}")

    df = pd.read_csv(data_path)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["gender_num"] = df["gender"].apply(
        lambda value: 1 if str(value).strip().lower() == "male" else 0
    )

    df["pretest"] = df[["pretest1", "pretest2", "pretest3"]].apply(
        np.mean,
        axis=1,
    )

    df["posttest"] = df[["posttest1", "postest2", "postest3"]].apply(
        np.mean,
        axis=1,
    )

    return df


def save_prepared_data(df: pd.DataFrame) -> None:
    output_file = OUTPUT_DIR / "prepared_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"\nPrepared dataset saved to: {output_file}")


def show_basic_info(df: pd.DataFrame) -> None:
    print_section("2) DATASET SHAPE")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print_section("3) COLUMN NAMES AND ENTRY COUNTS")
    summary = pd.DataFrame(
        {
            "column_name": df.columns,
            "non_null_entries": [df[column].count() for column in df.columns],
            "data_type": [str(df[column].dtype) for column in df.columns],
        }
    )
    print(summary.to_string(index=False))

    print_section("4) TOP 20 RECORDS")
    print(df.head(20).to_string(index=False))


def plot_histograms(df: pd.DataFrame) -> None:
    print_section("5) HISTOGRAMS FOR NUMERIC COLUMNS")

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    print("Numeric columns:", numeric_columns)

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    for axis, column in zip(axes, numeric_columns):
        sns.histplot(df[column], bins=15, kde=True, ax=axis, color="#2a9d8f")
        axis.set_title(column)

    for axis in axes[len(numeric_columns) :]:
        axis.axis("off")

    fig.suptitle("Histograms of Numeric Columns", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_histograms_numeric_columns.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {PLOT_DIR / '01_histograms_numeric_columns.png'}")


def show_groupby_counts(df: pd.DataFrame) -> None:
    print_section("6) GROUPBY COUNTS")

    gender_counts = df.groupby("gender").size().rename("count")
    age_counts = df.groupby("age").size().rename("count")
    gender_age_counts = df.groupby(["gender", "age"]).size().rename("count")

    print("Grouped by gender:")
    print(gender_counts.to_string())

    print("\nGrouped by age:")
    print(age_counts.to_string())

    print("\nGrouped by gender and age:")
    print(gender_age_counts.to_string())


def create_pretest_bins(df: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    print_section("7, 8, 9, 10) CREATE NEW COLUMNS AND PRETEST STATISTICS")

    print("gender_num created from gender with binary coding (female=0, male=1).")
    print("pretest created as the average of pretest1, pretest2, and pretest3.")

    quantiles = df["pretest"].quantile([0.00, 0.25, 0.50, 0.75, 1.00]).tolist()
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9

    df = df.copy()
    df["pretest_level"] = pd.cut(
        df["pretest"],
        bins=quantiles,
        labels=LABELS,
        include_lowest=True,
    )

    print(f"\nMean   : {df['pretest'].mean():.4f}")
    print(f"Median : {df['pretest'].median():.4f}")
    print(f"Mode   : {df['pretest'].mode().iloc[0]:.4f}")

    print("\nPercentile-based bins for pretest:")
    for left, right, label in zip(quantiles[:-1], quantiles[1:], LABELS):
        print(f"{label:<10} -> ({left:.4f}, {right:.4f}]")

    print("\nCategory counts:")
    print(df["pretest_level"].value_counts().sort_index().to_string())

    return df, quantiles


def correlation_analysis(df: pd.DataFrame) -> None:
    print_section("11) CORRELATION ANALYSIS")

    corr_matrix = df.select_dtypes(include=np.number).corr()
    corr_with_pretest = corr_matrix["pretest"].sort_values(ascending=False)

    print("Correlation with pretest:")
    print(corr_with_pretest.to_string())

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot to: {PLOT_DIR / '02_correlation_heatmap.png'}")
    print(
        "\nInterpretation: pretest is strongly related to pretest1, pretest2, and pretest3 "
        "because it is computed directly from those columns. Among the non-derived features, "
        "grade, hours, gender_num, and age have only very weak correlations."
    )


def plot_pretest_distribution(df: pd.DataFrame) -> None:
    print_section("12) DISTRIBUTION OF PRETEST")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["pretest"], bins=15, kde=True, ax=axes[0], color="#264653")
    axes[0].set_title("Histogram + KDE of pretest")

    sns.violinplot(y=df["pretest"], ax=axes[1], color="#e9c46a")
    axes[1].set_title("Violin Plot of pretest")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_pretest_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {PLOT_DIR / '03_pretest_distribution.png'}")


def visualize_pretest(df: pd.DataFrame) -> None:
    print_section("13) HEATMAP AND SCATTER VISUALIZATION FOR PRETEST")

    pivot_table = df.pivot_table(
        values="pretest",
        index="age",
        columns="gender",
        aggfunc="mean",
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("Mean pretest by age and gender")

    sns.scatterplot(
        data=df,
        x="grade",
        y="pretest",
        hue="gender",
        size="hours",
        palette="Set2",
        ax=axes[1],
    )
    axes[1].set_title("Scatter: grade vs pretest")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_pretest_heatmap_and_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {PLOT_DIR / '04_pretest_heatmap_and_scatter.png'}")


def plot_boxplots(df: pd.DataFrame) -> None:
    print_section("14) BOXPLOTS OF PRETEST BY GENDER AND AGE")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(data=df, x="gender", y="pretest", hue="gender", dodge=False, ax=axes[0])
    axes[0].set_title("pretest by gender")
    legend = axes[0].get_legend()
    if legend is not None:
        legend.remove()

    sns.boxplot(data=df, x="age", y="pretest", hue="gender", ax=axes[1])
    axes[1].set_title("pretest by age and gender")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_pretest_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {PLOT_DIR / '05_pretest_boxplots.png'}")


def plot_2d_and_3d(df: pd.DataFrame) -> None:
    print_section("15) 2D AND 3D PLOTS")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.regplot(data=df, x="grade", y="pretest", scatter_kws={"alpha": 0.65}, ax=axes[0])
    axes[0].set_title("2D Plot: grade vs pretest")

    sns.regplot(data=df, x="hours", y="pretest", scatter_kws={"alpha": 0.65}, ax=axes[1])
    axes[1].set_title("2D Plot: hours vs pretest")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_pretest_2d_plots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        df["age"],
        df["hours"],
        df["pretest"],
        c=df["grade"],
        cmap="viridis",
        s=60,
        alpha=0.8,
    )
    ax.set_xlabel("age")
    ax.set_ylabel("hours")
    ax.set_zlabel("pretest")
    ax.set_title("3D Plot: age, hours, and pretest")
    fig.colorbar(scatter, ax=ax, shrink=0.6, label="grade")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_pretest_3d_plot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {PLOT_DIR / '06_pretest_2d_plots.png'}")
    print(f"Saved plot to: {PLOT_DIR / '07_pretest_3d_plot.png'}")


def determine_influential_factors(df: pd.DataFrame) -> pd.Series:
    print_section("16) MOST INFLUENTIAL FACTORS")

    corr_with_pretest = (
        df[["gender_num", "age", "exercise", "hours", "grade", "pretest"]]
        .corr(numeric_only=True)["pretest"]
        .drop("pretest")
        .sort_values(key=lambda series: series.abs(), ascending=False)
    )

    print("Absolute correlation ranking using non-derived features:")
    print(corr_with_pretest.to_string())

    print(
        "\nInterpretation: grade and hours are the relatively strongest practical factors, "
        "but all correlations are very small. This means the available non-derived features "
        "do not explain pretest very well."
    )

    return corr_with_pretest


def build_and_evaluate_model(df: pd.DataFrame) -> tuple[Pipeline, float, float]:
    print_section("17, 18, 19) BUILD MODEL, EVALUATE, AND INTERPRET")

    feature_columns = ["gender", "age", "exercise", "hours", "grade"]
    target_column = "pretest"

    X = df[feature_columns]
    y = df[target_column]

    numeric_features = ["age", "exercise", "hours", "grade"]
    categorical_features = ["gender"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", Ridge()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    print("Selected model: Ridge Regression")
    print(f"R2 Score : {r2:.4f}")
    print(f"RMSE     : {rmse:.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, predictions, alpha=0.75, color="#457b9d")
    line_start = min(y_test.min(), predictions.min())
    line_end = max(y_test.max(), predictions.max())
    ax.plot([line_start, line_end], [line_start, line_end], color="red", linestyle="--")
    ax.set_xlabel("Actual pretest")
    ax.set_ylabel("Predicted pretest")
    ax.set_title("Actual vs Predicted pretest")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "08_model_actual_vs_predicted.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = pd.Series(
        model.named_steps["regressor"].coef_,
        index=feature_names,
    ).sort_values(key=lambda series: series.abs(), ascending=False)

    print("\nStandardized coefficient ranking:")
    print(coefficients.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=coefficients.values,
        y=coefficients.index,
        orient="h",
        color="#f4a261",
        ax=ax,
    )
    ax.set_title("Model Coefficients")
    ax.set_xlabel("Coefficient value")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "09_model_coefficients.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(
        "\nInterpretation: the model has a negative R2, so it performs worse than predicting "
        "the mean pretest score. The dataset does not contain strong non-leaky predictors "
        "for pretest."
    )
    print(f"\nSaved plot to: {PLOT_DIR / '08_model_actual_vs_predicted.png'}")
    print(f"Saved plot to: {PLOT_DIR / '09_model_coefficients.png'}")

    return model, r2, rmse


def confusion_matrix_analysis(df: pd.DataFrame, quantiles: list[float]) -> None:
    print_section("20) CONFUSION MATRIX BETWEEN PRETEST AND POSTTEST LEVELS")

    cm_edges = quantiles.copy()
    cm_edges[0] = min(df["pretest"].min(), df["posttest"].min()) - 1e-9
    cm_edges[-1] = max(df["pretest"].max(), df["posttest"].max()) + 1e-9

    df = df.copy()
    df["posttest_level"] = pd.cut(
        df["posttest"],
        bins=cm_edges,
        labels=LABELS,
        include_lowest=True,
    )

    matrix = confusion_matrix(
        df["pretest_level"].astype(str),
        df["posttest_level"].astype(str),
        labels=LABELS,
    )

    cm_df = pd.DataFrame(matrix, index=LABELS, columns=LABELS)
    print(cm_df.to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Posttest level")
    ax.set_ylabel("Pretest level")
    ax.set_title("Confusion Matrix: pretest level vs posttest level")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "10_confusion_matrix_pretest_posttest.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot to: {PLOT_DIR / '10_confusion_matrix_pretest_posttest.png'}")
    print(
        "\nInterpretation: the matrix is spread across many cells, so the categorized posttest "
        "results do not align tightly with the original pretest categories."
    )


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    sns.set_theme(style="whitegrid")

    ensure_output_dirs()

    df = load_data()
    df = add_features(df)

    save_prepared_data(df)
    show_basic_info(df)
    plot_histograms(df)
    show_groupby_counts(df)
    df, quantiles = create_pretest_bins(df)
    correlation_analysis(df)
    plot_pretest_distribution(df)
    visualize_pretest(df)
    plot_boxplots(df)
    plot_2d_and_3d(df)
    determine_influential_factors(df)
    build_and_evaluate_model(df)
    confusion_matrix_analysis(df, quantiles)

    print_section("DONE")
    print(f"All outputs were saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
