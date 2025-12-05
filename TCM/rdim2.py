import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ====================== 配置区：这里改参数 ======================

INPUT_CSV = "/Users/shulei/Documents/TCM/clean_behavior_dataset.csv"
OUTPUT_FIG = "/Users/shulei/Documents/TCM/behavior_pca_pubstyle.png"

# 使用哪些特征做 PCA（建议用你之前那 6 个标量）
FEATURE_COLS: Optional[List[str]] = [
    "total_displacement_mm",
    "avg_speed_mm_s",
    "top_time",
    "top_frequency",
    "freeze_time",
    "freeze_frequency",
]

# legend 用哪个列作为“药物名”
DRUG_COL = "drug"
# 同一个药物内部，用哪一列区分剂量（深浅）
DOSE_COL = "dose"

# 是否画特征向量箭头
SHOW_LOADINGS = True
LOADING_VECTOR_SCALE = 0.6   # 特征向量整体缩放
MARGIN_RATIO = 0.18          # 画布边缘留白比例
SHOW_FIG = True              # 画完是否 plt.show()

# ============================================================


def select_feature_columns(df: pd.DataFrame,
                           feature_cols: Optional[List[str]]) -> List[str]:
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"指定的特征列不存在: {missing}")
        return feature_cols
    # 兜底：自动选择全部数值列（去掉标签列）
    exclude = {"group", "drug", "dose", "seq", "trapezoid_side"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols = [c for c in numeric_cols if c not in exclude]
    if not cols:
        raise ValueError("没有找到可用的数值特征列")
    print("自动选择特征列：", cols)
    return cols


def adjust_lightness(color, factor: float):
    """同一 base color，不同明暗"""
    r, g, b = to_rgb(color)
    r = max(min(r * factor, 1), 0)
    g = max(min(g * factor, 1), 0)
    b = max(min(b * factor, 1), 0)
    return (r, g, b)


def parse_dose_value(dose_str) -> float:
    """从剂量字符串里提取数字，方便排序和深浅渐变"""
    import re
    s = str(dose_str)
    m = re.search(r"(\d+\.?\d*)", s)
    if m:
        return float(m.group(1))
    return 0.0


def confidence_ellipse(x, y, ax,
                       n_std=2.0,
                       facecolor='none',
                       edgecolor='red',
                       alpha=0.3,
                       linewidth=2.0):
    """画 2σ 置信椭圆"""
    if len(x) < 3:
        return  # 点太少就不画了
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # 长短轴
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    # 旋转角度
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
    )
    ax.add_patch(ellipse)
    return ellipse


def get_vector_alignment(x, y):
    """根据向量方向智能决定 label 对齐方式"""
    angle = np.arctan2(y, x)
    angle_deg = np.degrees(angle)
    if -45 <= angle_deg <= 45:
        return "left", "center"      # 右侧
    elif 45 < angle_deg <= 135:
        return "center", "bottom"    # 上方
    elif 135 < angle_deg <= 180 or -180 <= angle_deg <= -135:
        return "right", "center"     # 左侧
    else:
        return "center", "top"       # 下方


def calculate_plot_limits(pca_df: pd.DataFrame,
                          loadings: np.ndarray,
                          loading_vector_scale: float,
                          margin_ratio: float = 0.15):
    """让散点 + 椭圆 + 向量都能好好落在画布里"""
    def safe_division(a, b):
        return a / b if b != 0 else 0.0

    pc1_min, pc1_max = pca_df["PC1"].min(), pca_df["PC1"].max()
    pc2_min, pc2_max = pca_df["PC2"].min(), pca_df["PC2"].max()

    if loadings is not None and loading_vector_scale > 0:
        scale = min(
            safe_division((pc1_max - pc1_min),
                          (loadings[:, 0].max() - loadings[:, 0].min())),
            safe_division((pc2_max - pc2_min),
                          (loadings[:, 1].max() - loadings[:, 1].min())),
        ) * loading_vector_scale

        vector_pc1 = [loadings[i, 0] * scale for i in range(len(loadings))]
        vector_pc2 = [loadings[i, 1] * scale for i in range(len(loadings))]

        vector_pc1_min, vector_pc1_max = min(vector_pc1), max(vector_pc1)
        vector_pc2_min, vector_pc2_max = min(vector_pc2), max(vector_pc2)

        overall_pc1_min = min(pc1_min, vector_pc1_min)
        overall_pc1_max = max(pc1_max, vector_pc1_max)
        overall_pc2_min = min(pc2_min, vector_pc2_min)
        overall_pc2_max = max(pc2_max, vector_pc2_max)
    else:
        overall_pc1_min, overall_pc1_max = pc1_min, pc1_max
        overall_pc2_min, overall_pc2_max = pc2_min, pc2_max

    pc1_range = overall_pc1_max - overall_pc1_min
    pc2_range = overall_pc2_max - overall_pc2_min
    x_min = overall_pc1_min - pc1_range * margin_ratio
    x_max = overall_pc1_max + pc1_range * margin_ratio
    y_min = overall_pc2_min - pc2_range * margin_ratio
    y_max = overall_pc2_max + pc2_range * margin_ratio

    return x_min, x_max, y_min, y_max


def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入 CSV: {input_path}")

    print(f"读取数据: {input_path}")
    df = pd.read_csv(input_path)

    # 选择特征
    feature_cols = select_feature_columns(df, FEATURE_COLS)
    print("用于 PCA 的特征：", feature_cols)

    # 提取特征矩阵 + 去掉含 NaN 的样本
    X = df[feature_cols].astype(float).values
    nan_mask = np.isnan(X)
    if nan_mask.any():
        nan_per_col = nan_mask.sum(axis=0)
        print("\n[WARN] 检测到 NaN：")
        for col, n_missing in zip(feature_cols, nan_per_col):
            if n_missing > 0:
                print(f"  - {col}: {n_missing} 个 NaN")
        valid_mask = ~nan_mask.any(axis=1)
        dropped = (~valid_mask).sum()
        print(f"[INFO] 丢弃含 NaN 的样本: {dropped} 个")
        X = X[valid_mask]
        df = df.loc[valid_mask].reset_index(drop=True)

    # 标签列
    if DRUG_COL not in df.columns or DOSE_COL not in df.columns:
        raise ValueError(f"缺少 {DRUG_COL} 或 {DOSE_COL} 列，请检查 CSV。")

    drugs = df[DRUG_COL].values
    doses = df[DOSE_COL].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    pc1_var = explained[0] * 100
    pc2_var = explained[1] * 100

    # PCA 结果 DataFrame
    pca_df = pd.DataFrame({
        "PC1": pca_result[:, 0],
        "PC2": pca_result[:, 1],
        "drug": drugs,
        "dose": doses,
    })

    # 载荷（特征向量）
    loadings = pca.components_.T  # shape: (n_features, n_components)

    # 先计算绘图范围（考虑向量可能超出）
    x_min, x_max, y_min, y_max = calculate_plot_limits(
        pca_df, loadings, LOADING_VECTOR_SCALE, margin_ratio=MARGIN_RATIO
    )

    # 开始画图：1 个主图轴，未来你要加边缘图也可以在这基础上扩展
    fig, ax = plt.subplots(figsize=(7, 7))

    # 为不同 drug 分配 base color
    unique_drugs = sorted(pca_df["drug"].unique())
    cmap = plt.get_cmap("tab10")
    drug_to_color = {
        drug: cmap(i % 10) for i, drug in enumerate(unique_drugs)
    }

    # 画散点 & 记录 legend handle（每个 drug 只有一个 legend）
    handles = []
    labels = []

    for drug in unique_drugs:
        sub = pca_df[pca_df["drug"] == drug]
        # 剂量排序（低 → 高），配色从浅到深
        unique_doses = sorted(sub["dose"].unique(), key=parse_dose_value)
        n = len(unique_doses)
        if n == 1:
            factors = [1.0]
        else:
            factors = np.linspace(1.4, 0.7, n)

        # 本 drug 第一次画点时打 label，用于 legend
        first_label = drug
        base_color = drug_to_color[drug]

        for dose, factor in zip(unique_doses, factors):
            mask = (pca_df["drug"] == drug) & (pca_df["dose"] == dose)
            color = adjust_lightness(base_color, factor)
            label = first_label
            first_label = None  # 后续剂量不再打 label

            sc = ax.scatter(
                pca_df.loc[mask, "PC1"],
                pca_df.loc[mask, "PC2"],
                s=45,
                color=color,
                edgecolor="k",
                linewidth=0.4,
                alpha=0.85,
                label=label,
            )
            if label is not None:
                handles.append(sc)
                labels.append(label)

        # 为该 drug 画一个汇总置信椭圆（所有剂量的点一起）
        confidence_ellipse(
            sub["PC1"].values,
            sub["PC2"].values,
            ax=ax,
            n_std=2.0,
            facecolor=adjust_lightness(base_color, 1.15),
            edgecolor=base_color,
            alpha=0.18,
            linewidth=2.0,
        )

    # 画特征向量（载荷）
    if SHOW_LOADINGS and loadings is not None and loadings.shape[1] >= 2:
        # 基于当前范围重新算一个缩放系数（简单版）
        dx = x_max - x_min
        dy = y_max - y_min
        # loadings 的范围
        lx = loadings[:, 0]
        ly = loadings[:, 1]
        lx_span = lx.max() - lx.min() if lx.max() != lx.min() else 1.0
        ly_span = ly.max() - ly.min() if ly.max() != ly.min() else 1.0
        scale = min(dx / lx_span, dy / ly_span) * LOADING_VECTOR_SCALE

        for i, feat in enumerate(feature_cols):
            vx = loadings[i, 0] * scale
            vy = loadings[i, 1] * scale

            ax.arrow(
                0, 0,
                vx, vy,
                color="black",
                alpha=0.8,
                head_width=0.05 * max(dx, dy),
                linewidth=1.25,
                length_includes_head=True,
                zorder=5,
            )

            label_x = vx * 1.08
            label_y = vy * 1.08
            ha, va = get_vector_alignment(vx, vy)
            ax.text(
                label_x, label_y,
                feat,
                color="black",
                fontsize=11,
                ha=ha,
                va=va,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.2"),
                zorder=6,
            )

    # 坐标范围
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 轴标签：带上方差解释率
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)", fontsize=14, fontweight="bold")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)", fontsize=14, fontweight="bold")

    # 零基线
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3, linewidth=1.0)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3, linewidth=1.0)

    # 坐标轴线加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", width=1.2, length=5)

    # legend：每个 drug 一个条目
    if handles:
        leg = ax.legend(
            handles,
            labels,
            title="Drug",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.,
            fontsize=9,
        )
        leg.get_title().set_fontsize(10)

    # 标题
    ax.set_title("PCA of Behavioral Features\nColor=Drug, Shade=Dose", fontsize=14)

    plt.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=600)
    print("已保存图像到:", OUTPUT_FIG)

    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()