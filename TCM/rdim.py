from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (需要触发 3D 支持)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# =========================
# 配置区：这里改参数
# =========================

# 输入 / 输出文件
INPUT_CSV = "/Users/shulei/Documents/TCM/clean_behavior_dataset.csv"
OUTPUT_CSV = "/Users/shulei/Documents/TCM/behavior_tsne_2d.csv"

# 降维算法: "pca" / "umap" / "tsne"
ALGO = "pca"

# 降到几维：2 或 3（2D/3D 决定）
N_COMPONENTS = 3  # 改成 3 就会画 3D 图（不画边缘图和椭圆）

# 特征列：
# - None = 自动选择所有数值特征（除 group, drug, dose, seq）
# - 或者指定列表，例如：
FEATURE_COLS: Optional[List[str]] = [
    "total_displacement_mm",
    "avg_speed_mm_s",
    "top_time",
    "top_frequency",
    "freeze_time",
    "freeze_frequency",
]

# 只分析指定的 drug（类别），留空则使用全部
# 例如：ANALYZE_DRUGS = ["Kava", "Rose", "Control"]
ANALYZE_DRUGS: Optional[List[str]] = None

# 或者按 group（drug+剂量）过滤，留空则不过滤
# 例如：ANALYZE_GROUPS = ["Kava_10mg/L", "Kava_20mg/L"]
ANALYZE_GROUPS: Optional[List[str]] = None

# 是否显示图形窗口（PyCharm 里可以关掉弹窗只保存）
SHOW_FIG = True

# 置信椭圆的“sigma”倍数，大约 2 ~ 2.5 对应 ~95% 范围
ELLIPSE_N_STD = 2.0


# =========================
# 工具函数
# =========================

def select_feature_columns(df: pd.DataFrame,
                           feature_cols: Optional[List[str]]) -> List[str]:
    """自动或手动选择特征列"""
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"指定的特征列在 CSV 中不存在: {missing}")
        return feature_cols

    exclude = {"group", "drug", "dose", "seq"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols = [c for c in numeric_cols if c not in exclude]
    if not cols:
        raise ValueError("没有找到可用的数值特征列。")
    print("自动选择的特征列：")
    for c in cols:
        print("  -", c)
    return cols


def run_dim_reduction(X_scaled: np.ndarray,
                      algo: str,
                      n_components: int) -> np.ndarray:
    """根据指定算法进行降维"""
    algo = algo.lower()
    if algo == "pca":
        model = PCA(n_components=n_components, random_state=0)
        X_emb = model.fit_transform(X_scaled)
        print("PCA 解释方差比例:", model.explained_variance_ratio_)
        return X_emb

    elif algo == "umap":
        try:
            import umap  # type: ignore
        except ImportError:
            raise ImportError("需要先安装 umap-learn: pip install umap-learn")
        model = umap.UMAP(n_components=n_components, random_state=0)
        X_emb = model.fit_transform(X_scaled)
        return X_emb

    elif algo == "tsne":
        model = TSNE(
            n_components=n_components,
            random_state=0,
            init="pca",
            learning_rate="auto",
        )
        X_emb = model.fit_transform(X_scaled)
        return X_emb

    else:
        raise ValueError(f"未知算法: {algo}（支持 'pca'/'umap'/'tsne'）")


def adjust_lightness(color, factor: float):
    """
    调整颜色明暗：factor < 1 变暗，>1 变亮
    color 可以是 'tab:blue' 或 '#RRGGBB'
    """
    r, g, b = to_rgb(color)
    r = max(min(r * factor, 1), 0)
    g = max(min(g * factor, 1), 0)
    b = max(min(b * factor, 1), 0)
    return (r, g, b)


def parse_dose_value(dose_str: str) -> float:
    """
    尝试从 dose 字符串中提取数值部分，用于排序深浅：
    "10mg/L" -> 10
    "3g/L"   -> 3
    "0"      -> 0
    提取失败时返回 0
    """
    import re
    m = re.search(r"(\d+\.?\d*)", str(dose_str))
    if m:
        return float(m.group(1))
    return 0.0


def confidence_ellipse(x, y, ax, n_std=2.0, edgecolor="black", **kwargs):
    """
    在 ax 上画 2D 置信椭圆。
    x, y 为 1D 数组；n_std 为“标准差倍数”，如 2 表示 2σ 椭圆。
    """
    if x.size < 3:
        # 点数太少无法估计协方差，直接跳过
        return

    cov = np.cov(x, y)
    if np.linalg.det(cov) == 0:
        # 协方差病态，跳过
        return

    vals, vecs = np.linalg.eigh(cov)
    # 按从大到小排序
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    # 椭圆轴长：2*n_std*sqrt(特征值)
    width, height = 2 * n_std * np.sqrt(vals)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    ell = Ellipse(
        (mean_x, mean_y),
        width=width,
        height=height,
        angle=theta,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=1.2,
        alpha=0.8,
        **kwargs,
    )
    ax.add_patch(ell)

from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec

def plot_embedding(df_emb: pd.DataFrame,
                   algo: str,
                   n_components: int,
                   out_png: Path):

    if n_components not in (2, 3):
        print("[WARN] 只支持 2D/3D，本次跳过绘图。")
        return

    algo_name = algo.upper()

    # 配色：每个 drug 一个 base color
    unique_drugs = sorted(df_emb["drug"].unique())
    if not unique_drugs:
        print("[WARN] 没有 drug。")
        return

    cmap = plt.get_cmap("tab10")
    drug_to_color = {drug: cmap(i % 10) for i, drug in enumerate(unique_drugs)}

    # =============== 2D + KDE ===============
    if n_components == 2:
        fig = plt.figure(figsize=(9, 9))

        # 先用 gridspec 定一个大概的布局
        gs = gridspec.GridSpec(
            7, 7,
            wspace=0.02,
            hspace=0.02,
            left=0.08,
            right=0.88,   # 右边刻意留点空白给 legend
            bottom=0.08,
            top=0.92,
        )

        # 主图 + 侧边 KDE（位置后面会再手动微调）
        ax_scatter = fig.add_subplot(gs[1:, :5])
        ax_histx   = fig.add_subplot(gs[0,  :5], sharex=ax_scatter)
        ax_histy   = fig.add_subplot(gs[1:, 5],  sharey=ax_scatter)

        ax_histx.axis("off")
        ax_histy.axis("off")
    else:
        fig = plt.figure(figsize=(8, 6))
        ax_scatter = fig.add_subplot(111, projection="3d")
        ax_histx = None
        ax_histy = None

    # ============================================================
    # 第一次循环：先画主图散点 + 椭圆（不画 KDE）
    # ============================================================
    handles, labels = [], []

    for drug in unique_drugs:
        sub = df_emb[df_emb["drug"] == drug]
        base_color = drug_to_color[drug]

        # 剂量深浅
        doses = sorted(sub["dose"].unique(), key=parse_dose_value)
        factors = [1.0] if len(doses) == 1 else np.linspace(1.4, 0.7, len(doses))

        first_label = drug
        for dose, factor in zip(doses, factors):
            mask = (df_emb["drug"] == drug) & (df_emb["dose"] == dose)
            color = adjust_lightness(base_color, factor)

            if n_components == 2:
                sc = ax_scatter.scatter(
                    df_emb.loc[mask, "dim1"],
                    df_emb.loc[mask, "dim2"],
                    s=15,
                    color=color,
                    edgecolor="k",
                    linewidth=0.3,
                    alpha=0.9,
                    label=first_label,
                    zorder=3,
                )
            else:
                sc = ax_scatter.scatter(
                    df_emb.loc[mask, "dim1"],
                    df_emb.loc[mask, "dim2"],
                    df_emb.loc[mask, "dim3"],
                    s=15,
                    color=color,
                    edgecolor="k",
                    linewidth=0.3,
                    alpha=0.9,
                    label=first_label,
                )

            if first_label is not None:
                handles.append(sc)
                labels.append(first_label)
            first_label = None

        # 椭圆
        if n_components == 2 and len(sub) > 2:
            x = sub["dim1"].values
            y = sub["dim2"].values
            cov = np.cov(x, y)

            if np.linalg.det(cov) != 0:
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals = vals[order]
                vecs = vecs[:, order]

                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                W, H = 2 * ELLIPSE_N_STD * np.sqrt(vals)

                ell = Ellipse(
                    (x.mean(), y.mean()),
                    width=W, height=H,
                    angle=theta,
                    fill=False,
                    edgecolor=adjust_lightness(base_color, 0.6),
                    lw=1.3,
                    alpha=0.9,
                    zorder=2,
                )
                ax_scatter.add_patch(ell)

    # ============================================================
    # 第二步：根据散点范围 → 数据正方形 + 视觉正方形
    # ============================================================
    if n_components == 2:
        # 先让 Matplotlib 自动算所有点的范围
        ax_scatter.relim()
        ax_scatter.autoscale_view()

        x_min, x_max = ax_scatter.get_xlim()
        y_min, y_max = ax_scatter.get_ylim()

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        R  = max(x_max - x_min, y_max - y_min) / 2

        # 数据范围变成正方形
        ax_scatter.set_xlim(cx - R, cx + R)
        ax_scatter.set_ylim(cy - R, cy + R)

        # 视觉上的绘图区也强制为正方形
        ax_scatter.set_box_aspect(1)

        # 用“主图最终范围”生成 KDE 网格，保证完全对齐
        x_grid = np.linspace(cx - R, cx + R, 300)
        y_grid = np.linspace(cy - R, cy + R, 300)

    # ============================================================
    # 第三步：画 KDE（确保对齐主图）
    # ============================================================
    if n_components == 2:
        for drug in unique_drugs:
            sub = df_emb[df_emb["drug"] == drug]
            if len(sub) < 3:
                continue

            base_color = drug_to_color[drug]
            x = sub["dim1"].values
            y = sub["dim2"].values

            try:
                kde_x = gaussian_kde(x)
                kde_y = gaussian_kde(y)
                dens_x = kde_x(x_grid)
                dens_y = kde_y(y_grid)

                # 顶部 KDE
                ax_histx.plot(x_grid, dens_x, color=base_color, lw=1.2, alpha=0.8, zorder=1)
                ax_histx.fill_between(x_grid, 0, dens_x,
                                      color=base_color, alpha=0.25, zorder=0)

                # 右侧 KDE
                ax_histy.plot(dens_y, y_grid, color=base_color, lw=1.2, alpha=0.8, zorder=1)
                ax_histy.fill_betweenx(y_grid, 0, dens_y,
                                       color=base_color, alpha=0.25, zorder=0)
            except Exception as e:
                print(f"[WARN] KDE fail ({drug}): {e}")

        # ========================================================
        # 第四步：手动把侧图“贴”到主图边上
        # ========================================================
        fig.canvas.draw()  # 让布局生效，拿到真实位置

        bbox = ax_scatter.get_position()
        gap = 0.01
        top_h = bbox.height * 0.18   # 顶部 KDE 占主图高度的比例
        right_w = bbox.width * 0.18  # 右侧 KDE 占主图宽度的比例

        # 顶部：紧贴主图上沿
        ax_histx.set_position([
            bbox.x0,
            bbox.y1 + gap,
            bbox.width,
            top_h,
        ])

        # 右侧：紧贴主图右侧
        ax_histy.set_position([
            bbox.x1 + gap,
            bbox.y0,
            right_w,
            bbox.height,
        ])

    # ============================================================
    # Title & Labels
    # ============================================================
    ax_scatter.set_xlabel("dim1")
    ax_scatter.set_ylabel("dim2")

    fig.suptitle(
        f"{algo_name} embedding ({n_components}D)",
        y=0.98,
        fontsize=14,
        fontweight="bold"
    )

    # ============================================================
    # Legend：放在“主图 + 侧图”的外面（右侧空白区）
    # ============================================================
    # 用 figure 级别的 legend，锚到整张图的右边中部
    fig.legend(
        handles,
        labels,
        loc="upper right",
        # bbox_to_anchor=(0.88, 0.7),  # 靠右侧一点（> right=0.88），不挡侧图
        fontsize=8,
        title="Drug",
        frameon=True,
        borderpad=0.4,
        fancybox=True,
        framealpha=0.9,
    )

    # ============================================================
    # Save & Show
    # ============================================================
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    # 额外保存一份 SVG 矢量图
    svg_path = out_png.with_suffix(".svg")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight")

    print("已保存 PNG 到:", out_png)
    print("已保存 SVG 到:", svg_path)

    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig)

# =========================
# 主流程
# =========================

def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入 CSV: {input_path}")

    print(f"读取数据: {input_path}")
    df = pd.read_csv(input_path)

    # -------- 过滤想分析的类别 --------
    if ANALYZE_DRUGS is not None:
        df = df[df["drug"].isin(ANALYZE_DRUGS)].copy()
        print(f"按 drug 过滤后剩余样本数: {len(df)}")

    if ANALYZE_GROUPS is not None:
        df = df[df["group"].isin(ANALYZE_GROUPS)].copy()
        print(f"按 group 过滤后剩余样本数: {len(df)}")

    if df.empty:
        print("[WARN] 过滤后没有样本，退出。")
        return

    # 选择特征列
    feature_cols = select_feature_columns(df, FEATURE_COLS)
    print("\n用于分析的特征：")
    for c in feature_cols:
        print("  -", c)

    # 提取特征矩阵
    X = df[feature_cols].astype(float).values

    # NaN 检查 & 清理
    nan_mask = np.isnan(X)
    if nan_mask.any():
        nan_per_col = nan_mask.sum(axis=0)
        print("\n[WARN] 检测到 NaN：")
        for col, n_missing in zip(feature_cols, nan_per_col):
            if n_missing > 0:
                print(f"  - {col}: {n_missing} 个 NaN")

        valid_row_mask = ~nan_mask.any(axis=1)
        n_before = X.shape[0]
        n_after = valid_row_mask.sum()
        print(f"[INFO] 丢弃含 NaN 的样本: {n_before - n_after} 个，剩余 {n_after} 个。")

        X = X[valid_row_mask]
        df = df.loc[valid_row_mask].reset_index(drop=True)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n特征矩阵形状: {X_scaled.shape}")
    print("已完成标准化（均值=0，方差=1）。")

    # 降维
    print(f"\n开始降维: 算法={ALGO}, 维度={N_COMPONENTS}")
    X_emb = run_dim_reduction(X_scaled, ALGO, N_COMPONENTS)

    # 组装输出表：group/drug/dose/seq + dim1, dim2, (dim3)
    out_df = pd.DataFrame()
    for col in ["group", "drug", "dose", "seq"]:
        if col in df.columns:
            out_df[col] = df[col]

    for i in range(N_COMPONENTS):
        out_df[f"dim{i+1}"] = X_emb[:, i]

    output_path = Path(OUTPUT_CSV)
    out_df.to_csv(output_path, index=False)
    print(f"\n已保存降维结果到: {output_path}")
    print("输出列：", list(out_df.columns))

    # 绘图：根据输出文件名自动生成 PNG 名称
    png_path = output_path.with_suffix(f".{ALGO}_{N_COMPONENTS}d.png")
    plot_embedding(out_df, ALGO, N_COMPONENTS, png_path)


if __name__ == "__main__":
    main()