from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# =========================
# 配置区：这里改参数
# =========================

# 输入 / 输出
INPUT_CSV = "/Users/shulei/Documents/TCM/clean_behavior_dataset.csv"
OUTPUT_CSV = "/Users/shulei/Documents/TCM/behavior_kmeans_cluster.csv"

# 特征列（和你降维脚本保持一致）
FEATURE_COLS: Optional[List[str]] = [
    "total_displacement_mm",
    "avg_speed_mm_s",
    "top_time",
    "top_frequency",
    "freeze_time",
    "freeze_frequency",
]

# 是否只分析部分 drug / group（和 rdim 一致的逻辑）
ANALYZE_DRUGS: Optional[List[str]] = None   # 比如 ["Kava", "Rose", "Control"]
ANALYZE_GROUPS: Optional[List[str]] = None  # 比如 ["Kava_10mg/L", "Kava_20mg/L"]

# 尝试的聚类数 k 范围
K_MIN = 2
K_MAX = 8

# PCA 可视化的维度（目前只做 2D 图）
PCA_N_COMPONENTS = 2

SHOW_FIG = True  # 是否弹出图


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

    exclude = {"group", "drug", "dose", "seq", "cluster"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols = [c for c in numeric_cols if c not in exclude]
    if not cols:
        raise ValueError("没有找到可用的数值特征列。")
    print("自动选择的特征列：")
    for c in cols:
        print("  -", c)
    return cols


def run_kmeans_scan(X_scaled: np.ndarray,
                    k_min: int,
                    k_max: int):
    """
    在 [k_min, k_max] 内扫描 k，计算 silhouette score，
    返回 (最好k, 各k结果表)
    """
    results = []
    best_k = None
    best_score = -1.0

    print("\n=== 扫描不同聚类数 k 的效果（silhouette score） ===")
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=0, n_init="auto")
        labels = km.fit_predict(X_scaled)
        # silhouette 需要至少 2 个 cluster 且每类有 >=2 个点
        try:
            score = silhouette_score(X_scaled, labels)
        except Exception as e:
            print(f"k = {k}: silhouette 计算失败: {e}")
            continue

        results.append((k, score))
        print(f"k = {k}: silhouette = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        raise RuntimeError("在指定的 k 范围内无法有效计算 silhouette。")

    print(f"\n>>> 选定的最佳 k = {best_k}, silhouette = {best_score:.4f}")
    return best_k, results


def plot_clusters_pca(df: pd.DataFrame,
                      X_scaled: np.ndarray,
                      cluster_labels: np.ndarray,
                      out_png: Path):
    """
    用 PCA 2D 把聚类结果可视化：
    - 颜色：cluster
    - marker 形状：drug
    """
    if PCA_N_COMPONENTS != 2:
        print("[INFO] 当前只实现 PCA 2D 可视化。")
        return

    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X_scaled)

    df_plot = pd.DataFrame({
        "pc1": X_pca[:, 0],
        "pc2": X_pca[:, 1],
        "cluster": cluster_labels,
        "drug": df["drug"].values if "drug" in df.columns else ["NA"] * len(df),
    })

    # 颜色按 cluster
    clusters = sorted(df_plot["cluster"].unique())
    cmap = plt.get_cmap("tab10")
    cluster_to_color = {c: cmap(i % 10) for i, c in enumerate(clusters)}

    # marker 按 drug
    unique_drugs = sorted(df_plot["drug"].unique())
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    drug_to_marker = {
        d: markers[i % len(markers)] for i, d in enumerate(unique_drugs)
    }

    fig, ax = plt.subplots(figsize=(7, 6))

    for c in clusters:
        sub_c = df_plot[df_plot["cluster"] == c]
        for d in unique_drugs:
            mask = (sub_c["drug"] == d)
            if not mask.any():
                continue
            ax.scatter(
                sub_c.loc[mask, "pc1"],
                sub_c.loc[mask, "pc2"],
                s=40,
                color=cluster_to_color[c],
                marker=drug_to_marker[d],
                edgecolor="k",
                linewidth=0.3,
                alpha=0.9,
                label=f"Cluster {c} - {d}",
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-means clusters (PCA 2D)\nColor: cluster, Marker: drug")

    # 合并 legend（避免重复项）
    handles, labels = ax.get_legend_handles_labels()
    # 用 (cluster, drug) 去重其实也 OK，这里简单去重
    used = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in used:
            used.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print("已保存聚类 PCA 图像到:", out_png)

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

    # 可选：过滤 drug / group
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
    print("\n用于聚类的特征：")
    for c in feature_cols:
        print("  -", c)

    # 提取特征矩阵
    X = df[feature_cols].astype(float).values

    # 处理 NaN
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
    print("已完成标准化。")

    # 扫描 k 并选择最佳 k
    best_k, k_results = run_kmeans_scan(X_scaled, K_MIN, K_MAX)

    # 使用最佳 k 重新拟合 KMeans
    print(f"\n使用最佳 k = {best_k} 拟合最终模型...")
    kmeans = KMeans(n_clusters=best_k, random_state=0, n_init="auto")
    cluster_labels = kmeans.fit_predict(X_scaled)

    # 写回 DataFrame
    df_out = df.copy()
    df_out["cluster"] = cluster_labels

    output_path = Path(OUTPUT_CSV)
    df_out.to_csv(output_path, index=False)
    print(f"\n已保存带 cluster 的数据到: {output_path}")

    # 打印 cluster × drug 分布
    if "drug" in df_out.columns:
        print("\n=== Cluster x Drug 交叉表（样本数量）===")
        print(pd.crosstab(df_out["cluster"], df_out["drug"]))

    # 画 PCA 聚类散点图
    png_path = output_path.with_suffix(".kmeans_pca2d.png")
    plot_clusters_pca(df_out, X_scaled, cluster_labels, png_path)


if __name__ == "__main__":
    main()