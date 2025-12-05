#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Binary baseline: Control vs Drug
- 读取 clean_behavior_dataset.csv
- 使用行为特征做二分类（Control vs 其他所有药物）
- 可选 SMOTE 平衡
- 训练 Logistic Regression 和 Random Forest
- 输出分类报告、混淆矩阵、ROC 曲线
- 对 RF 做 SHAP 特征重要性分析
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
import shap
import seaborn as sns

# =========================
# 配置区
# =========================

INPUT_CSV = "/Users/shulei/Documents/TCM/clean_behavior_dataset.csv"
OUTPUT_PREFIX = "/Users/shulei/Documents/TCM/clean_behavior_dataset.control_vs_drug"

# 使用的特征列（你刚才选的那 6 个）
FEATURE_COLS: List[str] = [
    "total_displacement_mm",
    "avg_speed_mm_s",
    "top_time",
    "top_frequency",
    "freeze_time",
    "freeze_frequency",
]

# 样本平衡方法: "none" / "smote"
BALANCING_METHOD = "smote"

RANDOM_STATE = 42
TEST_SIZE = 0.25


# =========================
# 工具函数
# =========================

def plot_confusion_matrix(cm, class_names, out_path: Path, title: str):
    """画并保存混淆矩阵"""
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[FIG] Confusion matrix saved to: {out_path}")


def plot_roc_curve(y_test, y_score_dict, out_path: Path, title: str):
    """
    y_score_dict: dict[name -> prob_of_positive_class]
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    for name, y_score in y_score_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[FIG] ROC curve saved to: {out_path}")


# =========================
# 主流程
# =========================

def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入 CSV: {input_path}")

    print(f"读取数据: {input_path}")
    df = pd.read_csv(input_path)

    # 只保留 drug 非空的样本
    df = df[~df["drug"].isna()].copy()
    print(f"去掉缺失 drug 后样本数: {len(df)}")

    # 构造二分类标签：Control vs Drug
    # Control -> 0, 其他所有药 -> 1
    df["label_binary"] = np.where(df["drug"] == "Control", 0, 1)

    # 提取特征矩阵
    X = df[FEATURE_COLS].astype(float).values
    y = df["label_binary"].values

    # 严格检查 NaN
    nan_mask = np.isnan(X)
    if nan_mask.any():
        nan_per_col = nan_mask.sum(axis=0)
        print("\n[WARN] 特征中存在 NaN：")
        for col, n_missing in zip(FEATURE_COLS, nan_per_col):
            if n_missing > 0:
                print(f"  - {col}: {n_missing} 个 NaN")
        valid_row_mask = ~nan_mask.any(axis=1)
        n_before = X.shape[0]
        n_after = valid_row_mask.sum()
        print(f"[INFO] 丢弃含 NaN 的样本: {n_before - n_after} 个，剩余 {n_after} 个。")
        X = X[valid_row_mask]
        y = y[valid_row_mask]
        df = df.loc[valid_row_mask].reset_index(drop=True)

    # Train / Test 划分（使用二分类标签 stratify）
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTrain 样本数: {len(y_train)}, Test 样本数: {len(y_test)}")

    # 看一下原始类别分布
    unique, counts = np.unique(y_train, return_counts=True)
    print("\n训练集原始类别分布（0=Control, 1=Drug）：")
    for cls, cnt in zip(unique, counts):
        print(f"  - {cls}: {cnt}")

    # 可选 SMOTE 平衡
    if BALANCING_METHOD.lower() == "smote":
        print("\n样本平衡方法: smote")
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        unique_b, counts_b = np.unique(y_train, return_counts=True)
        print("训练集重采样后类别分布：")
        for cls, cnt in zip(unique_b, counts_b):
            print(f"  - {cls}: {cnt}")
    else:
        print("\n样本平衡方法: none（不做重采样）")

    # 标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ============
    # Logistic Regression
    # ============
    print("\n===== Logistic Regression (Control vs Drug) =====")
    logreg = LogisticRegression(
        max_iter=5000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    logreg.fit(X_train_s, y_train)
    y_pred_lr = logreg.predict(X_test_s)
    y_prob_lr = logreg.predict_proba(X_test_s)[:, 1]  # 预测为 Drug 的概率

    print(classification_report(y_test, y_pred_lr, target_names=["Control", "Drug"]))

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_path_lr = Path(OUTPUT_PREFIX + ".logreg_confusion_binary.png")
    plot_confusion_matrix(cm_lr, ["Control", "Drug"], cm_path_lr, "LogReg Confusion Matrix")

    # ============
    # Random Forest
    # ============
    print("\n===== Random Forest (Control vs Drug) =====")
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)  # RF 对原始尺度不太敏感，用未标准化的也可
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_rf, target_names=["Control", "Drug"]))

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_path_rf = Path(OUTPUT_PREFIX + ".rf_confusion_binary.png")
    plot_confusion_matrix(cm_rf, ["Control", "Drug"], cm_path_rf, "RF Confusion Matrix")

    # ============
    # ROC 曲线（两个模型一起画）
    # ============
    roc_path = Path(OUTPUT_PREFIX + ".roc_curve.png")
    plot_roc_curve(
        y_test,
        {"LogReg": y_prob_lr, "RandomForest": y_prob_rf},
        roc_path,
        "Control vs Drug ROC",
    )

    # ============
    # RF 特征重要性
    # ============
    importances = rf.feature_importances_
    feat_importance = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nRandom Forest feature importances:")
    print(feat_importance.to_string(float_format=lambda x: f"{x:0.4f}"))

    # 画一个简单的条形图
    fig, ax = plt.subplots(figsize=(5, 4))
    feat_importance[::-1].plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title("RF Feature Importances (Control vs Drug)")
    fig.tight_layout()
    fi_path = Path(OUTPUT_PREFIX + ".rf_feature_importance.png")
    fig.savefig(fi_path, dpi=300)
    plt.close(fig)
    print(f"[FIG] RF feature importance barplot saved to: {fi_path}")

    # ============
    # SHAP 分析（只针对 RF）
    # ============
    print("\n计算 RF 的 SHAP 值（可能稍慢）...")
    # 为了 SHAP 稳定，把训练数据缩减一部分也可以，这里直接全部用
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_train)  # 对二分类，返回 [class0, class1]

    # 取对 “Drug”（1 类）的 SHAP 值
    shap_drug = shap_values[1]  # shape: (n_samples_train, n_features)

    print(
        f"SHAP 数组形状: {shap_drug.shape} , "
        f"特征矩阵形状: {X_train.shape}"
    )

    shap_fig_path = Path(OUTPUT_PREFIX + ".rf_shap_summary.png")
    plt.figure(figsize=(6, 4))
    shap.summary_plot(
        shap_drug,
        features=X_train,
        feature_names=FEATURE_COLS,
        show=False
    )
    plt.tight_layout()
    plt.savefig(shap_fig_path, dpi=300)
    plt.close()
    print(f"SHAP summary 图已保存到：{shap_fig_path}")


if __name__ == "__main__":
    main()