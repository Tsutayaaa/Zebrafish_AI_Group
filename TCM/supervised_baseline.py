import numpy as np
import pandas as pd
from pathlib import Path

from typing import List

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import shap

# 如果要用 SMOTE / 随机过采样，需要安装 imbalanced-learn
# conda install -c conda-forge imbalanced-learn
from imblearn.over_sampling import SMOTE, RandomOverSampler


# =======================
# 配置区
# =======================

INPUT_CSV = "/Users/shulei/Documents/TCM/clean_behavior_dataset.csv"

# 使用哪些特征（这里已经排除了 tank_shape / trapezoid_side）
FEATURE_COLS: List[str] = [
    "total_displacement_mm",
    "avg_speed_mm_s",
    "top_time",
    "top_frequency",
    "freeze_time",
    "freeze_frequency",
]

# 预测的标签列：可以切换 "drug" 或 "group"
TARGET = "drug"

# 样本平衡方法：
#   "none"        : 不做重采样
#   "smote"       : SMOTE 过采样少数类（推荐）
#   "oversample"  : 简单随机过采样少数类
#   "class_weight": 不重采样，给模型加 class_weight="balanced"
BALANCE_METHOD = "smote"

TEST_SIZE = 0.25
RANDOM_STATE = 42

SHOW_FIG = True  # True = 画完 show，一般你可以保留 True


# =======================
# 辅助画图函数
# =======================

def plot_confusion_matrix(cm_array,
                          class_names,
                          title: str,
                          out_png: Path):
    """画并保存混淆矩阵热图"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_array, interpolation="nearest", cmap=cm.Blues)

    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # 在每个格子上标数字
    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(
                j,
                i,
                format(cm_array[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm_array[i, j] > thresh else "black",
                fontsize=7,
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[FIG] Confusion matrix saved to: {out_png}")
    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig)


def plot_rf_f1_bars(report_dict,
                    out_png: Path,
                    title: str = "Random Forest per-class F1-score"):
    """
    从 classification_report(output_dict=True) 里抽取每个类别的 F1，
    画柱状图。
    """
    # report_dict 的 key 中包含 "accuracy", "macro avg", "weighted avg" 等，需要排除
    class_labels = []
    f1_scores = []
    for k, v in report_dict.items():
        if k in ["accuracy", "macro avg", "weighted avg"]:
            continue
        # k 是类别名，v 是一个 dict，里面有 "precision", "recall", "f1-score"
        class_labels.append(k)
        f1_scores.append(v["f1-score"])

    # 按 f1 从高到低排序
    idx = np.argsort(f1_scores)[::-1]
    class_labels = [class_labels[i] for i in idx]
    f1_scores = [f1_scores[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(class_labels)), f1_scores)
    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1-score")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[FIG] RF per-class F1 barplot saved to: {out_png}")
    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig)


# =======================
# 主流程
# =======================

def main():
    print(f"读取数据: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # 1️⃣ 检查特征列是否存在
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]

    if missing_features:
        print("[WARN] 下列特征列在 CSV 中不存在，将被忽略：")
        for c in missing_features:
            print("  -", c)

    if not available_features:
        raise ValueError("FEATURE_COLS 中没有任何列在 CSV 中找到，无法训练。")

    print("\n本次使用的特征列：")
    for c in available_features:
        print("  -", c)

    FEATURE_USED = available_features

    # 2️⃣ 去掉 TARGET 为 NaN 的样本
    df = df.dropna(subset=[TARGET])
    print(f"\n去掉缺失 {TARGET} 后样本数: {len(df)}")
    if len(df) == 0:
        raise ValueError("所有样本的 TARGET 都是 NaN，无法训练。")

    # 3️⃣ 丢掉样本数 < 2 的类别
    vc = df[TARGET].value_counts()
    too_rare = vc[vc < 2].index.tolist()
    if too_rare:
        print("\n[INFO] 下面这些类别在数据中样本数 < 2，将被自动舍弃：")
        for cls in too_rare:
            print(f"  - {cls} (n={vc[cls]})")
        df = df[~df[TARGET].isin(too_rare)].copy()
        print(f"舍弃之后剩余样本数: {len(df)}")

    if len(df) == 0:
        raise ValueError("舍弃样本数 <2 的类别后没有剩余样本，无法训练。")

    # 4️⃣ 特征 NaN 处理（用中位数）
    for c in FEATURE_USED:
        if df[c].isna().any():
            med = df[c].median()
            n_missing = df[c].isna().sum()
            df[c] = df[c].fillna(med)
            print(f"[INFO] 特征 {c} 有 {n_missing} 个 NaN，已用中位数 {med:.3f} 填充。")

    X = df[FEATURE_USED].values
    y = df[TARGET].values

    # 5️⃣ 划分训练/测试
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"\nTrain 样本数: {len(X_train)}, Test 样本数: {len(X_test)}")

    # ====================
    # 样本平衡（只对训练集）
    # ====================
    print(f"\n样本平衡方法: {BALANCE_METHOD}")

    if BALANCE_METHOD in ["smote", "oversample"]:
        # 先看一下原始训练集各类别数量
        unique, counts = np.unique(y_train, return_counts=True)
        print("训练集原始类别分布：")
        for cls, cnt in zip(unique, counts):
            print(f"  - {cls}: {cnt}")

        if BALANCE_METHOD == "smote":
            sampler = SMOTE(random_state=RANDOM_STATE)
        else:  # "oversample"
            sampler = RandomOverSampler(random_state=RANDOM_STATE)

        X_train, y_train = sampler.fit_resample(X_train, y_train)

        unique, counts = np.unique(y_train, return_counts=True)
        print("\n训练集重采样后类别分布：")
        for cls, cnt in zip(unique, counts):
            print(f"  - {cls}: {cnt}")

        lr_class_weight = None
        rf_class_weight = None

    elif BALANCE_METHOD == "class_weight":
        # 不重采样，给模型 class_weight="balanced"
        lr_class_weight = "balanced"
        rf_class_weight = "balanced"
        print("使用 class_weight='balanced'，不做重采样。")

    else:  # "none"
        lr_class_weight = None
        rf_class_weight = None
        print("不进行任何样本平衡。")

    # 6️⃣ 标准化（在平衡之后 fit 在 X_train 上）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ====================
    # Logistic Regression
    # ====================
    print("\n===== Logistic Regression =====")
    lr = LogisticRegression(
        max_iter=500,
        multi_class="auto",  # 未来版本会自动变成 'multinomial'
        class_weight=lr_class_weight,
        solver="lbfgs",
    )
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)

    print(classification_report(y_test, y_pred_lr, digits=2))
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    print("混淆矩阵：")
    print(cm_lr)

    # 画 LR 混淆矩阵
    labels = sorted(np.unique(y_test))
    out_cm_lr = Path(INPUT_CSV).with_suffix(".logreg_confusion.png")
    plot_confusion_matrix(cm_lr, labels, "Logistic Regression - Confusion Matrix", out_cm_lr)

    # ====================
    # Random Forest
    # ====================
    print("\n===== Random Forest =====")
    clf_rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight=rf_class_weight,
    )
    clf_rf.fit(X_train_s, y_train)
    y_pred_rf = clf_rf.predict(X_test_s)

    report_rf = classification_report(y_test, y_pred_rf, digits=2)
    print(report_rf)

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print("混淆矩阵：")
    print(cm_rf)

    # 画 RF 混淆矩阵
    out_cm_rf = Path(INPUT_CSV).with_suffix(".rf_confusion.png")
    plot_confusion_matrix(cm_rf, labels, "Random Forest - Confusion Matrix", out_cm_rf)

    # RF per-class F1 柱状图
    report_rf_dict = classification_report(
        y_test, y_pred_rf, digits=3, output_dict=True
    )
    out_f1_rf = Path(INPUT_CSV).with_suffix(".rf_f1_bar.png")
    plot_rf_f1_bars(report_rf_dict, out_f1_rf)

    # RF feature importances
    print("\nRandom Forest feature importances:")
    for f, imp in zip(FEATURE_USED, clf_rf.feature_importances_):
        print(f"{f:25s}  {imp:.4f}")

    # ====================
    # SHAP 解释（用 RF）
    # ====================
    print("\n计算 SHAP 值（可能稍慢）...")
    explainer = shap.TreeExplainer(clf_rf)
    shap_values = explainer.shap_values(X_train_s)

    # 统一处理 shap_values 形状：
    # - 如果是 list（旧接口）：长度 = n_classes，每个元素 shape=(n_samples, n_features)
    #   -> 取绝对值后对类平均，得到 (n_samples, n_features)
    # - 如果是 ndarray（新接口）：
    #   - 2D: (n_samples, n_features) 直接用
    #   - 3D: (n_samples, n_features, n_classes) -> 取绝对值后在 axis=2 上平均
    if isinstance(shap_values, list):
        sv_list = [np.array(sv) for sv in shap_values]
        # stack -> (n_classes, n_samples, n_features)
        sv_stack = np.stack(sv_list, axis=0)
        # 按类别平均 -> (n_samples, n_features)
        sv = np.mean(np.abs(sv_stack), axis=0)
    else:
        sv_arr = np.array(shap_values)
        if sv_arr.ndim == 3:
            # (n_samples, n_features, n_classes)
            sv = np.mean(np.abs(sv_arr), axis=2)
        elif sv_arr.ndim == 2:
            # (n_samples, n_features)
            sv = sv_arr
        else:
            raise ValueError(
                f"未知的 shap_values 形状: {sv_arr.shape}，目前只支持 2D 或 3D。"
            )

    print("整理后的 SHAP 数组形状:", sv.shape, ", 特征矩阵形状:", X_train_s.shape)

    # 画 summary plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        sv,
        X_train_s,
        feature_names=FEATURE_USED,
        show=False,  # 先不 show，方便保存
    )
    out_shap = Path(INPUT_CSV).with_suffix(".shap_summary.png")
    plt.tight_layout()
    plt.savefig(out_shap, dpi=300, bbox_inches="tight")
    print(f"SHAP summary 图已保存到：{out_shap}")

    if SHOW_FIG:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()