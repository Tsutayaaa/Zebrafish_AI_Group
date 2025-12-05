import re
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any


# =========================
# 映射表
# =========================
PREFIX_TO_GROUP: Dict[str, str] = {
    "10": "Kava_10mg/L",
    "20mg": "Kava_20mg/L",
    "50mg": "Kava_50mg/L",
    "Kava_50": "Kava_50mg/L",
    "Kava_100": "Kava_100mg/L",
    "Kava_150": "Kava_150mg/L",

    "1g": "Rose_1g/L",
    "3g": "Rose_3g/L",
    "6g": "Rose_6g/L",
    "rose_1mgl": "Rose_1mg/L",
    "rose_3mgl": "Rose_3mg/L",
    "rose_6mgl": "Rose_6mg/L",
    "rose_12mgl": "Rose_12mg/L",
    "rose": "Control",

    "Binlang_1": "Binlang_1g/L",
    "Binlang_3": "Binlang_3g/L",
    "Binlang_6": "Binlang_6g/L",

    "Calendula_1": "Calendula_1g/L",
    "Calendula_3": "Calendula_3g/L",
    "Calendula_6": "Calendula_6g/L",

    "Celastrus_1": "Celastrus_1g/L",
    "celastrus_3": "Celastrus_3g/L",
    "celastrus_6": "Celastrus_6g/L",

    "Ginseng": "Control",
    "Ginseng_1": "Ginseng_1g/L",
    "Ginseng_3": "Ginseng_3g/L",
    "Ginseng_6": "Ginseng_6g/L",

    "Lavender_1": "Lavender_1g/L",
    "Lavender_3": "Lavender_3g/L",
    "Lavender_6": "Lavender_6g/L",
    "lavender": "Control",
    "lavendar": "Control",

    "Passiflora_3": "Passiflora_3g/L",
    "Passiflora_6": "Passiflora_6g/L",
    "passiflora_1": "Passiflora_1g/L",
    "passiflora": "Control",

    "Saffron": "Control",
    "Saffron_1gL": "Saffron_1g/L",
    "Saffron_3gL": "Saffron_3g/L",
    "Saffron_6gL": "Saffron_6g/L",
    "safferon": "Control",

    "Tobacco_0": "Control",
    "Tobacco_1": "Tobacco_1g/L",
    "Tobacco_3": "Tobacco_3g/L",
    "Tobacco_6": "Tobacco_6g/L",

    "peppermint_1": "Peppermint_1g/L",
    "peppermint_3": "Peppermint_3g/L",
    "peppermint_6": "Peppermint_6g/L",

    "Stjohnswort": "Stjohnswort",
    "albizia": "Control",

    "Control": "Control",
    "Ctrl": "Control",
    "ctrl": "Control",
    "control": "Control",
    "comtrol": "Control",

    "test": "test",
}

# 文件模式匹配：XXX_003_behavior_data.npy
FILENAME_PATTERN = re.compile(r"^(.+?)_(\d+)_behavior_data\.npy$")


def extract_prefix_and_seq(fname: str):
    """
    解析 prefix 和 seq，例如：
    'rose_003_behavior_data.npy' → ('rose', '003')
    """
    m = FILENAME_PATTERN.match(fname)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def load_behavior(path: Path) -> Dict[str, Any]:
    arr = np.load(path, allow_pickle=True)
    return arr.item()


def split_drug_and_dose(group: str):
    if "_" in group:
        drug, dose = group.split("_", 1)
        return drug, dose
    if group == "Control":
        return "Control", "0"
    return group, "NA"


def collect_dataset(root_folder: str) -> pd.DataFrame:
    root = Path(root_folder)
    files = list(root.rglob("*_behavior_data.npy"))
    print(f"Found {len(files)} raw behavior files.")

    rows = []

    for f in files:
        prefix, seq = extract_prefix_and_seq(f.name)

        if prefix is None:
            print(f"[WARN] skip (bad name): {f.name}")
            continue

        if prefix not in PREFIX_TO_GROUP:
            print(f"[ERROR] Unknown prefix '{prefix}' in file {f.name}")
            continue

        unified = PREFIX_TO_GROUP[prefix]
        if unified in [None, ""]:
            print(f"[ERROR] Bad mapping for prefix '{prefix}' (None/Empty)")
            continue

        drug, dose = split_drug_and_dose(unified)

        # 读取 n
        data = load_behavior(f)

        row = {
            "group": unified,
            "drug": drug,
            "dose": dose,
            "seq": int(seq),   # 重要：数字化 → 方便排序
        }

        # save all behavior features
        for k, v in data.items():
            row[k] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Collected {len(df)} valid samples.")

    # 排序：drug → dose → seq
    df = df.sort_values(by=["drug", "dose", "seq"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    ROOT = "/Users/shulei/Documents/TCM"
    OUT = "/Users/shulei/Documents/TCM/clean_behavior_dataset.csv"

    df = collect_dataset(ROOT)
    df.to_csv(OUT, index=False)

    print("Saved:", OUT)