import re
from pathlib import Path

def count_drug_prefixes(folder: str):
    folder = Path(folder)

    # 匹配 XXX_003_behavior_data.npy
    pattern = re.compile(r"^(.+?)_\d+_behavior_data\.npy$")

    prefixes = set()

    # ★ 遍历所有子目录：rglob
    for f in folder.rglob("*_behavior_data.npy"):
        m = pattern.match(f.name)
        if m:
            prefixes.add(m.group(1))

    # 打印结果
    print("=== Drug Prefix Summary ===")
    print(f"Total types (XXX): {len(prefixes)}")
    print("Types:")
    for p in sorted(prefixes):
        print(" -", p)


if __name__ == "__main__":
    FOLDER = "/Users/shulei/Documents/TCM"
    count_drug_prefixes(FOLDER)