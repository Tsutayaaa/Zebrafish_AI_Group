import shutil
from pathlib import Path


def organize_npy_from_F1(src_F1: str, dst_root: str):
    """
    遍历 src_F1 下所有一级子目录（例如 P1、P2、P3），
    将每个子目录下的 .npy 文件复制到 dst_root 下同名子目录中。

    例如：
    src_F1/
      P1/  *.npy
      P2/  *.npy

    输出：
    dst_root/
      P1/  <- P1 下所有 .npy
      P2/  <- P2 下所有 .npy
    """
    src_F1_path = Path(src_F1)
    dst_root_path = Path(dst_root)

    if not src_F1_path.exists():
        raise FileNotFoundError(f"Source folder not exists: {src_F1}")

    dst_root_path.mkdir(parents=True, exist_ok=True)

    # 遍历 F1 下的 P1、P2、P3……
    for second_level in src_F1_path.iterdir():
        if not second_level.is_dir():
            continue

        second_name = second_level.name
        target_dir = dst_root_path / second_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # 这个 P 目录下的所有 .npy（允许更深层级）
        npy_files = list(second_level.rglob("*.npy"))
        if not npy_files:
            continue

        print(f"[{second_name}] found {len(npy_files)} .npy files.")

        for npy in npy_files:
            dst_file = target_dir / npy.name

            # 如果目标已存在，自动加 _1, _2 后缀避免覆盖
            if dst_file.exists():
                stem = dst_file.stem
                suffix = dst_file.suffix
                idx = 1
                new_dst = target_dir / f"{stem}_{idx}{suffix}"
                while new_dst.exists():
                    idx += 1
                    new_dst = target_dir / f"{stem}_{idx}{suffix}"
                dst_file = new_dst

            shutil.copy2(npy, dst_file)
            print(f"  Copied: {npy} -> {dst_file}")

    print("\nDone!")


if __name__ == "__main__":
    # 你的 F1 路径（置顶的那个）
    src_F1 = "/Volumes/LA31A/FYP_TCM/TCM NTT"
    # 目标根目录
    dst_root = "/Users/shulei/Documents/TCM/TCM"

    organize_npy_from_F1(src_F1, dst_root)