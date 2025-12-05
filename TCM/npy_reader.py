import numpy as np
from pathlib import Path
from typing import Dict, Any

def load_behavior_npy(file_path: str) -> Dict[str, Any]:
    """
    读取一个 behavior_data.npy 文件，返回其中保存的 dict。

    Parameters
    ----------
    file_path : str
        路径，例如 'control_002_behavior_data.npy'

    Returns
    -------
    dict
        包含 tank_shape, speeds_mm_s, top_times 等键值的字典
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    arr = np.load(file_path, allow_pickle=True)

    if not isinstance(arr, np.ndarray) or arr.dtype != object or arr.ndim != 0:
        raise ValueError(f"Unexpected format in {file_path}, "
                         f"expected 0D object np.ndarray, got {type(arr)}, "
                         f"shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}")

    data = arr.item()
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict inside {file_path}, got {type(data)}")

    return data


if __name__ == "__main__":
    path = "/Users/shulei/Documents/TCM/20251016 Lavender_Saffron_Rose/control_002_behavior_data.npy"
    d = load_behavior_npy(path)
    print("Keys:", d.keys())
    print("tank_shape:", d["tank_shape"])
    print("total_displacement_mm:", d["total_displacement_mm"])
    print("len(speeds_mm_s):", len(d["speeds_mm_s"]))