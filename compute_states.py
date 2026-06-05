# 📁 compute_stats.py
import numpy as np
import h5py
from tqdm import tqdm
import config
import utils
import os

def main():
    hdf5_path = config.DATA_CONFIG["hdf5_path"]
    print(f"HDF5ファイルから統計量を計算します: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        num_scenarios = f['input_real'].shape[2]  # 100

        all_real_parts = []
        all_imag_parts = []

        print("全学習データを読み込み中...")
        for s in tqdm(range(num_scenarios)):
            # shape: (512, 128) = (range_bins, chirps)
            real = f['input_real'][:, :, s]
            imag = f['input_imag'][:, :, s]
            all_real_parts.append(real.flatten())
            all_imag_parts.append(imag.flatten())

    all_real_parts = np.concatenate(all_real_parts)
    all_imag_parts = np.concatenate(all_imag_parts)

    mean_real = np.mean(all_real_parts)
    std_real  = np.std(all_real_parts)
    mean_imag = np.mean(all_imag_parts)
    std_imag  = np.std(all_imag_parts)

    print(f"\n計算された統計量:")
    print(f"  実部  平均: {mean_real:.6f}, 標準偏差: {std_real:.6f}")
    print(f"  虚部  平均: {mean_imag:.6f}, 標準偏差: {std_imag:.6f}")

    stats = {
        'mean': np.array([mean_real, mean_imag], dtype=np.float32),
        'std':  np.array([std_real,  std_imag],  dtype=np.float32)
    }

    save_dir = os.path.dirname(config.PREPROCESS_CONFIG["stats_path"])
    os.makedirs(save_dir, exist_ok=True)
    utils.save_stats(config.PREPROCESS_CONFIG["stats_path"], stats)
    print(f"\n統計量を保存しました: {config.PREPROCESS_CONFIG['stats_path']}")

if __name__ == '__main__':
    main()