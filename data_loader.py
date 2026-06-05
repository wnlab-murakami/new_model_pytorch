# 📁 data_loader.py

import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

import config
import utils


class HDF5RadarDataset(Dataset):
    def __init__(self, hdf5_path):
        self.norm_method = config.PREPROCESS_CONFIG["normalization_method"]

        # 初期化時に全データをメモリに展開（I/Oは1回だけ）
        print("HDF5ファイルをメモリに読み込み中...")
        with h5py.File(hdf5_path, 'r') as f:
            # shape: (512, 128, 100) → transpose → (100, 128, 512)
            self.input_real = f['input_real'][:].transpose(2, 1, 0)
            self.input_imag = f['input_imag'][:].transpose(2, 1, 0)
            self.label_real = f['label_real'][:].transpose(2, 1, 0)
            self.label_imag = f['label_imag'][:].transpose(2, 1, 0)

        self.num_scenarios  = self.input_real.shape[0]  # 100
        self.num_chirps     = self.input_real.shape[1]  # 128
        self.total_samples  = self.num_scenarios * self.num_chirps  # 12,800
        print(f"読み込み完了: {self.total_samples}サンプル ({self.num_scenarios}シナリオ × {self.num_chirps}チャープ)")

        if self.norm_method == "standardization":
            self.stats = utils.load_stats(config.PREPROCESS_CONFIG["stats_path"])
        else:
            self.stats = None

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        scenario_idx = idx // self.num_chirps
        chirp_idx    = idx  % self.num_chirps

        # メモリから直接取得（ファイルI/Oなし）
        real_input = self.input_real[scenario_idx, chirp_idx, :]  # (512,)
        imag_input = self.input_imag[scenario_idx, chirp_idx, :]
        real_label = self.label_real[scenario_idx, chirp_idx, :]
        imag_label = self.label_imag[scenario_idx, chirp_idx, :]

        input_channels = np.stack([real_input, imag_input], axis=-1).astype(np.float32)
        label_channels = np.stack([real_label, imag_label], axis=-1).astype(np.float32)

        # 正規化（変更なし）
        if self.norm_method == "max_abs_scaling":
            input_batch = np.expand_dims(input_channels, axis=0)
            normalized_input, max_abs_val = utils.max_abs_normalize_complex_channels(input_batch)
            normalized_label = label_channels / (max_abs_val.squeeze(0) + 1e-8 ) # ゼロ除算を防ぐために小さな値を加える
            processed_input  = normalized_input.squeeze(0)
            processed_label  = normalized_label

        elif self.norm_method == "standardization":
            processed_input = utils.standardize_data(input_channels, self.stats)
            processed_label = utils.standardize_data(label_channels, self.stats)

        else:
            processed_input = input_channels
            processed_label = label_channels

        return (torch.from_numpy(processed_input.copy()),
                torch.from_numpy(processed_label.copy()))
    
def get_dataloaders(batch_size, validation_split):
    """学習用と検証用のDataLoaderを返す"""

    dataset = HDF5RadarDataset(config.DATA_CONFIG["hdf5_learning_path"])

    # 分割
    num_total      = len(dataset)
    num_validation = max(1, int(num_total * validation_split))
    num_train      = num_total - num_validation

    train_dataset, valid_dataset = random_split(dataset, [num_train, num_validation])

    # ⚠️ h5pyはマルチプロセス非対応のため num_workers=0 を維持
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n総サンプル数: {num_total}  (シナリオ×チャープ = "
          f"{dataset.num_scenarios} × {dataset.num_chirps})")
    print(f"学習: {len(train_dataset)}, 検証: {len(valid_dataset)}")
    print(f"正規化手法: {config.PREPROCESS_CONFIG['normalization_method']}")

    return train_loader, valid_loader