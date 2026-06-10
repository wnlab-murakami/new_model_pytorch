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
            # MATLABが (range_bins, chirps, scenarios) で保存
            # → transpose → (scenarios, chirps, range_bins)
            self.input_real = f['input_real'][:].transpose(2, 1, 0).astype(np.float32)
            self.input_imag = f['input_imag'][:].transpose(2, 1, 0).astype(np.float32)
            self.label_real = f['label_real'][:].transpose(2, 1, 0).astype(np.float32)
            self.label_imag = f['label_imag'][:].transpose(2, 1, 0).astype(np.float32)

        self.num_scenarios = self.input_real.shape[0]   # シナリオ数
        self.num_chirps    = self.input_real.shape[1]   # チャープ数
        self.total_samples = self.num_scenarios * self.num_chirps
        print(f"読み込み完了: {self.total_samples}サンプル "
              f"({self.num_scenarios}シナリオ × {self.num_chirps}チャープ)")

        # [変更] シナリオ単位のmax_absを事前計算（__init__で1回だけ）
        # 推論時と同じスケールにするため、シナリオ全体の最大値を使用
        if self.norm_method == "max_abs_scaling":
            abs_val = np.sqrt(self.input_real**2 + self.input_imag**2)
            # shape: (scenarios, chirps, range_bins) → 各シナリオの最大値 → (scenarios,)
            self.scenario_max_abs = np.max(abs_val, axis=(1, 2)) + 1e-8
            print(f"シナリオ単位のmax_absを事前計算完了: "
                  f"min={self.scenario_max_abs.min():.4e}, "
                  f"max={self.scenario_max_abs.max():.4e}")
        elif self.norm_method == "standardization":
            self.stats = utils.load_stats(config.PREPROCESS_CONFIG["stats_path"])

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        scenario_idx = idx // self.num_chirps
        chirp_idx    = idx  % self.num_chirps

        # shape: (512,) → (512, 2)
        input_channels = np.stack(
            [self.input_real[scenario_idx, chirp_idx, :],
             self.input_imag[scenario_idx, chirp_idx, :]],
            axis=-1
        )
        label_channels = np.stack(
            [self.label_real[scenario_idx, chirp_idx, :],
             self.label_imag[scenario_idx, chirp_idx, :]],
            axis=-1
        )

        # 正規化
        if self.norm_method == "max_abs_scaling":
            # [変更] シナリオ単位のmax_absで正規化（推論と同じスケール）
            max_val = self.scenario_max_abs[scenario_idx]  # スカラー
            processed_input = input_channels / max_val
            processed_label = label_channels / max_val

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

    num_total      = len(dataset)
    num_validation = max(1, int(num_total * validation_split))
    num_train      = num_total - num_validation

    train_dataset, valid_dataset = random_split(dataset, [num_train, num_validation])

    # 全データをメモリ展開済みのためnum_workers>0が使用可能
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n総サンプル数: {num_total}  "
          f"(シナリオ×チャープ = {dataset.num_scenarios} × {dataset.num_chirps})")
    print(f"学習: {len(train_dataset)}, 検証: {len(valid_dataset)}")
    print(f"正規化手法: {config.PREPROCESS_CONFIG['normalization_method']}")

    return train_loader, valid_loader