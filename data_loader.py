# 📁 data_loader.py
# PyTorchのDatasetとDataLoaderを作成する

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import glob
from natsort import natsorted
import os

import config
import utils

class RadarDataset(Dataset):
    """
    複素数レーダーデータ用のカスタムデータセットクラス。
    入力とラベルのファイルパスをペアで扱い、動的にデータを読み込んで前処理を行う。
    """
    def __init__(self, real_input_files, imag_input_files, real_label_files, imag_label_files):
        self.real_input_files = real_input_files
        self.imag_input_files = imag_input_files
        self.real_label_files = real_label_files
        self.imag_label_files = imag_label_files
        
        self.norm_method = config.PREPROCESS_CONFIG["normalization_method"]
        
        # 標準化の場合のみ、最初に一度だけ統計量を読み込む
        if self.norm_method == "standardization":
            self.stats = utils.load_stats(config.PREPROCESS_CONFIG["stats_path"])
        else:
            self.stats = None

        # 1ファイルあたりのチャープ（サンプル）数
        self.chirps_per_file = 16 

    def __len__(self):
        return len(self.real_input_files) * self.chirps_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.chirps_per_file
        chirp_idx = idx % self.chirps_per_file
        
        # --- 該当チャープのデータを読み込み ---
        real_input_all = utils.load_data_from_txt(self.real_input_files[file_idx])
        imag_input_all = utils.load_data_from_txt(self.imag_input_files[file_idx])
        input_channels = np.stack([real_input_all[chirp_idx, :], imag_input_all[chirp_idx, :]], axis=-1).astype(np.float32)

        real_label_all = utils.load_data_from_txt(self.real_label_files[file_idx])
        imag_label_all = utils.load_data_from_txt(self.imag_label_files[file_idx])
        label_channels = np.stack([real_label_all[chirp_idx, :], imag_label_all[chirp_idx, :]], axis=-1).astype(np.float32)

        # --- 前処理 ---
        if self.norm_method == "max_abs_scaling":
            # バッチ次元を追加して正規化関数に渡す
            input_channels_batch = np.expand_dims(input_channels, axis=0)
            normalized_input, max_abs_val = utils.max_abs_normalize_complex_channels(input_channels_batch)
            
            # ラベルも同じ最大絶対値で正規化 (ゼロ除算防止のためepsilonを追加)
            normalized_label = label_channels / (max_abs_val.squeeze(0))
            
            processed_input = normalized_input.squeeze(0)
            processed_label = normalized_label

        elif self.norm_method == "standardization":
            processed_input = utils.standardize_data(input_channels, self.stats)
            processed_label = utils.standardize_data(label_channels, self.stats)
        
        else: # 正規化しない場合
            processed_input = input_channels
            processed_label = label_channels

        return torch.from_numpy(processed_input.copy()), torch.from_numpy(processed_label.copy())

def get_dataloaders(batch_size, validation_split):
    """学習用と検証用のDataLoaderを返す"""
    
    # --- ファイルパスの構築 ---
    base_path = config.DATA_CONFIG["learning_data_path"]
    input_real_path = os.path.join(base_path, config.DATA_CONFIG["input_dir_name"], config.DATA_CONFIG["real_dir_name"])
    input_imag_path = os.path.join(base_path, config.DATA_CONFIG["input_dir_name"], config.DATA_CONFIG["imag_dir_name"])
    label_real_path = os.path.join(base_path, config.DATA_CONFIG["label_dir_name"], config.DATA_CONFIG["real_dir_name"])
    label_imag_path = os.path.join(base_path, config.DATA_CONFIG["label_dir_name"], config.DATA_CONFIG["imag_dir_name"])

    # 基準となる入力の実部ファイルを取得
    real_input_files = natsorted(glob.glob(os.path.join(input_real_path, "real_input_*.txt")))

    # 他のファイルパスを生成
    imag_input_files = [f.replace(input_real_path, input_imag_path).replace("real_", "imag_") for f in real_input_files]
    real_label_files = [f.replace(input_real_path, label_real_path).replace("input", "label") for f in real_input_files]
    imag_label_files = [f.replace(input_real_path, label_imag_path).replace("input", "label").replace("real_", "imag_") for f in real_input_files]

    if not real_input_files:
        raise FileNotFoundError(f"学習データが見つかりませんでした: {input_real_path}")
    if not (len(real_input_files) == len(imag_input_files) == len(real_label_files) == len(imag_label_files)):
        raise ValueError("InputとLabelのファイル数が一致しません。パスの命名規則を確認してください。")

    # --- Datasetの作成 ---
    full_dataset = RadarDataset(
        real_input_files=real_input_files,
        imag_input_files=imag_input_files,
        real_label_files=real_label_files,
        imag_label_files=imag_label_files
    )

    # --- データセットの分割 ---
    num_data = len(full_dataset)
    num_validation = max(1, int(num_data * validation_split))
    num_train = num_data - num_validation

    if num_train <= 0:
        raise ValueError("データセットが小さすぎるため、学習データがありません。")

    train_dataset, valid_dataset = random_split(full_dataset, [num_train, num_validation])
    
    # --- DataLoaderの作成 ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"\n総サンプル数: {num_data}, 学習データ: {len(train_dataset)}, 検証データ: {len(valid_dataset)}")
    print(f"正規化手法: {config.PREPROCESS_CONFIG['normalization_method']}")
    return train_loader, valid_loader