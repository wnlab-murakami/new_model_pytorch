# 📁 utils.py
# 共通のヘルパー関数を定義するファイル

import numpy as np
import scipy.io
import pickle
import os

def load_data_from_txt(filepath):
    """(単一の)テキストファイルからデータを読み込み、NumPy配列として返す"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # 空白で終わる行がある場合を考慮
    data = [list(map(float, line.strip().split())) for line in lines if line.strip()]
    return np.array(data, dtype=np.float32)

# --- 標準化 (Z-score) 関連の関数 ---
def standardize_data(data, stats):
    """データをチャンネルごとに標準化する (z-score)"""
    mean_vec = stats['mean']
    std_vec = stats['std']
    epsilon = 1e-8
    return (data - mean_vec) / (std_vec + epsilon)

def destandardize_data(standardized_data, stats):
    """標準化されたデータを元のスケールに戻す"""
    mean_vec = stats['mean']
    std_vec = stats['std']
    return standardized_data * std_vec + mean_vec

def save_stats(filepath, stats_dict):
    """統計量辞書をpickleファイルとして保存する"""
    with open(filepath, 'wb') as f:
        pickle.dump(stats_dict, f)

def load_stats(filepath):
    """pickleファイルから統計量辞書を読み込む"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"統計量ファイルが見つかりません: {filepath}\n先に `compute_stats.py` を実行してください。")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# --- 最大絶対値正規化 関連の関数 (新規追加) ---
def max_abs_normalize_complex_channels(data_channels):
    """
    複素数データ（2チャンネル形式）を行ごと（サンプルごと）に最大絶対値で正規化する。
    Args:
        data_channels (np.ndarray): 形状が (..., sequence_length, 2) のNumpy配列。
                                    最後の次元は [実部, 虚部]。
    Returns:
        np.ndarray: 正規化されたデータ。形状は入力と同じ。
        np.ndarray: 各サンプル（行）の最大絶対値。逆正規化に使用。形状は (..., 1, 1)。
    """
    # 実部と虚部を取得
    real_part = data_channels[..., 0]
    imag_part = data_channels[..., 1]
    
    # 複素数の絶対値を計算
    abs_val = np.sqrt(real_part**2 + imag_part**2)
    
    # 各サンプル（行）の最大絶対値を見つける (..., sequence_length) -> (..., 1)
    max_abs = np.max(abs_val, axis=-1, keepdims=True)
    
    # ブロードキャストのため次元を拡張 (..., 1) -> (..., 1, 1)
    max_abs_reshaped = np.expand_dims(max_abs, axis=-1)
    
    normalized_data = data_channels / (max_abs_reshaped)
    
    return normalized_data, max_abs_reshaped

def max_abs_denormalize_complex_channels(normalized_data, max_abs_values):
    """
    最大絶対値で正規化された複素数データ（2チャンネル形式）を元のスケールに戻す。
    Args:
        normalized_data (np.ndarray): 正規化されたデータ。形状 (..., sequence_length, 2)。
        max_abs_values (np.ndarray): 正規化に使用した最大絶対値。形状 (..., 1, 1)。
    Returns:
        np.ndarray: 元のスケールに戻されたデータ。
    """
    return normalized_data * max_abs_values

# --- ファイル保存関連 ---
def save_as_mat(filepath, data_dict):
    """データを.matファイルとして保存する"""
    scipy.io.savemat(filepath, data_dict)
    print(f"ファイルを保存しました: {filepath}")