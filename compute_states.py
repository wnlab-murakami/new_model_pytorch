# 📁 compute_stats.py
# 学習データ全体(複素数)の実部と虚部の平均と標準偏差を計算するスクリプト

import numpy as np
import glob
from natsort import natsorted
import os
from tqdm import tqdm

import config
import utils

def main():
    """学習データ全体(複素数)の実部と虚部の平均と標準偏差を計算する"""
    input_base_path = os.path.join(config.DATA_CONFIG["learning_data_path"], config.DATA_CONFIG["input_dir_name"])
    real_path = os.path.join(input_base_path, config.DATA_CONFIG["real_dir_name"])
    
    file_pattern = os.path.join(real_path, config.DATA_CONFIG["file_pattern"].replace("input", "real_input"))
    real_files = natsorted(glob.glob(file_pattern))

    if not real_files:
        raise FileNotFoundError(f"統計量計算用の学習データが見つかりませんでした。パスを確認してください: {file_pattern}")

    all_real_parts, all_imag_parts = [], []
    print("全学習データ（複素数）を読み込み中...")
    
    for real_file in tqdm(real_files):
        imag_file = real_file.replace(config.DATA_CONFIG["real_dir_name"], config.DATA_CONFIG["imag_dir_name"]).replace("real_", "imag_")
        if not os.path.exists(imag_file):
            print(f"警告: ペアとなる虚数部ファイルが見つかりません: {imag_file}")
            continue
        
        real_data = utils.load_data_from_txt(real_file)
        imag_data = utils.load_data_from_txt(imag_file)
        
        all_real_parts.append(real_data)
        all_imag_parts.append(imag_data)

    # 全データを巨大なNumPy配列に結合
    all_real_parts = np.concatenate(all_real_parts, axis=0)
    all_imag_parts = np.concatenate(all_imag_parts, axis=0)

    # 実部と虚部から複素数を作成
    complex_data = all_real_parts + 1j * all_imag_parts

    # 振幅をdBスケールに変換（20*log10(|z|)）
    amplitude = np.abs(complex_data)
    amplitude_db = 10 * np.log10(amplitude)

    print(f"\n複素数データのdBスケール変換例（先頭5つ）: {amplitude_db[:5]}")

    # 各々の平均と標準偏差を計算
    mean_real = np.mean(all_real_parts)
    std_real = np.std(all_real_parts)
    mean_imag = np.mean(all_imag_parts)
    std_imag = np.std(all_imag_parts)
    
    print(f"\n計算された統計量:")
    print(f"  実部 平均: {mean_real:.6f}, 標準偏差: {std_real:.6f}")
    print(f"  虚部 平均: {mean_imag:.6f}, 標準偏差: {std_imag:.6f}")

    # 辞書として保存
    stats = {
        'mean': np.array([mean_real, mean_imag], dtype=np.float32), 
        'std': np.array([std_real, std_imag], dtype=np.float32)
    }
    
    save_dir = os.path.dirname(config.PREPROCESS_CONFIG["stats_path"])
    os.makedirs(save_dir, exist_ok=True)
    utils.save_stats(config.PREPROCESS_CONFIG["stats_path"], stats)
    
    print(f"\n統計量をファイルに保存しました: {config.PREPROCESS_CONFIG['stats_path']}")

if __name__ == '__main__':
    main()