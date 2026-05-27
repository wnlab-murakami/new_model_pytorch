# 📁 inference.py
import torch
import numpy as np
import glob
from natsort import natsorted
import os
from tqdm import tqdm

import config
import utils
import model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"推論デバイス: {device}")
    
    # --- モデルの読み込み ---
    net = model.build_model()
    model_path = config.INFERENCE_CONFIG["trained_model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_path}")

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    print(f"モデルを読み込みました: {model_path}")

    # --- 正規化手法の確認と統計量の読み込み ---
    norm_method = config.PREPROCESS_CONFIG["normalization_method"]
    print(f"使用する正規化手法: {norm_method}")
    if norm_method == "standardization":
        stats = utils.load_stats(config.PREPROCESS_CONFIG["stats_path"])

    # --- テストデータのファイルリストを取得 ---
    test_base_path = config.DATA_CONFIG["test_data_path"]
    real_test_path = os.path.join(test_base_path, config.DATA_CONFIG["input_dir_name"], config.DATA_CONFIG["real_dir_name"])
    imag_test_path = os.path.join(test_base_path, config.DATA_CONFIG["input_dir_name"], config.DATA_CONFIG["imag_dir_name"])
    
    real_test_files = natsorted(glob.glob(os.path.join(real_test_path, "real_*.txt")))
    
    if not real_test_files:
        raise FileNotFoundError(f"テストデータが見つかりません: {real_test_path}")
    os.makedirs(config.INFERENCE_CONFIG["output_mat_path"], exist_ok=True)

    with torch.no_grad():
        for real_file in tqdm(real_test_files, desc="推論処理中"):
            imag_file = real_file.replace(real_test_path, imag_test_path).replace("real_", "imag_")
            if not os.path.exists(imag_file):
                print(f"\nスキップ: ペアとなる虚数部ファイルが見つかりません: {imag_file}")
                continue

            # --- データの読み込み ---
            real_data = utils.load_data_from_txt(real_file) # (16, 512)
            imag_data = utils.load_data_from_txt(imag_file) # (16, 512)
            input_channels = np.stack([real_data, imag_data], axis=-1).astype(np.float32)

            # --- 前処理（正規化/標準化） ---
            if norm_method == "max_abs_scaling":
                processed_input, max_abs_values = utils.max_abs_normalize_complex_channels(input_channels)
            elif norm_method == "standardization":
                processed_input = utils.standardize_data(input_channels, stats)
            else:
                processed_input = input_channels
                max_abs_values = None # 逆変換で使わないことを明示

            input_tensor = torch.from_numpy(processed_input).to(device)

            # --- 推論実行 ---
            output_tensor = net(input_tensor)
            output_processed = output_tensor.cpu().numpy()

            # --- 後処理（逆正規化/逆標準化）---
            if norm_method == "max_abs_scaling":
                output_real_imag = utils.max_abs_denormalize_complex_channels(output_processed, max_abs_values)
            elif norm_method == "standardization":
                output_real_imag = utils.destandardize_data(output_processed, stats)
            else:
                output_real_imag = output_processed
            
            # --- 結果を複素数に再構成 ---
            final_complex_output = output_real_imag[..., 0] + 1j * output_real_imag[..., 1]
            
            # --- .matファイルとして保存 ---
            base_filename = os.path.basename(real_file).replace(".txt", "")
            output_filename = base_filename.replace("real_input", "suppressed_complex") + ".mat"
            output_filepath = os.path.join(config.INFERENCE_CONFIG["output_mat_path"], output_filename)
            utils.save_as_mat(output_filepath, {"suppressed_signal_complex": final_complex_output})

    print(f"\n推論が完了し、結果を {config.INFERENCE_CONFIG['output_mat_path']} に保存しました。")

if __name__ == '__main__':
    main()