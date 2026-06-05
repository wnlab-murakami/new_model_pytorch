# 📁 inference.py
import torch
import numpy as np
import h5py
import os
from tqdm import tqdm

import config
import utils
import model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"推論デバイス: {device}")

    # --- モデル読み込み ---
    net = model.build_model()
    model_path = config.INFERENCE_CONFIG["trained_model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_path}")

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    print(f"モデルを読み込みました: {model_path}")

    # --- 正規化設定 ---
    norm_method = config.PREPROCESS_CONFIG["normalization_method"]
    print(f"正規化手法: {norm_method}")
    if norm_method == "standardization":
        stats = utils.load_stats(config.PREPROCESS_CONFIG["stats_path"])

    # --- 出力先の準備 ---
    os.makedirs(config.INFERENCE_CONFIG["output_hdf5_path"], exist_ok=True)

    # --- HDF5テストデータの読み込みと推論 ---
    hdf5_path = config.DATA_CONFIG["hdf5_test_path"]  # テスト用HDF5

    with h5py.File(hdf5_path, 'r') as f:
        num_range_bins = f['input_real'].shape[0]   # 512
        num_chirps     = f['input_real'].shape[1]   # 128
        num_scenarios  = f['input_real'].shape[2]   # 100

        print(f"\nテストデータ: {num_scenarios}シナリオ × {num_chirps}チャープ")

        with torch.no_grad():
            for s in tqdm(range(num_scenarios), desc="シナリオ処理中"):

                # シナリオ単位で読み込み: (512, 128)
                real_data = f['input_real'][:, :, s]  # (512, 128)
                imag_data = f['input_imag'][:, :, s]  # (512, 128)

                # チャープを行として扱う: (128, 512)
                real_data = real_data.T  # (chirps, range_bins)
                imag_data = imag_data.T  # (chirps, range_bins)

                # (128, 512, 2) に整形
                input_channels = np.stack(
                    [real_data, imag_data], axis=-1
                ).astype(np.float32)

                # --- 前処理 ---
                if norm_method == "max_abs_scaling":
                    processed_input, max_abs_values = \
                        utils.max_abs_normalize_complex_channels(input_channels)

                elif norm_method == "standardization":
                    processed_input = utils.standardize_data(input_channels, stats)

                else:
                    processed_input = input_channels
                    max_abs_values = None

                # --- 推論 ---
                input_tensor  = torch.from_numpy(processed_input).to(device)
                output_tensor = net(input_tensor)               # (128, 512, 2)
                output_np     = output_tensor.cpu().numpy()

                # --- 後処理 ---
                if norm_method == "max_abs_scaling":
                    output_real_imag = utils.max_abs_denormalize_complex_channels(
                        output_np, max_abs_values
                    )
                elif norm_method == "standardization":
                    output_real_imag = utils.destandardize_data(output_np, stats)
                else:
                    output_real_imag = output_np

                # --- 複素数に再構成: (128, 512) → 転置して (512, 128) ---
                suppressed = (output_real_imag[..., 0] + 
                              1j * output_real_imag[..., 1]).T  # (512, 128)

                # --- .hdf5保存 ---
                output_path = os.path.join(
                    config.INFERENCE_CONFIG["output_hdf5_path"],
                    f"suppressed_scenario_{s:04d}.hdf5"
                )
                utils.save_as_hdf5(output_path, {"suppressed_signal_complex": suppressed})

    print(f"\n推論完了: {config.INFERENCE_CONFIG['output_hdf5_path']}")

if __name__ == '__main__':
    main()