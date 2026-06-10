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
    print(f"使用する正規化手法: {norm_method}")
    if norm_method == "standardization":
        stats = utils.load_stats(config.PREPROCESS_CONFIG["stats_path"])

    # --- HDF5テストデータの読み込み ---
    hdf5_path = config.DATA_CONFIG["hdf5_test_path"]
    with h5py.File(hdf5_path, 'r') as f:
        num_scenarios  = f['input_real'].shape[0]
        num_chirps     = f['input_real'].shape[1]
        num_range_bins = f['input_real'].shape[2]

    print(f"テストデータ: {num_scenarios}シナリオ × {num_chirps}チャープ × {num_range_bins}レンジビン")

    # --- 出力先の準備 ---
    os.makedirs(config.INFERENCE_CONFIG["output_hdf5_path"], exist_ok=True)
    output_filepath = os.path.join(
        config.INFERENCE_CONFIG["output_hdf5_path"], "suppressed_results.hdf5"
    )

    with h5py.File(hdf5_path, 'r') as f_in, \
         h5py.File(output_filepath, 'w') as f_out:

        ds_real = f_out.create_dataset(
            "suppressed_real",
            shape=(num_scenarios, num_chirps, num_range_bins),
            dtype=np.float32,
            chunks=(1, num_chirps, num_range_bins),
            compression="gzip", compression_opts=4
        )
        ds_imag = f_out.create_dataset(
            "suppressed_imag",
            shape=(num_scenarios, num_chirps, num_range_bins),
            dtype=np.float32,
            chunks=(1, num_chirps, num_range_bins),
            compression="gzip", compression_opts=4
        )

        f_out.attrs["axes"]           = "scenarios x chirps x range_bins"
        f_out.attrs["num_scenarios"]  = num_scenarios
        f_out.attrs["num_chirps"]     = num_chirps
        f_out.attrs["num_range_bins"] = num_range_bins
        f_out.attrs["source"]         = "inference"
        f_out.attrs["norm_method"]    = norm_method

        with torch.no_grad():
            for s in tqdm(range(num_scenarios), desc="シナリオ処理中"):

                # shape: (num_chirps, num_range_bins) = (128, 512)
                real_data = f_in['input_real'][s, :, :]
                imag_data = f_in['input_imag'][s, :, :]

                # shape: (128, 512, 2)
                input_channels = np.stack(
                    [real_data, imag_data], axis=-1
                ).astype(np.float32)

                # --- 前処理 ---
                if norm_method == "max_abs_scaling":
                    # [変更] シナリオ全体の最大値1つで正規化
                    # → チャープ間のスケールが統一され、縦縞アーティファクトを抑制
                    processed_input, max_abs_val = \
                        utils.max_abs_normalize_scenario(input_channels)
                elif norm_method == "standardization":
                    processed_input = utils.standardize_data(input_channels, stats)
                else:
                    processed_input = input_channels
                    max_abs_val = None

                # --- 推論 ---
                input_tensor  = torch.from_numpy(processed_input).to(device)
                output_tensor = net(input_tensor)   # (128, 512, 2)
                output_np     = output_tensor.cpu().numpy()

                # --- 後処理 ---
                if norm_method == "max_abs_scaling":
                    # [変更] シナリオ単位の逆正規化
                    output_real_imag = utils.max_abs_denormalize_scenario(
                        output_np, max_abs_val
                    )
                elif norm_method == "standardization":
                    output_real_imag = utils.destandardize_data(output_np, stats)
                else:
                    output_real_imag = output_np

                # shape: (128, 512) で保存
                ds_real[s, :, :] = output_real_imag[..., 0]
                ds_imag[s, :, :] = output_real_imag[..., 1]

    print(f"\n推論完了: {output_filepath}")
    print(f"出力構造: suppressed_real / suppressed_imag "
          f"({num_scenarios} x {num_chirps} x {num_range_bins})")

if __name__ == '__main__':
    main()