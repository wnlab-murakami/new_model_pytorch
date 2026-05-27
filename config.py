# 📁 config.py
# すべての設定を管理するファイル

import os
import datetime

# --- 基本設定 ---
DT_NOW = datetime.datetime.now()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- データ設定 ---
DATA_CONFIG = {
    "learning_data_path": os.path.join(BASE_DIR, "learning_data/2025_6_20_23"),
    "test_data_path": os.path.join(BASE_DIR, "test_data/2025_6_21_22"),
    "input_dir_name": "input",
    "label_dir_name": "label",
    "real_dir_name": "real",
    "imag_dir_name": "imag",
    "file_pattern": "real_input_*.txt", # 実部ファイルを基準にファイルを検索
}

# --- 前処理・正規化設定 ---
PREPROCESS_CONFIG = {
    # 'standardization' または 'max_abs_scaling' を選択
    "normalization_method": "max_abs_scaling", 
    
    # 'standardization' を使用する場合のみ必要
    "stats_path": os.path.join(BASE_DIR, "saved_models_pytorch", "complex_norm_stats.pkl"),

    "sequence_length": 512,
    "num_features": 2, # 2チャンネル (実部, 虚部)
}

# --- モデル設定 ---
MODEL_CONFIG = {
    "hidden_size": 256,
    "num_attention_heads": 4,
    "dropout_rate": 0.1,
}

# --- 学習設定 ---
TRAIN_CONFIG = {
    "epochs": 150,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "validation_split": 0.2,
    "model_save_path": os.path.join(BASE_DIR, "saved_models_pytorch", f"Complex_Attention_biLSTM_{DT_NOW:%Y%m%d_%H%M}"),
}

# --- 推論設定 ---
INFERENCE_CONFIG = {
    "trained_model_path": os.path.join(TRAIN_CONFIG["model_save_path"], "best_model.pth"),
    "output_mat_path": os.path.join(BASE_DIR, "inference_results_pytorch", f"complex_{DT_NOW:%Y%m%d_%H%M}"),
}