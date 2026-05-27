# 📁 train.py
# PyTorchモデルの学習を実行するメインスクリプト

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import wandb  # wandbをインポート

# 各モジュールをインポート
import config
import data_loader
import model

def main():
    # --- 1. wandbの初期化 ---
    # 設定ファイルからハイパーパラメータを読み込み、wandbに渡す
    hyperparameters = {
        "learning_rate": config.TRAIN_CONFIG["learning_rate"],
        "batch_size": config.TRAIN_CONFIG["batch_size"],
        "epochs": config.TRAIN_CONFIG["epochs"],
        "architecture": "AttentionBiLSTM (PyTorch)",
        "hidden_size": config.MODEL_CONFIG["hidden_size"],
        "num_attention_heads": config.MODEL_CONFIG["num_attention_heads"],
        "dropout_rate": config.MODEL_CONFIG["dropout_rate"]
    }
    wandb.init(
        project="Radar-Interference-Suppression", # wandb上のプロジェクト名
        name=f"run_{config.DT_NOW:%Y%m%d_%H%M%S}", # 各実験の実行名
        config=hyperparameters
    )

    # --- 2. デバイス、データローダー、モデルの準備  ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用するデバイス: {device}")

    train_loader, valid_loader = data_loader.get_dataloaders(
        batch_size=config.TRAIN_CONFIG["batch_size"],
        validation_split=config.TRAIN_CONFIG["validation_split"]
    )

    net = model.build_model().to(device)

    def init_weights(m):
            # もし層がLinear（全結合層）なら、これまで通り初期化
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                    
            # もし層がLSTMなら、内部の全パラメータを個別に処理
            elif isinstance(m, nn.LSTM):
                # LSTM層は複数の重みとバイアスを持つため、ループで処理する
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        # 'weight'という名前を含むパラメータをXavier初期化
                        torch.nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        # 'bias'という名前を含むパラメータをゼロで初期化
                        torch.nn.init.zeros_(param)

    # モデルの全レイヤーに上記の初期化関数を適用する
    net.apply(init_weights)
    print("モデルの重みをXavier初期化法で初期化しました。")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config.TRAIN_CONFIG["learning_rate"], eps=1e-6)
    
    # wandbにモデルの勾配や構造を監視させる (オプション)
    wandb.watch(net, log="all", log_freq=100)

    # --- 3. 学習ループ  ---
    best_valid_loss = float('inf')
    os.makedirs(config.TRAIN_CONFIG["model_save_path"], exist_ok=True)
    
    print("\n学習を開始します...")
    for epoch in range(config.TRAIN_CONFIG["epochs"]):
        # --- 学習フェーズ ---
        net.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.TRAIN_CONFIG['epochs']} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)

        # --- 検証フェーズ ---
        net.eval()
        valid_loss = 0.0
        with torch.no_grad():
            valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{config.TRAIN_CONFIG['epochs']} [Valid]")
            for inputs, labels in valid_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                valid_pbar.set_postfix({'loss': loss.item()})

        avg_valid_loss = valid_loss / len(valid_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Valid Loss = {avg_valid_loss:.6f}")

        # --- 4. wandbにメトリクスをロギング ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "valid_loss": avg_valid_loss
        })

        # 最良モデルの保存 
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            model_path = os.path.join(config.TRAIN_CONFIG["model_save_path"], "best_model.pth")
            torch.save(net.state_dict(), model_path)
            # wandbに最高のスコアを記録
            wandb.summary["best_validation_loss"] = best_valid_loss
            print(f"モデルを保存しました: {model_path} (Valid Loss: {best_valid_loss:.6f})")

    # --- 5. wandbの実行を終了 ---
    wandb.finish()
    print("学習が完了しました。")

if __name__ == '__main__':
    main()