# 📁 train.py
# PyTorchモデルの学習を実行するメインスクリプト

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
import wandb  # wandbをインポート

# 各モジュールをインポート
import config
import data_loader
import model

def main():
    # --- 1. wandbの初期化 ---
    hyperparameters = {
        "learning_rate": config.TRAIN_CONFIG["learning_rate"],
        "batch_size": config.TRAIN_CONFIG["batch_size"],
        "epochs": config.TRAIN_CONFIG["epochs"],
        "architecture": "AttentionBiLSTM (PyTorch)",
        "hidden_size": config.MODEL_CONFIG["hidden_size"],
        "num_attention_heads": config.MODEL_CONFIG["num_attention_heads"],
        "dropout_rate": config.MODEL_CONFIG["dropout_rate"],
        # [追加] schedulerのパラメータも記録
        "lr_scheduler": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 10,
        "grad_clip_max_norm": 1.0,
    }
    wandb.init(
        project="Radar-Interference-Suppression",
        name=f"run_{config.DT_NOW:%Y%m%d_%H%M%S}",
        config=hyperparameters
    )

    # --- 2. デバイス、データローダー、モデルの準備 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用するデバイス: {device}")

    train_loader, valid_loader = data_loader.get_dataloaders(
        batch_size=config.TRAIN_CONFIG["batch_size"],
        validation_split=config.TRAIN_CONFIG["validation_split"]
    )

    net = model.build_model().to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)

    net.apply(init_weights)
    print("モデルの重みをXavier初期化法で初期化しました。")

    class ComplexMSELoss(nn.Module):
        """
        実部・虚部を直接比較するMSEに
        dBスケールの振幅損失を加算して弱信号も学習させる。
        """
        def __init__(self, db_weight=0.5, eps=1e-10):
            super().__init__()
            self.db_weight = db_weight
            self.eps = eps

        def forward(self, pred, target):
            mse = F.mse_loss(pred, target)
            pred_amp   = torch.sqrt(pred[...,0]**2 + pred[...,1]**2 + self.eps)
            target_amp = torch.sqrt(target[...,0]**2 + target[...,1]**2 + self.eps)
            db_loss = F.mse_loss(torch.log10(pred_amp), torch.log10(target_amp))
            return mse + self.db_weight * db_loss

    criterion = ComplexMSELoss(db_weight=0.1)
    optimizer = optim.Adam(net.parameters(), lr=config.TRAIN_CONFIG["learning_rate"], eps=1e-6)

    # 学習率スケジューラ
    # valid_lossがpatience=10エポック改善しなければLRを0.5倍に下げる
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True,   # LR変更時にコンソールに表示
        min_lr=1e-7,    # これ以上は下げない下限
    )

    wandb.watch(net, log="all", log_freq=100)

    # --- 3. 学習ループ ---
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

            # 勾配クリッピング（勾配爆発によるloss振動を抑制）
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

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

        # [追加①] スケジューラをvalid_lossで更新
        scheduler.step(avg_valid_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Valid Loss = {avg_valid_loss:.6f}, LR = {current_lr:.2e}")

        # --- 4. wandbにメトリクスをロギング ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "valid_loss": avg_valid_loss,
            "learning_rate": current_lr,  # [追加③] LRの推移をwandbで可視化
        })

        # 最良モデルの保存
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            model_path = os.path.join(config.TRAIN_CONFIG["model_save_path"], "best_model.pth")
            torch.save(net.state_dict(), model_path)
            wandb.summary["best_validation_loss"] = best_valid_loss
            print(f"モデルを保存しました: {model_path} (Valid Loss: {best_valid_loss:.6f})")

    # --- 5. wandbの実行を終了 ---
    wandb.finish()
    print("学習が完了しました。")

if __name__ == '__main__':
    main()