# 📁 model.py
import torch
import torch.nn as nn
import config

class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout_rate):
        super().__init__()
        
        self.bilstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.attention1 = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.layernorm1 = nn.LayerNorm(hidden_size * 2)
        
        self.bilstm2 = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.layernorm2 = nn.LayerNorm(hidden_size * 2)
        
        self.bilstm3 = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        
        self.dense_out = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        # x の形状: (batch, sequence_length, num_features) = (batch, 512, 2)
        
        x, _ = self.bilstm1(x)
        attention_output, _ = self.attention1(query=x, key=x, value=x)
        x = self.layernorm1(x + attention_output)
        
        x, _ = self.bilstm2(x)
        attention_output, _ = self.attention2(query=x, key=x, value=x)
        x = self.layernorm2(x + attention_output)
        
        x, _ = self.bilstm3(x)
        output = self.dense_out(x)
        
        return output

def build_model():
    """設定に基づいてモデルを構築して返す"""
    model = AttentionBiLSTM(
        input_size=config.PREPROCESS_CONFIG["num_features"],
        hidden_size=config.MODEL_CONFIG["hidden_size"],
        num_heads=config.MODEL_CONFIG["num_attention_heads"],
        dropout_rate=config.MODEL_CONFIG["dropout_rate"]
    )
    return model