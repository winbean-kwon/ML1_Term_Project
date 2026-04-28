"""
ARIMA, LSTM, Transformer 모델 학습 및 비교

각 모델을 학습하고 예측 결과를 저장합니다.
"""
import numpy as np
import pandas as pd
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import DATA_DIR, MODEL_DIR, SEED, WINDOW_SIZE

np.random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────

def load_dataset():
    data = np.load(os.path.join(DATA_DIR, "dataset.npz"))
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]


# ──────────────────────────────────────
# 1. ARIMA (로그수익률 기반, 종목별 개별 학습)
# ──────────────────────────────────────

def train_arima(X_train, X_test, y_train, y_test):
    """
    ARIMA는 시계열 회귀 모델이므로 로그수익률을 예측한 뒤
    부호로 방향(상승/하락)을 분류합니다.
    TODO: 종목별로 분리하여 학습하도록 확장
    """
    from statsmodels.tsa.arima.model import ARIMA

    # 간단 버전: 마지막 윈도우의 log_return(첫 번째 피처)만 사용
    log_return_idx = 0
    train_series = X_train[:, -1, log_return_idx]  # 각 샘플의 마지막 시점 로그수익률

    # ARIMA(5,1,0) 학습
    try:
        model = ARIMA(train_series, order=(5, 1, 0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(y_test))
        preds = (forecast > 0).astype(int)
    except Exception as e:
        print(f"ARIMA 학습 실패: {e}")
        preds = np.zeros(len(y_test), dtype=int)

    return preds


# ──────────────────────────────────────
# 2. LSTM
# ──────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


def train_lstm(X_train, X_test, y_train, y_test,
               epochs=20, batch_size=256, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[2]

    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = LSTMClassifier(input_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  LSTM Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # 예측
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test).to(device))
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)

    # 모델 저장
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm.pt"))
    return preds


# ──────────────────────────────────────
# 3. Transformer
# ──────────────────────────────────────

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x[:, -1, :]  # 마지막 시점
        return self.fc(x).squeeze(-1)


def train_transformer(X_train, X_test, y_train, y_test,
                      epochs=20, batch_size=256, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[2]

    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = TransformerClassifier(input_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Transformer Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test).to(device))
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "transformer.pt"))
    return preds


# ──────────────────────────────────────
# 메인
# ──────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test = load_dataset()
    print(f"데이터 로드: Train={X_train.shape}, Test={X_test.shape}")

    results = {}

    print("\n[1/3] ARIMA 학습...")
    results["arima"] = train_arima(X_train, X_test, y_train, y_test)

    print("\n[2/3] LSTM 학습...")
    results["lstm"] = train_lstm(X_train, X_test, y_train, y_test)

    print("\n[3/3] Transformer 학습...")
    results["transformer"] = train_transformer(X_train, X_test, y_train, y_test)

    # 예측 결과 저장
    pred_path = os.path.join(DATA_DIR, "predictions.npz")
    np.savez(pred_path, y_test=y_test, **results)
    print(f"\n예측 결과 저장: {pred_path}")


if __name__ == "__main__":
    main()
