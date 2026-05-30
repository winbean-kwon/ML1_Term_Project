"""
LSTM 모델 성능 개선 (Google Colab 실행용)
- BiLSTM + Temporal Attention
- BCEWithLogitsLoss + pos_weight (Focal Loss 대비 안정적, class collapse 없음)
- AdamW + CosineAnnealingWarmRestarts
- Stratified subsampling으로 학습 속도 대폭 단축
- Val set 기반 최적 threshold 탐색

[Colab 사용법]
1. Google Drive에 term_project 폴더 업로드
2. 이 파일 전체를 Colab 셀에 붙여넣거나
   !python 05_train_lstm_improved.py 로 실행
"""

# ──────────────────────────────────────
# [STEP 1] Google Drive 마운트 & 경로 설정
# ──────────────────────────────────────

import os, sys

DRIVE_PROJECT_PATH = "/content/drive/MyDrive/term_project"

try:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR  = DRIVE_PROJECT_PATH
    DATA_DIR  = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    print(f"[Drive 마운트 완료] BASE_DIR={BASE_DIR}")
except Exception:
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR  = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    print(f"[로컬 환경] BASE_DIR={BASE_DIR}")

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42

# ──────────────────────────────────────
# [STEP 2] 라이브러리 임포트
# ──────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"PyTorch version : {torch.__version__}")

# CUDA > MPS(Apple Silicon) > CPU 순 자동 감지
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device          : {DEVICE}")


# ──────────────────────────────────────
# [STEP 3] 데이터 로드
# ──────────────────────────────────────

def load_dataset():
    path = os.path.join(DATA_DIR, "dataset.npz")
    assert os.path.exists(path), f"dataset.npz 가 없습니다: {path}"
    data = np.load(path)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]


# ──────────────────────────────────────
# [STEP 4] 균형 서브샘플링
# 학습 데이터가 수백만 건인 경우, 클래스 균형을 맞춘 서브셋만 사용.
# 정확도는 거의 동일하게 유지되며 학습 시간이 5~10배 단축됨.
# ──────────────────────────────────────

def balanced_subsample(X, y, sample_ratio=0.2, rng=None):
    if rng is None:
        rng = np.random.RandomState(SEED)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    n_minor = min(len(idx0), len(idx1))
    n_take  = max(1, int(n_minor * sample_ratio))

    chosen0 = rng.choice(idx0, n_take, replace=False)
    chosen1 = rng.choice(idx1, n_take, replace=False)
    chosen  = np.concatenate([chosen0, chosen1])
    rng.shuffle(chosen)

    print(f"  서브샘플: Down={len(chosen0):,}  Up={len(chosen1):,}  합계={len(chosen):,}")
    return X[chosen], y[chosen]


# ──────────────────────────────────────
# [STEP 5] 모델: BiLSTM + Temporal Attention
# ──────────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out):
        scores  = self.attn(lstm_out).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (weights * lstm_out).sum(dim=1)


class ImprovedLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = TemporalAttention(hidden_size)
        self.norm    = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2     = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        context = self.norm(context)
        context = self.dropout(context)
        out = F.gelu(self.fc1(context))
        out = self.dropout(out)
        return self.fc2(out).squeeze(-1)


# ──────────────────────────────────────
# [STEP 6] Val set 기반 최적 threshold 탐색
# ──────────────────────────────────────

def find_best_threshold(model, X_val, y_val, device):
    model.eval()
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val)),
        batch_size=1024, shuffle=False,
    )
    probs_list = []
    with torch.no_grad():
        for (xb,) in val_loader:
            probs = torch.sigmoid(model(xb.to(device))).cpu().numpy()
            probs_list.append(probs)
    probs = np.concatenate(probs_list)

    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.3, 0.71, 0.02):
        preds = (probs > thr).astype(int)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    print(f"  최적 threshold={best_thr:.2f}  val macro-F1={best_f1:.4f}")
    return best_thr


# ──────────────────────────────────────
# [STEP 7] 학습
# ──────────────────────────────────────

def train_lstm_improved(
    X_train, X_test, y_train, y_test,
    sample_ratio=0.2,
    epochs=30,
    batch_size=512,
    lr=3e-4,
    patience=7,
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
):
    print(f"\n  원본 학습 데이터: {len(X_train):,}  테스트: {len(X_test):,}")

    # 균형 서브샘플링
    X_sub, y_sub = balanced_subsample(X_train, y_train, sample_ratio=sample_ratio)

    # train / val 분리 (85 / 15)
    n_val = max(1, int(len(X_sub) * 0.15))
    X_tr, X_val = X_sub[:-n_val], X_sub[-n_val:]
    y_tr, y_val = y_sub[:-n_val], y_sub[-n_val:]

    print(f"  학습={len(X_tr):,}  검증={len(X_val):,}  Up비율={y_tr.mean():.3f}")

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(DEVICE.type != "cpu"),
    )
    val_loss_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size * 2, shuffle=False,
    )

    input_size = X_train.shape[2]
    model = ImprovedLSTMClassifier(
        input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
    ).to(DEVICE)

    # pos_weight 만으로 클래스 불균형 처리 (Focal Loss 없음 → class collapse 방지)
    pos_w = torch.tensor(
        [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loss_loader:
                val_loss += criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
        val_loss /= len(val_loss_loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1:3d}/{epochs}  "
                f"train={total_loss/len(train_loader):.4f}  "
                f"val={val_loss:.4f}  lr={lr_now:.1e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  → Early stopping at epoch {epoch+1} (best val={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)

    # Val set 기반 최적 threshold 탐색
    threshold = find_best_threshold(model, X_val, y_val, DEVICE)

    # 전체 테스트 예측
    model.eval()
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test)),
        batch_size=batch_size * 2, shuffle=False,
    )
    probs_list = []
    with torch.no_grad():
        for (xb,) in test_loader:
            probs = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            probs_list.append(probs)
    avg_probs = np.concatenate(probs_list)
    preds     = (avg_probs > threshold).astype(int)

    save_path = os.path.join(MODEL_DIR, "lstm_improved.pt")
    torch.save(best_state, save_path)
    print(f"\n  모델 저장: {save_path}")

    return preds


# ──────────────────────────────────────
# [STEP 8] 메인 실행
# ──────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test = load_dataset()
    print(f"\n데이터 로드 완료")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")

    print("\n" + "="*50)
    print("  개선된 LSTM 학습 시작")
    print("="*50)

    preds = train_lstm_improved(X_train, X_test, y_train, y_test)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="weighted")
    f1m = f1_score(y_test, preds, average="macro")

    print(f"\n{'='*50}")
    print("  최종 결과")
    print(f"{'='*50}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  F1 (weighted) : {f1:.4f}")
    print(f"  F1 (macro)    : {f1m:.4f}")
    print(f"  Up 예측 비율  : {preds.mean():.3f}  (실제: {y_test.mean():.3f})")
    print()
    print(classification_report(y_test, preds, target_names=["Down", "Up"]))

    pred_path = os.path.join(DATA_DIR, "predictions.npz")
    if os.path.exists(pred_path):
        existing = dict(np.load(pred_path))
        existing["lstm"] = preds
        np.savez(pred_path, **existing)
    else:
        np.savez(pred_path, lstm=preds, y_test=y_test)
    print(f"predictions.npz 업데이트 완료: {pred_path}")


if __name__ == "__main__":
    main()
