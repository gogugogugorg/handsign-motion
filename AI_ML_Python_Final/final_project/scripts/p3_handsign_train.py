"""
p3_handsign_train.py
프레임 단위 인 분류 MLP 학습.

실행 (final_project 폴더에서):
    python scripts/p3_handsign_train.py              # 손 + 팔 특징 (기본)
    python scripts/p3_handsign_train.py --no-arm     # 손 특징만 (비교 실험용)

학습 중 증강:
    - 회전 / 크기 / 노이즈 (utils.handsign_features.augment)
    - hand dropout: 일부 프레임에서 한 손을 지움 → 실시간에서 한 손이 안 잡혀도 버티도록

평가:
    - 일반 검증 정확도 (프레임 / 클립 다수결)
    - 한 손 제거 검증: 검증 프레임마다 한 손을 지운 상태의 정확도 (팔 특징의 효과 확인용)

출력:
    checkpoints/handsign_mlp.pth (또는 handsign_mlp_noarm.pth)
        state_dict + labels + use_arm + feature_dim  → 실시간 인식에서 그대로 로드
    data/handsign_dataset/_report/<arm|noarm>/ training_curve.png, confusion_matrix.png, val_report.txt
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils.handsign_features import N_HAND, augment, feature_dim, frame_features  # noqa: E402

DATA_DIR = ROOT / "data" / "handsign_dataset"
CKPT_DIR = ROOT / "checkpoints"

C_TRAIN, C_VAL = "#257de0", "#f47230"
C_TEXT, C_TEXT2, C_GRID = "#191919", "#666666", "#e6e6e6"


class HandSignMLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=256, dropout=0.3):
        super().__init__()
        # 특징 표준화 통계 (학습 데이터에서 계산, 체크포인트에 함께 저장)
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sigma", torch.ones(in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net((x - self.mu) / self.sigma)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-arm", action="store_true", help="팔 특징 없이 손 특징만 사용")
    p.add_argument("--hand-dropout", type=float, default=0.1, help="학습 시 한 손을 지울 프레임 비율")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load(split):
    d = np.load(DATA_DIR / f"{split}.npz", allow_pickle=True)
    return d["norm"], d["mask"], d["arm"], d["y"], d["clip"]


def drop_one_hand(norm, mask, rng, ratio):
    """ratio 비율의 프레임에서 왼손/오른손 중 하나를 무작위로 지운다 (결측 흉내)."""
    norm, mask = norm.copy(), mask.copy()
    idx = np.where(rng.random(len(norm)) < ratio)[0]
    side = rng.integers(0, 2, len(idx))
    for i, s in zip(idx, side):
        norm[i, s * N_HAND:(s + 1) * N_HAND] = 0.0
        mask[i, s] = False
    return norm, mask


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        loss = nn.functional.cross_entropy(logits, torch.from_numpy(y)).item()
        pred = logits.argmax(1).numpy()
    return loss, (pred == y).mean(), pred


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT2, labelsize=8)
    ax.yaxis.grid(True, color=C_GRID)
    ax.set_axisbelow(True)


def plot_curves(hist, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    for ax, key, title in [(axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")]:
        ax.plot(ep, hist[f"train_{key}"], color=C_TRAIN, lw=2, label="Train")
        ax.plot(ep, hist[f"val_{key}"], color=C_VAL, lw=2, label="Validation")
        ax.set_title(title, color=C_TEXT, fontsize=11, loc="left")
        ax.set_xlabel("epoch", color=C_TEXT2, fontsize=9)
        style(ax)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(cm, names, title, path):
    rate = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted", color=C_TEXT2)
    ax.set_ylabel("true", color=C_TEXT2)
    ax.tick_params(colors=C_TEXT2, labelsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, f"{rate[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if rate[i, j] > 0.5 else C_TEXT)
    ax.set_title(title, color=C_TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def class_report(cm, names):
    lines = [f"{'class':8s} {'frames':>7s} {'acc':>6s}  most confused with"]
    for i, n in enumerate(names):
        tot = cm[i].sum()
        row = cm[i].copy()
        row[i] = 0
        conf = f"{names[row.argmax()]} ({row.max() / tot:.0%})" if tot and row.max() else "-"
        lines.append(f"{n:8s} {tot:7d} {cm[i, i] / max(tot, 1):6.1%}  {conf}")
    return lines


def main():
    args = parse_args()
    use_arm = not args.no_arm
    tag = "arm" if use_arm else "noarm"
    model_path = CKPT_DIR / ("handsign_mlp.pth" if use_arm else "handsign_mlp_noarm.pth")
    report_dir = DATA_DIR / "_report" / tag

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    with open(DATA_DIR / "labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    names = [k for k, _ in sorted(labels.items(), key=lambda kv: kv[1])]
    C = len(names)
    D = feature_dim(use_arm)

    tr_norm, tr_mask, tr_arm, y_tr, _ = load("train")
    va_norm, va_mask, va_arm, y_va, clip_va = load("val")
    if not use_arm:
        tr_arm = va_arm = None

    X_tr_clean = frame_features(tr_norm, tr_mask, tr_arm)
    X_va = frame_features(va_norm, va_mask, va_arm)
    # 한 손 제거 검증 세트 (고정 시드)
    dn, dm = drop_one_hand(va_norm, va_mask, np.random.default_rng(0), 1.0)
    X_va_drop = frame_features(dn, dm, va_arm)
    print(f"[{tag}] Train {len(y_tr)} frames | Val {len(y_va)} frames | {C} classes | {D} features")

    model = HandSignMLP(D, C)
    model.mu.copy_(torch.from_numpy(X_tr_clean.mean(0)))
    # 하한 0.1: 손목 z(항상 0)·검출 플래그처럼 분산이 0인 특징을 1e-6 으로 나누면
    # 증강 노이즈가 수만 배로 증폭돼 학습이 깨진다.
    model.sigma.copy_(torch.from_numpy(np.maximum(X_tr_clean.std(0), 0.1)))

    # 클래스 불균형 보정 (none 이 많음)
    counts = np.bincount(y_tr, minlength=C).astype(np.float32)
    weight = torch.from_numpy(counts.sum() / (C * np.maximum(counts, 1)))
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best = (-1.0, None, -1)

    for epoch in range(1, args.epochs + 1):
        # 매 epoch 새로 증강 → 특징 재계산
        n, m, a = tr_norm, tr_mask, tr_arm
        if not args.no_augment:
            n, a = augment(n, m, rng, arm=a)
        if args.hand_dropout > 0:
            n, m = drop_one_hand(n, m, rng, args.hand_dropout)
        X_tr = frame_features(n, m, a)

        model.train()
        perm = rng.permutation(len(y_tr))
        tot_loss, correct = 0.0, 0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = torch.from_numpy(X_tr[idx]), torch.from_numpy(y_tr[idx])
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()

        va_loss, va_acc, _ = evaluate(model, X_va, y_va)
        hist["train_loss"].append(tot_loss / len(y_tr))
        hist["train_acc"].append(correct / len(y_tr))
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)

        if va_acc > best[0]:
            best = (va_acc, {k: v.clone() for k, v in model.state_dict().items()}, epoch)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d} | train loss {hist['train_loss'][-1]:.4f} acc {hist['train_acc'][-1]:.3f} "
                  f"| val loss {va_loss:.4f} acc {va_acc:.3f}")

    # ---- best model ----
    model.load_state_dict(best[1])
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best[1], "labels": names, "use_arm": use_arm, "feature_dim": D},
               model_path)

    _, acc, pred = evaluate(model, X_va, y_va)
    _, acc_drop, pred_drop = evaluate(model, X_va_drop, y_va)
    cm = np.zeros((C, C), int)
    np.add.at(cm, (y_va, pred), 1)
    cm_drop = np.zeros((C, C), int)
    np.add.at(cm_drop, (y_va, pred_drop), 1)

    # 클립 단위 다수결 정확도 (실시간에서 여러 프레임을 모아 판정하는 것과 비슷)
    clip_ok = []
    for c in np.unique(clip_va):
        mm = clip_va == c
        clip_ok.append(np.bincount(pred[mm], minlength=C).argmax() == y_va[mm][0])

    report_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(hist, report_dir / "training_curve.png")
    plot_confusion(cm, names, f"Validation confusion matrix [{tag}]", report_dir / "confusion_matrix.png")
    plot_confusion(cm_drop, names, f"Validation, one hand removed [{tag}]",
                   report_dir / "confusion_matrix_one_hand.png")

    lines = [f"Features: {'hand + arm' if use_arm else 'hand only'} ({D})",
             f"Best epoch: {best[2]}",
             f"Frame accuracy: {acc:.3f}",
             f"Clip accuracy (majority vote): {np.mean(clip_ok):.3f} ({sum(clip_ok)}/{len(clip_ok)})",
             f"Frame accuracy with one hand removed: {acc_drop:.3f}",
             "", "[normal]"] + class_report(cm, names) + \
            ["", "[one hand removed]"] + class_report(cm_drop, names)
    report = "\n".join(lines)
    (report_dir / "val_report.txt").write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nModel  → {model_path}")
    print(f"Report → {report_dir}")


if __name__ == "__main__":
    main()
