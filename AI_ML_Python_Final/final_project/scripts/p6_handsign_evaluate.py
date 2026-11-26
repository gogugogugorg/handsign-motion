"""
p6_handsign_evaluate.py
학습에 쓰지 않은 시험용 데이터로 여러 모델을 같은 기준으로 채점한다.

시험 데이터 만들기 (학습 폴더와 분리!):
    python scripts/p0_record_hand_signs.py --out data/handsign_test
채점:
    python scripts/p6_handsign_evaluate.py                       # checkpoints/handsign_mlp*.pth 전부
    python scripts/p6_handsign_evaluate.py --models handsign_mlp_v1.pth handsign_mlp.pth

채점 방식 (실시간 p5 와 같은 처리):
    - p1_3 후처리 (튐 제거, 보간, 팔 기반 추정, One-Euro, 정규화)
    - 두 손이 모두 유효한 프레임만 모델에 넣고, 아니면 none 으로 처리
    지표
      frame acc (both)  : 두 손이 유효한 프레임에서의 분류 정확도  → 분류기 자체 성능
      frame acc (all)   : 모든 프레임 기준 (손 검출 실패 포함)        → 파이프라인 전체 성능
      clip confirm acc  : p5 의 SealDetector 규칙(conf, hold, window)으로 클립을 재생했을 때
                          인 클립은 '정답 인이 확정되고 다른 인은 확정 안 됨', none 클립은 '아무 인도 확정 안 됨'
                          → 실제 사용감에 가장 가까운 지표
      none FP / min     : none 클립에서 1분당 잘못 확정된 인 수

출력: data/handsign_test/_report/
    summary.csv, per_class_<model>.txt, confusion_<model>.png
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import p1_3_handsign_post_processing as pp  # noqa: E402
from p3_handsign_train import HandSignMLP, plot_confusion  # noqa: E402
from p5_handsign_realtime import SealDetector  # noqa: E402
from utils.handsign_features import frame_features  # noqa: E402

NONE_LABEL = "none"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", default=str(ROOT / "data" / "handsign_test"))
    p.add_argument("--models", nargs="+", default=None, help="checkpoints/ 안의 파일 이름들")
    p.add_argument("--conf", type=float, default=0.8)
    p.add_argument("--hold", type=float, default=0.3)
    p.add_argument("--window", type=int, default=3)
    return p.parse_args()


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = HandSignMLP(ck["feature_dim"], len(ck["labels"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck["labels"], ck["use_arm"]


def predict_clip(model, labels, use_arm, res):
    """모든 프레임의 확률 (T, C). 두 손이 유효하지 않은 프레임은 none one-hot."""
    T = len(res["norm"])
    probs = np.tile(np.eye(len(labels))[labels.index(NONE_LABEL)], (T, 1))
    ok = res["mask_clean"].all(1)
    if ok.any():
        x = frame_features(res["norm"][ok].reshape(-1, 42, 3), res["mask_clean"][ok],
                           res["arm"][ok] if use_arm else None)
        with torch.no_grad():
            probs[ok] = torch.softmax(model(torch.from_numpy(x)), 1).numpy()
    return probs, ok


def replay(labels, probs, ts, args):
    """SealDetector 로 클립을 재생해서 확정된 인 목록 반환."""
    det = SealDetector(labels, args.window, args.conf, args.hold)
    out = []
    for p, t in zip(probs, ts):
        s = det.update(p, float(t))
        if s is not None:
            out.append(s)
    return out


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)
    report_dir = test_dir / "_report"
    paths = sorted(p for p in test_dir.glob("*/*.npz") if not p.parent.name.startswith("_"))
    if not paths:
        raise RuntimeError(f"No test clips in {test_dir}. Record with: "
                           f"python scripts/p0_record_hand_signs.py --out data/handsign_test")
    model_names = args.models or sorted(p.name for p in (ROOT / "checkpoints").glob("handsign_mlp*.pth"))

    # ---- 후처리는 모델과 무관하므로 한 번만 ----
    pp_args = pp.parse_args([])
    clips = []
    for p in paths:
        c = pp.load_clip(p)
        clips.append((p.stem, c["label"], pp.process_clip(c, pp_args), c["ts"]))
    from collections import Counter
    cnt = Counter(l for _, l, _, _ in clips)
    print(f"Test clips: {len(clips)}  " + "  ".join(f"{k}:{v}" for k, v in sorted(cnt.items())))
    print(f"Detector: conf {args.conf}  hold {args.hold}s  window {args.window}\n")

    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'model':26s} {'acc(both)':>10s} {'acc(all)':>9s} {'clip ok':>9s} {'none FP/min':>12s}")
    for name in model_names:
        model, labels, use_arm = load_model(ROOT / "checkpoints" / name)
        C = len(labels)
        cm = np.zeros((C, C), int)
        both_ok = both_n = all_ok = all_n = 0
        clip_ok, none_fp, none_sec = [], 0, 0.0
        per_class = {l: [0, 0, 0, 0] for l in labels}   # clips, clip_ok, frames_both, frames_both_ok
        for stem, label, res, ts in clips:
            y = labels.index(label)
            probs, ok = predict_clip(model, labels, use_arm, res)
            pred = probs.argmax(1)
            np.add.at(cm, (np.full(ok.sum(), y), pred[ok]), 1)
            both_ok += (pred[ok] == y).sum()
            both_n += ok.sum()
            all_ok += (pred == y).sum()
            all_n += len(pred)
            confirmed = replay(labels, probs, ts, args)
            if label == NONE_LABEL:
                good = len(confirmed) == 0
                none_fp += len(confirmed)
                none_sec += float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
            else:
                good = label in confirmed and all(s == label for s in confirmed)
            clip_ok.append(good)
            pc = per_class[label]
            pc[0] += 1
            pc[1] += good
            pc[2] += ok.sum()
            pc[3] += (pred[ok] == y).sum()

        acc_both = both_ok / max(both_n, 1)
        acc_all = all_ok / max(all_n, 1)
        clip_acc = np.mean(clip_ok)
        fp_min = none_fp / (none_sec / 60) if none_sec > 0 else float("nan")
        print(f"{name:26s} {acc_both:10.1%} {acc_all:9.1%} {clip_acc:9.1%} {fp_min:12.2f}")
        rows.append({"model": name, "acc_both": acc_both, "acc_all": acc_all,
                     "clip_confirm_acc": clip_acc, "none_fp_per_min": fp_min})

        lines = [f"{name}  (conf {args.conf}, hold {args.hold}s)",
                 f"{'class':8s} {'clips':>5s} {'clip ok':>8s} {'frames(both)':>13s} {'acc(both)':>10s}  most confused with"]
        for i, l in enumerate(labels):
            c, cok, fb, fbok = per_class[l]
            if c == 0:
                continue
            row = cm[i].copy()
            row[i] = 0
            conf = f"{labels[row.argmax()]} ({row.max() / max(cm[i].sum(), 1):.0%})" if row.max() else "-"
            lines.append(f"{l:8s} {c:5d} {cok:4d}/{c:<3d} {fb:13d} {fbok / max(fb, 1):10.1%}  {conf}")
        (report_dir / f"per_class_{Path(name).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        present = [i for i, l in enumerate(labels) if per_class[l][0] > 0]
        plot_confusion(cm[np.ix_(present, present)], [labels[i] for i in present],
                       f"Test set confusion (both hands valid) [{Path(name).stem}]",
                       report_dir / f"confusion_{Path(name).stem}.png")

    with open(report_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-class reports → {report_dir}")


if __name__ == "__main__":
    main()
