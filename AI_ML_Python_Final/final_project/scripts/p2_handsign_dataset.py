"""
p2_handsign_dataset.py
p1_3 후처리 결과(data/handsign_processed)를 프레임 단위 학습 샘플로 만든다.

실행 (final_project 폴더에서):
    python scripts/p2_handsign_dataset.py
    python scripts/p2_handsign_dataset.py --min-detection 0.6 --val-ratio 0.25

규칙:
    - 클립 단위로 train / val 분리 (같은 클립의 프레임이 양쪽에 섞이면 정확도가 부풀려짐)
      배정은 클립 이름 해시로 고정 → 데이터를 추가해도 기존 클립의 배정이 유지됨
    - 인 클래스: 원본 두 손 검출률 < min_detection 인 클립은 제외, 두 손 모두 유효한 프레임만 사용
    - none 클래스: 손이 하나라도 유효한 프레임 사용 (손이 0개면 실시간에서 모델 없이 none 처리)

출력: data/handsign_dataset/
    train.npz, val.npz   norm (N,42,3), mask (N,2), arm (N,10,2), y (N,), clip (N,)
    labels.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "handsign_processed"
OUT_DIR = ROOT / "data" / "handsign_dataset"
NONE_LABEL = "none"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--min-detection", type=float, default=0.5, help="인 클립의 최소 원본 두 손 검출률")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def stable_hash(name):
    """클립 이름 → [0, 1) 고정 값 (실행할 때마다, 컴퓨터가 달라도 같음)."""
    return int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000


def load_clip(path, is_none, min_detection):
    d = np.load(path, allow_pickle=True)
    norm, mask, arm = d["norm"], d["mask_clean"].astype(bool), d["arm"]
    det_raw = d["mask_raw"].astype(bool).all(1).mean()

    if not is_none and det_raw < min_detection:
        return None, det_raw
    keep = mask.any(1) if is_none else mask.all(1)
    return (norm[keep], mask[keep], arm[keep]), det_raw


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    class_dirs = sorted(p for p in IN_DIR.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if not class_dirs:
        raise RuntimeError(f"No processed clips in {IN_DIR}. Run p1_3 first.")
    labels = {d.name: i for i, d in enumerate(class_dirs)}

    split = {s: {"norm": [], "mask": [], "arm": [], "y": [], "clip": []} for s in ("train", "val")}

    print(f"{'class':8s} {'clips':>5s} {'dropped':>8s} {'train clips':>12s} {'val clips':>10s} "
          f"{'train frames':>13s} {'val frames':>11s}")
    for d in class_dirs:
        name, y = d.name, labels[d.name]
        is_none = name == NONE_LABEL
        clips, dropped = [], []
        for path in sorted(d.glob("*.npz")):
            data, det = load_clip(path, is_none, args.min_detection)
            if data is None or len(data[0]) == 0:
                dropped.append(f"{path.stem} ({det:.0%})")
            else:
                clips.append((path.stem, data))

        if len(clips) < 2:
            raise RuntimeError(f"'{name}' has fewer than 2 usable clips - record more.")

        # 클립 이름 해시로 val 배정 → 클립을 추가해도 기존 클립의 train/val 이 바뀌지 않는다.
        # (예전: 인마다 무작위 순열 → 한 인에 클립이 늘면 다른 인들의 val 클립까지 바뀌어
        #  모델 버전 간 검증 정확도를 비교할 수 없었음)
        h = np.array([stable_hash(stem) for stem, _ in clips])
        val_idx = set(np.where(h < args.val_ratio)[0].tolist())
        if not val_idx:                                  # 최소 1개는 val
            val_idx = {int(h.argmin())}
        if len(val_idx) == len(clips):                   # 최소 1개는 train
            val_idx.discard(int(h.argmax()))
        n_val = len(val_idx)

        counts = {"train": 0, "val": 0}
        for i, (stem, (norm, mask, arm)) in enumerate(clips):
            s = "val" if i in val_idx else "train"
            split[s]["norm"].append(norm)
            split[s]["mask"].append(mask)
            split[s]["arm"].append(arm)
            split[s]["y"].append(np.full(len(norm), y))
            split[s]["clip"].append(np.full(len(norm), stem))
            counts[s] += len(norm)

        print(f"{name:8s} {len(clips) + len(dropped):5d} {len(dropped):8d} "
              f"{len(clips) - n_val:12d} {n_val:10d} {counts['train']:13d} {counts['val']:11d}")
        for dr in dropped:
            print(f"           dropped: {dr}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s, v in split.items():
        np.savez(
            OUT_DIR / f"{s}.npz",
            norm=np.concatenate(v["norm"]).astype(np.float32),
            mask=np.concatenate(v["mask"]),
            arm=np.concatenate(v["arm"]).astype(np.float32),
            y=np.concatenate(v["y"]).astype(np.int64),
            clip=np.concatenate(v["clip"]),
        )
    with open(OUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    n_tr = sum(len(a) for a in split["train"]["y"])
    n_va = sum(len(a) for a in split["val"]["y"])
    print(f"\nTrain frames: {n_tr}   Val frames: {n_va}   Classes: {len(labels)}")
    print(f"Saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
