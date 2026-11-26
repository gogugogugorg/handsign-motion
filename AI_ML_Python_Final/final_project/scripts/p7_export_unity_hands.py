"""
p7_export_unity_hands.py
후처리 결과(data/handsign_processed)의 손 42점을 Unity 재생용 CSV 로 내보낸다.
원본과 후처리를 각각 파일로 저장 → Unity 에서 나란히 비교.

실행 (final_project 폴더에서):
    python scripts/p7_export_unity_hands.py                                  # 인마다 1클립 + 예시 클립
    python scripts/p7_export_unity_hands.py --clips boar_20261127_015305 snake_20261127_000347
    python scripts/p7_export_unity_hands.py --out ../../HandSignViewer/Assets/StreamingAssets/HandSign

CSV 형식 (한 줄 = 한 관절):
    frame, time, landmark, x, y, z, state
      landmark 0~20 왼손, 21~41 오른손 (MediaPipe 손 관절 번호 + 21)
      x, y, z  : Unity 좌표 (미터). 클립 평균 손목 위치 = 0, 클립 손 크기 중앙값 = 1 로 정규화한 뒤
                 --hand-size 를 곱함 (클립 안에서 기준이 고정 → 실제 움직임/크기 보존)
                 y 는 위쪽이 +, z 는 카메라 쪽이 - (MediaPipe 와 Unity 카메라 방향이 같음)
      state    : 0 = 없음, 1 = 검출됨, 2 = 보간/팔 기반 추정으로 채움 (후처리 파일에만)
"""

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "handsign_processed"
DEFAULT_OUT = ROOT.parent.parent / "HandSignViewer" / "Assets" / "StreamingAssets" / "HandSign"
SIGN_ORDER = ["rat", "ox", "tiger", "hare", "dragon", "snake", "horse", "ram", "monkey", "bird", "dog", "boar"]
EXAMPLE = "boar_20261127_015305"
FIT_X, FIT_Y = 2.6, 1.2   # 손 크기 단위. Unity 장면에서 원본/후처리 한 칸의 반폭 / 라벨 사이 반높이


def clip_normalize(arr, ref):
    """
    시각화용 정규화: 클립 전체에 '고정된' 원점/스케일 사용.
    (모델 입력용 norm 은 프레임마다 화면상 손 크기로 나누는데, 손이 카메라 쪽을 향하면
     손 크기가 작게 잡혀 좌표가 몇 배로 부풀어 보임 → 시각화에는 부적합)
    arr, ref: (T,42,3) 비율 보정된 좌표, 결측 = NaN.  ref(후처리 결과)로 원점/스케일 계산.
    """
    r = ref.reshape(-1, 2, 21, 3)
    with np.errstate(all="ignore"):
        # 원점: x, y 는 클립 전체 관절이 차지하는 영역의 한가운데, z 는 평균 손목 깊이
        lo, hi = np.nanmin(ref, axis=(0, 1)), np.nanmax(ref, axis=(0, 1))
        center = (lo + hi) / 2
        center[2] = np.nanmean(r[:, :, 0, 2])
        # 손 크기: z 까지 포함한 3D 길이 (손끝이 카메라를 향하면 화면상 길이는 짧아지므로)
        size = np.nanmedian(np.linalg.norm(r[:, :, 9, :] - r[:, :, 0, :], axis=-1))
        out = (arr - center) / size
        # Unity 한 칸(가로 ±FIT_X, 세로 ±FIT_Y 손 크기)을 벗어나면 전체를 축소
        half = (hi - lo)[:2] / 2 / size
    k = min(1.0, FIT_X / half[0], FIT_Y / half[1]) if np.all(np.isfinite(half)) else 1.0
    return np.nan_to_num(out * k, nan=0.0)


def write_csv(path, pts, state, ts, hand_size):
    """pts (T,42,3) 정규화 좌표, state (T,2)"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time", "landmark", "x", "y", "z", "state"])
        for t in range(len(pts)):
            for j in range(42):
                s = int(state[t, j // 21])
                x, y, z = pts[t, j] * hand_size
                w.writerow([t, f"{ts[t]:.4f}", j, f"{x:.5f}", f"{-y:.5f}", f"{z:.5f}", s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=None, help="클립 이름들 (기본: 인마다 첫 클립 + 예시)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--hand-size", type=float, default=0.09, help="정규화 단위 1 = 몇 m (손목~중지 뿌리 약 9cm)")
    args = ap.parse_args()

    all_clips = {p.stem: p for p in IN_DIR.glob("*/*.npz") if not p.parent.name.startswith("_")}
    if args.clips:
        names = args.clips
    else:
        names = [EXAMPLE] if EXAMPLE in all_clips else []
        for s in SIGN_ORDER:
            cand = sorted(n for n, p in all_clips.items() if p.parent.name == s and n != EXAMPLE)
            # 원본 검출률이 높은 클립을 고름 (비교가 잘 보이도록 80% 이상 중 첫 번째)
            for n in cand:
                d = np.load(all_clips[n], allow_pickle=True)
                if d["mask_raw"].all(1).mean() >= 0.8:
                    names.append(n)
                    break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        d = np.load(all_clips[n], allow_pickle=True)
        ts = d["timestamps"] - d["timestamps"][0]
        m_raw, m_clean = d["mask_raw"].astype(bool), d["mask_clean"].astype(bool)
        state_raw = m_raw.astype(int)
        state_clean = np.where(m_raw, 1, np.where(m_clean, 2, 0))
        raw, clean = d["raw"].astype(np.float64), d["clean"].astype(np.float64)
        write_csv(out / f"{n}_raw.csv", clip_normalize(raw, clean), state_raw, ts, args.hand_size)
        write_csv(out / f"{n}_clean.csv", clip_normalize(clean, clean), state_clean, ts, args.hand_size)
        print(f"{n:28s} frames {len(ts):3d}  both raw {m_raw.all(1).mean():4.0%}  "
              f"filled frames {(state_clean == 2).sum():3d}")
    print(f"\nSaved {len(names)} clip pairs → {out}")


if __name__ == "__main__":
    main()
