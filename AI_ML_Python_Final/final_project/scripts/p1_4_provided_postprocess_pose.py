"""
p1_4_provided_postprocess_pose.py
과제에서 제공된 후처리(p1_2_post_processing.create_height_corrected_target: 발 기준 높이 보정)를
인 녹화 데이터의 전신 3D 좌표(pose_world)에 그대로 적용하고, 전/후를 비교한다.

실행 (final_project 폴더에서):
    python scripts/p1_4_provided_postprocess_pose.py
    python scripts/p1_4_provided_postprocess_pose.py --show snake_20261127_000347   # viser 로 raw/보정 나란히 재생

좌표 변환은 제공 코드 p1_create_raw_keypoints.py 와 동일: (x, -y, z)  → y 위쪽이 +
보정은 제공 함수를 import 해서 그대로 사용 (수정 없음).

분석:
    - 발(31, 32) visibility: 웹캠 상반신 촬영에서 발이 실제로 보이는가
    - 보정 전/후 손목 y 의 떨림 (2차 차분): 발 추정값의 흔들림이 몸 전체로 전달되는가
    - 보정 후 골반 높이의 흔들림
출력: data/handsign_processed/_report/provided/
    provided_summary.txt, foot_visibility.png, wrist_jitter.png, example_<clip>.png, <clip>_target.csv (Unity)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from p1_2_post_processing import create_height_corrected_target, export_unity_csv  # noqa: E402  (제공 코드)

IN_DIR = ROOT / "data" / "handsign_keypoints"
OUT_DIR = ROOT / "data" / "handsign_processed" / "_report" / "provided"

L_FOOT, R_FOOT = 31, 32
L_HIP, R_HIP = 23, 24
L_WRIST, R_WRIST = 15, 16
POSE_CONNECTIONS = [
    (11, 12), (12, 14), (14, 16), (11, 13), (13, 15), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32),
]
C_RAW, C_POST, C_TEXT, C_TEXT2, C_GRID = "#f47230", "#257de0", "#191919", "#666666", "#e6e6e6"


def load_pose_world(path):
    """npz → (T,33,3) 제공 코드와 같은 좌표계 (y 반전), (T,33) visibility. pose 미검출 프레임 제외."""
    d = np.load(path, allow_pickle=True)
    pw = d["pose_world"].astype(np.float64)          # (T,33,4) x,y,z,vis
    detected = np.abs(pw[:, :, :3]).sum((1, 2)) > 0
    pw = pw[detected]
    seq = pw[:, :, :3].copy()
    seq[:, :, 1] *= -1                                # 제공 코드: "y": -lm.y
    return seq, pw[:, :, 3], d["timestamps"][detected], str(d["label"])


def jitter_y(y):
    """1차원 시계열의 평균 |2차 차분| (m / frame²)"""
    return float(np.abs(y[2:] - 2 * y[1:-1] + y[:-2]).mean()) if len(y) > 2 else np.nan


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT2, labelsize=8)
    ax.yaxis.grid(True, color=C_GRID)
    ax.set_axisbelow(True)


def draw_skeleton(ax, pts, color, label):
    for a, b in POSE_CONNECTIONS:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color=color, lw=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color=color, label=label, zorder=3)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--example", default=None, help="예시로 그릴 클립 이름 (기본: 인 클립 중 첫 번째)")
    p.add_argument("--show", default=None, help="이 클립을 viser 로 raw/보정 나란히 재생")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in IN_DIR.glob("*/*.npz"))

    rows = []
    for p in paths:
        seq, vis, ts, label = load_pose_world(p)
        if len(seq) < 5:
            continue
        tgt = create_height_corrected_target(seq)      # ← 제공된 후처리
        foot_vis = vis[:, [L_FOOT, R_FOOT]].mean()
        ground = np.minimum(seq[:, L_FOOT, 1], seq[:, R_FOOT, 1])
        hip_after = tgt[:, [L_HIP, R_HIP], 1].mean(1)
        rows.append({
            "clip": p.stem, "label": label, "frames": len(seq),
            "foot_vis": foot_vis,
            "ground_std": ground.std(),
            "hip_after_std": hip_after.std(),
            "hip_after_mean": hip_after.mean(),
            "wrist_jit_raw": np.mean([jitter_y(seq[:, w, 1]) for w in (L_WRIST, R_WRIST)]),
            "wrist_jit_post": np.mean([jitter_y(tgt[:, w, 1]) for w in (L_WRIST, R_WRIST)]),
            "clamped": float((tgt[:, :, 1] == 0).mean()),
        })

    seal = [r for r in rows if r["label"] != "none"]
    fv = np.array([r["foot_vis"] for r in rows])
    jr = np.array([r["wrist_jit_raw"] for r in seal])
    jp = np.array([r["wrist_jit_post"] for r in seal])
    lines = [
        "Provided post-processing (foot-based height correction) applied to pose_world",
        f"clips: {len(rows)} (seal clips: {len(seal)})",
        "",
        f"foot visibility (mean of landmarks 31, 32): mean {fv.mean():.3f}, "
        f"clips with mean > 0.5: {(fv > 0.5).sum()} / {len(fv)}",
        f"estimated ground (min foot y) std within clip: median {np.median([r['ground_std'] for r in rows]) * 100:.1f} cm",
        f"hip height after correction: mean {np.mean([r['hip_after_mean'] for r in rows]):.2f} m, "
        f"within-clip std median {np.median([r['hip_after_std'] for r in rows]) * 100:.1f} cm",
        f"wrist vertical jitter (seal clips, |2nd diff|): raw {jr.mean() * 1000:.2f} mm/frame²  "
        f"→ corrected {jp.mean() * 1000:.2f} mm/frame²  ({(jp.mean() / jr.mean() - 1) * 100:+.0f}%)",
        f"clips where correction increased wrist jitter: {(jp > jr).sum()} / {len(jr)}",
        f"joints clamped to y=0: {np.mean([r['clamped'] for r in rows]) * 100:.2f}% of joint-frames",
    ]
    report = "\n".join(lines)
    (OUT_DIR / "provided_summary.txt").write_text(report, encoding="utf-8")
    print(report)

    # ---- 그림 1: 발 visibility 분포 ----
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
    ax.hist(fv, bins=20, range=(0, 1), color=C_POST, rwidth=0.9)
    ax.axvline(0.5, color=C_TEXT2, lw=1, ls="--")
    ax.set_xlabel("mean foot visibility per clip (landmarks 31, 32)", color=C_TEXT2, fontsize=9)
    ax.set_ylabel("clips", color=C_TEXT2, fontsize=9)
    ax.set_title("Feet are (almost) never visible in a webcam upper-body view", color=C_TEXT,
                 fontsize=11, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "foot_visibility.png")
    plt.close(fig)

    # ---- 그림 2: 손목 떨림 전/후 (인 클립) ----
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    lim = max(jr.max(), jp.max()) * 1000 * 1.05
    ax.scatter(jr * 1000, jp * 1000, s=14, color=C_POST, alpha=0.7)
    ax.plot([0, lim], [0, lim], color=C_TEXT2, lw=1, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("wrist y jitter, raw (mm/frame²)", color=C_TEXT2, fontsize=9)
    ax.set_ylabel("after provided correction", color=C_TEXT2, fontsize=9)
    ax.set_title("Above the line = correction made it shakier", color=C_TEXT, fontsize=11, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wrist_jitter.png")
    plt.close(fig)

    # ---- 그림 3: 예시 클립 (시간축 + 스켈레톤) ----
    ex = args.example or seal[0]["clip"]
    ex_path = next(p for p in paths if p.stem == ex)
    seq, vis, ts, label = load_pose_world(ex_path)
    tgt = create_height_corrected_target(seq)
    t = ts - ts[0]
    fig = plt.figure(figsize=(9, 3.3), dpi=200)
    ax1 = fig.add_subplot(1, 3, (1, 2))
    ax1.plot(t, np.minimum(seq[:, L_FOOT, 1], seq[:, R_FOOT, 1]), color=C_TEXT2, lw=1.5,
             label="estimated ground (min foot y, raw)")
    ax1.plot(t, seq[:, R_WRIST, 1], color=C_RAW, lw=2, label="right wrist y, raw (hip-centered)")
    ax1.plot(t, tgt[:, R_WRIST, 1], color=C_POST, lw=2, label="right wrist y, after correction")
    ax1.set_xlabel("time (s)", color=C_TEXT2, fontsize=9)
    ax1.set_ylabel("y (m, up +)", color=C_TEXT2, fontsize=9)
    ax1.set_title(f"{ex}: holding a sign - wrist should be still", color=C_TEXT, fontsize=11, loc="left")
    ax1.legend(frameon=False, fontsize=8)
    style(ax1)
    ax2 = fig.add_subplot(1, 3, 3)
    k = len(seq) // 2
    draw_skeleton(ax2, seq[k], C_RAW, "raw (hip at 0)")
    draw_skeleton(ax2, tgt[k], C_POST, "after correction")
    ax2.axhline(0, color=C_TEXT2, lw=1)
    ax2.set_aspect("equal")
    ax2.set_title("front view, middle frame", color=C_TEXT, fontsize=11, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    style(ax2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"example_{ex}.png")
    plt.close(fig)
    export_unity_csv(tgt, str(OUT_DIR / f"{ex}_target.csv"))   # 제공 함수로 Unity CSV
    print(f"\nFigures → {OUT_DIR}")

    if args.show:
        from utils.viser_test import PoseViser
        p = next(p for p in paths if p.stem == args.show)
        seq, _, _, _ = load_pose_world(p)
        PoseViser(fps=15).play_two_sequences(seq, create_height_corrected_target(seq))


if __name__ == "__main__":
    main()
