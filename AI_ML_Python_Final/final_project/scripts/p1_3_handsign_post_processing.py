"""
p1_3_handsign_post_processing.py
p0_record_hand_signs.py 로 녹화한 손 키포인트를 후처리하고, 전/후 비교 그래프를 만든다.

실행 (final_project 폴더에서):
    python scripts/p1_3_handsign_post_processing.py
    python scripts/p1_3_handsign_post_processing.py --max-gap 0.5 --min-cutoff 1.0

파이프라인 (손 하나씩 따로 처리):
    ① 비율 보정   x, z 에 W/H 를 곱해 가로세로 단위를 맞춤
    ② 튐 제거     앞뒤 프레임은 서로 비슷한데 가운데만 크게 벗어난 프레임 → 결측 처리
    ③ 보간        max_gap 초 이하의 결측 구간은 선형 보간 (양 끝은 가까운 값 유지)
    ③-b 팔 기반 추정  그래도 남은 결측: 가장 가까운 유효 손 모양을 가져와서,
                     전신(pose) 손목이 손 크기 * anchor_move 이상 움직였으면 그만큼 옮기고
                     아니면 그대로 유지 (anchor_gap 초 이내의 기준 프레임만)
    ④ 스무딩      One-Euro 필터 (causal → 실시간 인식에서도 같은 필터 사용 가능)
    ⑤ 정규화      두 손목 중점 = 원점, 손 크기(손목→중지 MCP) = 1. 팔 점들도 같은 기준으로 정규화

출력:
    data/handsign_processed/<인>/<클립>.npz
        raw        (T,42,3)  ① 까지만 적용, 결측 = NaN
        clean      (T,42,3)  ②~④ 적용, 여전히 결측이면 NaN
        norm       (T,42,3)  clean 을 ⑤ 정규화, 결측 = 0   ← 모델 입력용
        arm        (T,10,2)  팔 점(ARM_POINTS) xy 를 같은 기준으로 정규화, 결측 = 0  ← 모델 입력용
        raw_norm   (T,42,3)  raw 를 ⑤ 정규화 (비교용)
        mask_raw   (T,2)     원본 검출 여부 [왼손, 오른손]
        mask_interp(T,2)     ③ 보간까지 적용 후 유효 여부
        mask_clean (T,2)     ③-b 까지 적용 후 유효 여부
        estimated  (T,2)     ③-b 팔 기반으로 추정된 프레임
        spikes     (T,2)     튐으로 제거된 프레임
        timestamps, fps, img_size, label
    data/handsign_processed/_report/
        detection_rate.png, jitter.png, anchor_eval.png, example_<클립>.png, summary.csv
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils.handsign_features import ARM_POINTS, POSE_ELBOW, POSE_WRIST  # noqa: E402

IN_DIR = ROOT / "data" / "handsign_keypoints"
OUT_DIR = ROOT / "data" / "handsign_processed"
REPORT_DIR = OUT_DIR / "_report"

N_HAND = 21
WRIST, MIDDLE_MCP, INDEX_TIP = 0, 9, 8
NONE_LABEL = "none"

VIS_TH = 0.5           # pose 점 visibility 기준

# chart colors (light surface)
C_RAW = "#f47230"      # orange
C_CLEAN = "#257de0"    # blue
C_POSE = "#00b894"     # aqua
C_TEXT = "#191919"
C_TEXT2 = "#666666"
C_GRID = "#e6e6e6"
C_GAP = "#efefef"


# ============================================================
# ② Spike removal
# ============================================================
def hand_size(h):
    """h: (T,21,3) → (T,) 손목~중지 MCP 거리 (xy)"""
    return np.linalg.norm(h[:, MIDDLE_MCP, :2] - h[:, WRIST, :2], axis=1)


def remove_spikes(h, k):
    """
    가운데 프레임이 앞뒤 프레임의 평균에서 손 크기 * k 이상 벗어나고,
    앞뒤 프레임끼리는 그보다 가까우면 → 순간 오검출로 보고 NaN 처리.
    """
    out = h.copy()
    spikes = np.zeros(len(h), bool)
    size = hand_size(h)
    for t in range(1, len(h) - 1):
        a, b, c = h[t - 1], h[t], h[t + 1]
        if np.isnan(a[0, 0]) or np.isnan(b[0, 0]) or np.isnan(c[0, 0]):
            continue
        mid = (a + c) / 2
        err = np.linalg.norm(b[:, :2] - mid[:, :2], axis=1).mean()
        neighbor = np.linalg.norm(a[:, :2] - c[:, :2], axis=1).mean()
        ref = np.nanmean([size[t - 1], size[t + 1]])
        if err > k * ref and neighbor < err:
            spikes[t] = True
    out[spikes] = np.nan
    return out, spikes


# ============================================================
# ③ Gap filling
# ============================================================
def fill_gaps(h, ts, max_gap):
    """
    h: (T,21,3) 결측 = NaN, ts: (T,) 초
    두 유효 프레임 사이 시간이 max_gap 이하이면 선형 보간.
    시작/끝 결측은 max_gap 이내면 가장 가까운 유효 프레임으로 채움.
    """
    out = h.copy()
    valid = ~np.isnan(h[:, 0, 0])
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return out

    for i0, i1 in zip(idx[:-1], idx[1:]):
        if i1 - i0 > 1 and ts[i1] - ts[i0] <= max_gap:
            w = (ts[i0 + 1:i1] - ts[i0]) / (ts[i1] - ts[i0])
            out[i0 + 1:i1] = (1 - w)[:, None, None] * h[i0] + w[:, None, None] * h[i1]

    first, last = idx[0], idx[-1]
    lead = np.arange(first)[ts[first] - ts[:first] <= max_gap]
    out[lead] = h[first]
    tail = np.arange(last + 1, len(h))
    tail = tail[ts[tail] - ts[last] <= max_gap]
    out[tail] = h[last]
    return out


# ============================================================
# ③-b Pose(팔) 기반 손 추정
# ============================================================
def anchor_transform(ref, pw_r, pe_r, pw, pe, rotate=True):
    """
    ref: (21,3) 기준 프레임의 손, pw_r/pe_r: 기준 프레임의 pose 손목/팔꿈치 xy,
    pw/pe: 현재 프레임의 pose 손목/팔꿈치 xy.
    손 모양은 그대로 두고, (손목 이동 + 전완 회전) 만큼 강체 변환.
    기준 프레임에서 잰 '손 모델 손목 - pose 손목' 오프셋도 같이 회전시켜 pose 손목 오차를 상쇄.
    """
    dth = 0.0
    if rotate:
        dth = np.arctan2(*(pw - pe)[::-1]) - np.arctan2(*(pw_r - pe_r)[::-1])
    c, s = np.cos(dth), np.sin(dth)
    Rm = np.array([[c, -s], [s, c]])
    out = np.empty_like(ref)
    shape = ref[:, :2] - ref[0, :2]
    offset = ref[0, :2] - pw_r
    out[:, :2] = pw + (offset + shape) @ Rm.T
    out[:, 2] = ref[:, 2]
    return out


def estimate_from_arm(ref, pw_r, pe_r, pw, pe, ref_size, min_move, rotate=False):
    """
    조건부 팔 기반 추정.
    pose 손목이 기준 프레임 대비 손 크기 * min_move 이상 움직였을 때만 이동을 반영하고,
    그보다 작으면 pose 손목의 떨림으로 보고 기준 손을 그대로 유지한다.
    (실험 결과: 정지 상태에선 '유지'가, 움직일 땐 '팔 기반'이 더 정확했고,
     전완 회전까지 반영하면 팔꿈치 노이즈 때문에 오히려 오차가 커졌음 → rotate 기본 False)
    """
    if np.linalg.norm(pw - pw_r) <= min_move * ref_size:
        return ref.copy()
    return anchor_transform(ref, pw_r, pe_r, pw, pe, rotate)


def pose_anchor_fill(h, pose, side, ts, max_gap, min_move):
    """h: (T,21,3) 결측 = NaN. 남은 결측 프레임을 팔 기반으로 추정. 반환 (h, estimated)"""
    out = h.copy()
    est = np.zeros(len(h), bool)
    valid = ~np.isnan(h[:, 0, 0])
    idx = np.where(valid)[0]
    if len(idx) == 0 or max_gap <= 0:
        return out, est
    pw_all, pe_all = pose[:, POSE_WRIST[side]], pose[:, POSE_ELBOW[side]]
    size = hand_size(h)

    for t in np.where(~valid)[0]:
        if pw_all[t, 3] < VIS_TH:
            continue
        r = idx[np.argmin(np.abs(ts[idx] - ts[t]))]          # 가장 가까운 유효 프레임
        if abs(ts[r] - ts[t]) > max_gap or pw_all[r, 3] < VIS_TH:
            continue
        out[t] = estimate_from_arm(h[r], pw_all[r, :2], pe_all[r, :2],
                                   pw_all[t, :2], pe_all[t, :2], size[r], min_move)
        est[t] = True
    return out, est


EVAL_METHODS = ("hold", "arm", "adaptive")


GAP_BINS = np.arange(0.0, 1.61, 0.2)   # 빈 구간 길이(초) 구간: 0~0.2, 0.2~0.4, ... 1.4~1.6


def eval_anchor(raw, pose, ts, min_move, max_k=25):
    """
    추정 정확도 검증: 실제로 검출된 프레임 t 의 손을 '없다고 치고'
    k 프레임 전의 손으로 추정 → 진짜 값과 비교 (손 크기 단위 평균 오차).
      hold     : k 프레임 전 손을 그대로 유지
      arm      : 항상 pose 손목 이동만큼 옮김
      adaptive : estimate_from_arm (실제 사용하는 방법)
    클립마다 fps 가 달라서(8 / 14 / 4 fps) 프레임 수가 아니라 실제 시간 간격으로 묶는다.
    반환: {gap_bin_index: {method: [오차, ...]}}
    """
    res = {}
    for s in range(2):
        h = raw[:, s]
        valid = ~np.isnan(h[:, 0, 0])
        pw_all, pe_all = pose[:, POSE_WRIST[s]], pose[:, POSE_ELBOW[s]]
        size = hand_size(h)
        for k in range(1, max_k + 1):
            for t in range(k, len(h)):
                r = t - k
                if not (valid[t] and valid[r]) or pw_all[t, 3] < VIS_TH or pw_all[r, 3] < VIS_TH:
                    continue
                args = (h[r], pw_all[r, :2], pe_all[r, :2], pw_all[t, :2], pe_all[t, :2])
                b = int(np.searchsorted(GAP_BINS, ts[t] - ts[r], side="right")) - 1
                if b < 0 or b >= len(GAP_BINS) - 1:
                    continue
                est = {
                    "hold": h[r],
                    "arm": anchor_transform(*args, rotate=False),
                    "adaptive": estimate_from_arm(*args, size[r], min_move),
                }
                slot = res.setdefault(b, {m: [] for m in EVAL_METHODS})
                for m, e in est.items():
                    err = np.linalg.norm(e[:, :2] - h[t, :, :2], axis=1).mean() / size[t]
                    slot[m].append(err)
    return res


# ============================================================
# ④ One-Euro filter (Casiez et al., 2012)
# ============================================================
class OneEuroFilter:
    """
    배열 단위 One-Euro 필터. 과거 값만 사용하므로 실시간에서도 그대로 쓸 수 있다.
    min_cutoff ↓ → 멈춰 있을 때 더 강하게 스무딩 (대신 지연 ↑)
    beta ↑       → 빠르게 움직일 때 지연 ↓
    """

    def __init__(self, min_cutoff=1.5, beta=3.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = self.dx_prev = self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def smooth(h, ts, min_cutoff, beta):
    """결측 프레임을 만나면 필터를 리셋 (끊긴 구간끼리 섞이지 않게)."""
    f = OneEuroFilter(min_cutoff, beta)
    out = h.copy()
    for t in range(len(h)):
        if np.isnan(h[t, 0, 0]):
            f.reset()
            continue
        out[t] = f(h[t], ts[t])
    return out


# ============================================================
# ⑤ Normalization
# ============================================================
def normalize(h2):
    """
    h2: (T,2,21,3) 결측 = NaN → (T,2,21,3) 결측 = 0
    원점 = 유효한 손목들의 평균, 스케일 = 유효한 손 크기의 평균
    반환: (norm, center (T,3), size (T,))
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nanmean(h2[:, :, WRIST, :], axis=1)                          # (T,3)
        size = np.nanmean(np.stack([hand_size(h2[:, 0]), hand_size(h2[:, 1])], 1), axis=1)  # (T,)
    norm = (h2 - center[:, None, None, :]) / size[:, None, None, None]
    return np.nan_to_num(norm, nan=0.0), center, size


def normalize_arm(pose, center, size):
    """pose (T,33,4) → (T,10,2): ARM_POINTS xy 를 손과 같은 원점/스케일로. 안 보이는 점 = 0"""
    p = pose[:, ARM_POINTS, :]
    arm = (p[:, :, :2] - center[:, None, :2]) / size[:, None, None]
    arm[p[:, :, 3] < VIS_TH] = np.nan
    return np.nan_to_num(arm, nan=0.0)


# ============================================================
# Clip processing
# ============================================================
def load_clip(path):
    d = np.load(path, allow_pickle=True)
    W, H = d["img_size"]
    hands = d["hands"].astype(np.float64).reshape(-1, 2, N_HAND, 3)
    mask = d["hand_mask"].astype(bool)
    hands[..., 0] *= W / H          # ① 비율 보정
    hands[..., 2] *= W / H
    hands[~mask] = np.nan
    pose = d["pose"].astype(np.float64)
    pose[..., 0] *= W / H
    pose[..., 2] *= W / H
    return {
        "hands": hands, "mask": mask, "pose": pose,
        "ts": d["timestamps"].astype(np.float64),
        "fps": float(d["fps"]), "img_size": d["img_size"], "label": str(d["label"]),
    }


def process_clip(clip, args):
    raw = clip["hands"]
    pose = clip["pose"]
    ts = clip["ts"]
    clean = np.empty_like(raw)
    spikes = np.zeros(raw.shape[:2], bool)
    estimated = np.zeros(raw.shape[:2], bool)
    mask_interp = np.zeros(raw.shape[:2], bool)

    for s in range(2):  # 0 = 왼손, 1 = 오른손
        h, spikes[:, s] = remove_spikes(raw[:, s], args.spike_k)
        h = fill_gaps(h, ts, args.max_gap)
        mask_interp[:, s] = ~np.isnan(h[:, 0, 0])
        h, estimated[:, s] = pose_anchor_fill(h, pose, s, ts, args.anchor_gap, args.anchor_move)
        clean[:, s] = smooth(h, ts, args.min_cutoff, args.beta)

    mask_clean = ~np.isnan(clean[:, :, 0, 0])
    raw_norm, _, _ = normalize(raw)
    norm, center, size = normalize(clean)
    return {
        "raw": raw, "clean": clean, "spikes": spikes, "estimated": estimated,
        "mask_raw": clip["mask"], "mask_interp": mask_interp, "mask_clean": mask_clean,
        "raw_norm": raw_norm, "norm": norm, "arm": normalize_arm(pose, center, size),
    }


def jitter(norm, valid):
    """
    프레임 간 가속도 크기 평균 (손 크기 단위 / frame²).
    t-1, t, t+1 이 모두 valid 인 구간만 사용. 값이 작을수록 덜 떨림.
    """
    acc = norm[2:] - 2 * norm[1:-1] + norm[:-2]            # (T-2,2,21,3)
    ok = valid[2:] & valid[1:-1] & valid[:-2]               # (T-2,2)
    if not ok.any():
        return np.nan
    mag = np.linalg.norm(acc, axis=-1).mean(axis=-1)        # (T-2,2)
    return float(mag[ok].mean())


def deviation(raw_norm, clean_norm, valid):
    """
    후처리 결과가 원본에서 평균적으로 얼마나 벗어났는지 (손 크기 단위).
    스무딩을 세게 할수록 jitter 는 줄지만 deviation(=지연/왜곡)은 커진다 → trade-off 확인용.
    """
    if not valid.any():
        return np.nan
    d = np.linalg.norm(raw_norm - clean_norm, axis=-1).mean(axis=-1)   # (T,2)
    return float(d[valid].mean())


# ============================================================
# Plots
# ============================================================
def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT2, labelsize=9)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bar(labels, series, title, ylabel, path, percent=False):
    """series: [(name, values, color), ...]"""
    x = np.arange(len(labels))
    n = len(series)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    for i, (name, vals, color) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w - 0.02, color=color, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel, color=C_TEXT2, fontsize=10)
    ax.set_title(title, color=C_TEXT, fontsize=12, loc="left", pad=12)
    if percent:
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    style_axes(ax)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for t in leg.get_texts():
        t.set_color(C_TEXT)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def example_plot(name, res, ts, path):
    """검출률이 낮은 클립: 검지 끝 y 좌표의 시간 변화 (왼손/오른손)."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.2), dpi=200, sharex=True)
    for s, ax in enumerate(axes):
        missing = ~res["mask_raw"][:, s]
        # 원본 결측 구간 음영 (연속 구간을 하나로 합쳐서, 이웃 프레임과의 중간점까지)
        mids = np.concatenate([[ts[0]], (ts[1:] + ts[:-1]) / 2, [ts[-1]]])
        t = 0
        while t < len(ts):
            if missing[t]:
                start = t
                while t + 1 < len(ts) and missing[t + 1]:
                    t += 1
                ax.axvspan(mids[start], mids[t + 1], color=C_GAP, lw=0)
            t += 1
        spikes = res["spikes"][:, s]
        ax.plot(ts, res["raw"][:, s, INDEX_TIP, 1], color=C_RAW, lw=2,
                marker="o", ms=4, label="Raw")
        ax.plot(ts, res["clean"][:, s, INDEX_TIP, 1], color=C_CLEAN, lw=2, label="Post-processed")
        if spikes.any():
            ax.plot(ts[spikes], res["raw"][spikes, s, INDEX_TIP, 1], "x", color=C_TEXT,
                    ms=8, mew=2, label="Spike removed")
        est = res["estimated"][:, s]
        if est.any():
            ax.plot(ts[est], res["clean"][est, s, INDEX_TIP, 1], "D", color=C_POSE,
                    ms=6, mec="#ffffff", mew=1.5, label="Estimated from arm (pose)")
        ax.invert_yaxis()  # 이미지 좌표: 아래가 +
        ax.set_ylabel(f"{'Left' if s == 0 else 'Right'} index tip y", color=C_TEXT2, fontsize=9)
        style_axes(ax)
    axes[-1].set_xlabel("time (s)   [gray = hand not detected in raw]", color=C_TEXT2, fontsize=9)
    axes[0].set_title(f"Example clip: {name}", color=C_TEXT, fontsize=12, loc="left", pad=12)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for hnd, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(hnd)
                labels.append(lab)
    leg = fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(C_TEXT)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(path)
    plt.close(fig)


def anchor_eval_plot(curves, path):
    """결측 길이별 추정 오차: 정지 / 움직임 두 패널. curves: {group: (title, gaps, {method: errs})}"""
    styles = {
        "hold": ("Hold last hand", C_RAW),
        "arm": ("Always move with pose wrist", C_CLEAN),
        "adaptive": ("Adaptive (used)", C_POSE),
    }
    fig, axes = plt.subplots(1, len(curves), figsize=(11, 4), dpi=150)
    for ax, (title, gaps, errs) in zip(axes, curves.values()):
        for m, (name, color) in styles.items():
            ax.plot(gaps, errs[m], color=color, lw=2, marker="o", ms=8, mec="#ffffff", mew=2, label=name)
        ax.set_title(title, color=C_TEXT, fontsize=11, loc="left")
        ax.set_xlabel("simulated gap length (s, bin center)", color=C_TEXT2, fontsize=9)
        ax.set_ylim(bottom=0)
        style_axes(ax)
    axes[0].set_ylabel("mean landmark error (hand-size)", color=C_TEXT2, fontsize=9)
    leg = axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(C_TEXT)
    fig.suptitle("Missing-hand estimation error (lower is better)", color=C_TEXT, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def parse_args(argv=None):
    """argv=[] 로 부르면 기본값만 담긴 설정을 돌려준다 (다른 스크립트에서 재사용용)."""
    p = argparse.ArgumentParser(description="Hand sign keypoint post-processing")
    p.add_argument("--max-gap", type=float, default=0.75, help="보간할 최대 결측 길이(초)")
    p.add_argument("--spike-k", type=float, default=0.5, help="튐 판정 기준 (손 크기 배수)")
    p.add_argument("--min-cutoff", type=float, default=1.5, help="One-Euro min cutoff (Hz)")
    p.add_argument("--beta", type=float, default=3.0, help="One-Euro beta")
    p.add_argument("--anchor-gap", type=float, default=2.0,
                   help="팔 기반 추정에 쓸 기준 프레임까지 최대 거리(초). 0 = 사용 안 함")
    p.add_argument("--anchor-move", type=float, default=0.5,
                   help="pose 손목이 손 크기의 몇 배 이상 움직여야 이동을 반영할지")
    return p.parse_args(argv)


def main():
    args = parse_args()
    paths = sorted(IN_DIR.glob("*/*.npz"))
    if not paths:
        raise RuntimeError(f"No clips found in {IN_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(paths)} clips  |  max_gap={args.max_gap}s  spike_k={args.spike_k}  "
          f"min_cutoff={args.min_cutoff}  beta={args.beta}  anchor_gap={args.anchor_gap}s")

    rows = []
    results = {}
    anchor_err = {"seal": {}, "none": {}}   # 정지(인 유지) vs 움직임(none)
    for path in paths:
        clip = load_clip(path)
        res = process_clip(clip, args)
        label = clip["label"]

        group = "none" if label == NONE_LABEL else "seal"
        for b, per_m in eval_anchor(clip["hands"], clip["pose"], clip["ts"], args.anchor_move).items():
            for m, errs in per_m.items():
                anchor_err[group].setdefault(b, {}).setdefault(m, []).extend(errs)

        out_dir = OUT_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / path.name,
            raw=res["raw"].reshape(-1, 42, 3).astype(np.float32),
            clean=res["clean"].reshape(-1, 42, 3).astype(np.float32),
            norm=res["norm"].reshape(-1, 42, 3).astype(np.float32),
            raw_norm=res["raw_norm"].reshape(-1, 42, 3).astype(np.float32),
            arm=res["arm"].astype(np.float32),
            mask_raw=res["mask_raw"], mask_interp=res["mask_interp"],
            mask_clean=res["mask_clean"], estimated=res["estimated"], spikes=res["spikes"],
            timestamps=clip["ts"].astype(np.float32), fps=np.float32(clip["fps"]),
            img_size=clip["img_size"], label=label,
        )

        row = {
            "clip": path.stem, "label": label, "frames": len(clip["ts"]),
            "both_raw": res["mask_raw"].all(1).mean(),
            "both_interp": res["mask_interp"].all(1).mean(),
            "both_clean": res["mask_clean"].all(1).mean(),
            "spikes": int(res["spikes"].sum()),
            "estimated": int(res["estimated"].sum()),
            # 같은 프레임 집합(원본에서 검출된 프레임)에서 비교
            "jitter_raw": jitter(res["raw_norm"], res["mask_raw"]),
            "jitter_clean": jitter(res["norm"], res["mask_raw"] & res["mask_clean"]),
            "deviation": deviation(res["raw_norm"], res["norm"],
                                   res["mask_raw"] & res["mask_clean"] & ~res["spikes"]),
        }
        rows.append(row)
        results[path.stem] = (res, clip["ts"])

    # ---- summary.csv ----
    with open(REPORT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- per-sign aggregate ----
    labels = sorted({r["label"] for r in rows} - {NONE_LABEL})
    agg = {}
    print(f"\n{'sign':8s} {'clips':>5s} {'both raw':>9s} {'+interp':>8s} {'+arm':>7s} {'est.fr':>7s} "
          f"{'spikes':>7s} {'jitter raw':>11s} {'jitter post':>12s} {'reduction':>10s} {'deviation':>10s}")
    for lab in labels + ([NONE_LABEL] if any(r["label"] == NONE_LABEL for r in rows) else []):
        rs = [r for r in rows if r["label"] == lab]
        a = {
            "n": len(rs),
            "both_raw": np.mean([r["both_raw"] for r in rs]) * 100,
            "both_interp": np.mean([r["both_interp"] for r in rs]) * 100,
            "both_clean": np.mean([r["both_clean"] for r in rs]) * 100,
            "estimated": sum(r["estimated"] for r in rs),
            "spikes": sum(r["spikes"] for r in rs),
            "jitter_raw": np.nanmean([r["jitter_raw"] for r in rs]),
            "jitter_clean": np.nanmean([r["jitter_clean"] for r in rs]),
            "deviation": np.nanmean([r["deviation"] for r in rs]),
        }
        agg[lab] = a
        red = (1 - a["jitter_clean"] / a["jitter_raw"]) * 100 if a["jitter_raw"] else np.nan
        print(f"{lab:8s} {a['n']:5d} {a['both_raw']:8.1f}% {a['both_interp']:7.1f}% "
              f"{a['both_clean']:6.1f}% {a['estimated']:7d} {a['spikes']:7d} {a['jitter_raw']:11.4f} {a['jitter_clean']:12.4f} {red:9.1f}% "
              f"{a['deviation']:10.4f}")

    # ---- plots (none 제외) ----
    grouped_bar(
        labels,
        [("Raw", [agg[l]["both_raw"] for l in labels], C_RAW),
         ("+ Interpolation", [agg[l]["both_interp"] for l in labels], C_CLEAN),
         ("+ Arm-anchored estimate", [agg[l]["both_clean"] for l in labels], C_POSE)],
        "Both-hands valid rate per sign", "frames with both hands", REPORT_DIR / "detection_rate.png",
        percent=True,
    )
    grouped_bar(
        labels,
        [("Raw", [agg[l]["jitter_raw"] for l in labels], C_RAW),
         ("Post-processed", [agg[l]["jitter_clean"] for l in labels], C_CLEAN)],
        "Jitter per sign (lower is smoother)", "mean |acceleration| (hand-size / frame²)",
        REPORT_DIR / "jitter.png",
    )

    # ---- 팔 기반 추정 검증 ----
    curves = {}
    for g, title in (("seal", "holding a sign (static)"), ("none", "moving hands (none clips)")):
        bs = [b for b in sorted(anchor_err[g]) if len(anchor_err[g][b]["hold"]) >= 30]
        centers = [(GAP_BINS[b] + GAP_BINS[b + 1]) / 2 for b in bs]
        curves[g] = (title, centers,
                     {m: [np.mean(anchor_err[g][b][m]) for b in bs] for m in EVAL_METHODS})
        print(f"\nMissing-hand estimation error, {title} (hand-size units)")
        print(f"{'gap (s)':>9s} {'n':>6s} " + " ".join(f"{m:>9s}" for m in EVAL_METHODS))
        for i, b in enumerate(bs):
            print(f"{GAP_BINS[b]:.1f}-{GAP_BINS[b + 1]:.1f} {len(anchor_err[g][b]['hold']):6d} "
                  + " ".join(f"{curves[g][2][m][i]:9.3f}" for m in EVAL_METHODS))
    anchor_eval_plot(curves, REPORT_DIR / "anchor_eval.png")
    with open(REPORT_DIR / "anchor_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "gap_center_s"] + list(EVAL_METHODS))
        for g, (_, centers, errs) in curves.items():
            for i, c in enumerate(centers):
                w.writerow([g, f"{c:.1f}"] + [f"{errs[m][i]:.4f}" for m in EVAL_METHODS])
    # 예시 클립: 검출률 30% 이상 중 가장 나쁜 것 (0% 클립은 그릴 게 없어서 제외)
    seal_rows = [r for r in rows if r["label"] != NONE_LABEL and r["both_raw"] >= 0.3]
    worst = min(seal_rows, key=lambda r: r["both_raw"])
    res, ts = results[worst["clip"]]
    example_plot(worst["clip"], res, ts, REPORT_DIR / f"example_{worst['clip']}.png")

    print(f"\nProcessed clips → {OUT_DIR}")
    print(f"Report          → {REPORT_DIR}")


if __name__ == "__main__":
    main()
