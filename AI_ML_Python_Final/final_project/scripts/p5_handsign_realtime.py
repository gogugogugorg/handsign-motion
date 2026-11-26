"""
p5_handsign_realtime.py
웹캠으로 인을 실시간 인식하고, 정해진 순서대로 인을 맺으면 술법 이펙트를 띄운다.

실행 (final_project 폴더에서):
    python scripts/p5_handsign_realtime.py
    python scripts/p5_handsign_realtime.py --complexity 0 --record demo.mp4   # 화면 녹화 (보고서/시연용)
    python scripts/p5_handsign_realtime.py --video a.mp4 b.mp4 --headless   # 녹화 영상으로 테스트

조작:
    R : 인 순서 초기화      S : 스켈레톤 표시 켜기/끄기      Q / ESC : 종료

처리 흐름 (프레임마다):
    Holistic → 손별 추적기 (사라지면 조건부 팔 기반 추정, One-Euro 스무딩)
    → 정규화 + 특징 (학습과 같은 함수) → MLP 확률
    → SealDetector: 최근 window 프레임 평균 확률이 conf 이상으로 hold 초 유지되면 인 확정
    → JutsuMatcher: 확정된 인 순서가 술법과 일치하면 이펙트 발동 (timeout 초 동안 다음 인이 없으면 초기화)
"""

import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import p1_3_handsign_post_processing as pp  # noqa: E402
from p3_handsign_train import HandSignMLP  # noqa: E402
from utils.handsign_features import POSE_ELBOW, POSE_WRIST, frame_features  # noqa: E402

# ------------------------------------------------------------
# 술법 정의 (여기만 고치면 새 술법 추가 가능)
# effect: "fire" | "lightning"
# ------------------------------------------------------------
JUTSUS = {
    "fireball": {"name": "화둔 · 호화구의 술",
                 "seals": ["snake", "ram", "monkey", "boar", "horse", "tiger"], "effect": "fire"},
    "chidori": {"name": "치도리",
                "seals": ["ox", "hare", "monkey"], "effect": "lightning"},
}

SEAL_KR = {
    "none": "-", "rat": "쥐", "ox": "소", "tiger": "호랑이", "hare": "토끼", "dragon": "용",
    "snake": "뱀", "horse": "말", "ram": "양", "monkey": "원숭이", "bird": "닭", "dog": "개", "boar": "돼지",
}
NONE_LABEL = "none"
DEFAULT_MODEL = ROOT / "checkpoints" / "handsign_mlp.pth"
FONT_PATHS = ["C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf"]

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
rng = np.random.default_rng()


# ============================================================
# 1) 손 추적 (p1_3 후처리의 실시간 버전)
# ============================================================
class HandTracker:
    """
    한 손의 상태를 유지.
      검출됨      → 그대로 사용, '마지막 정상 손'과 그때의 pose 손목/팔꿈치 저장
      안 잡힘     → max_gap 초 동안은 마지막 정상 손을 기준으로 조건부 팔 기반 추정
      그보다 오래 → 결측 (필터 리셋)
    결과는 One-Euro 필터로 스무딩.
    """

    def __init__(self, side, max_gap, min_move, min_cutoff, beta):
        self.side = side
        self.max_gap, self.min_move = max_gap, min_move
        self.filter = pp.OneEuroFilter(min_cutoff, beta)
        self.last = None

    def update(self, hand, pose, t):
        pw, pe = pose[POSE_WRIST[self.side]], pose[POSE_ELBOW[self.side]]
        if hand is not None:
            self.last = {"hand": hand, "t": t, "pw": pw.copy(), "pe": pe.copy(),
                         "size": float(pp.hand_size(hand[None])[0])}
            return self.filter(hand, t), "detected"

        if self.last is not None and t - self.last["t"] <= self.max_gap:
            L = self.last
            if pw[3] >= pp.VIS_TH and L["pw"][3] >= pp.VIS_TH:
                h = pp.estimate_from_arm(L["hand"], L["pw"][:2], L["pe"][:2], pw[:2], pe[:2],
                                         L["size"], self.min_move)
            else:
                h = L["hand"]
            return self.filter(h, t), "estimated"

        self.filter.reset()
        return None, "missing"


# ============================================================
# 2) 인 확정
# ============================================================
class SealDetector:
    def __init__(self, labels, window, conf, hold):
        self.labels, self.conf, self.hold = labels, conf, hold
        self.none_idx = labels.index(NONE_LABEL)
        self.buf = deque(maxlen=window)
        self.cand, self.cand_since = None, 0.0
        self.last_confirmed = None
        self.avg = np.zeros(len(labels))

    def update(self, probs, t):
        """확정된 인 이름을 반환 (없으면 None)."""
        self.buf.append(probs)
        self.avg = np.mean(self.buf, axis=0)
        top = int(self.avg.argmax())
        label, p = self.labels[top], float(self.avg[top])

        if label != self.cand:
            self.cand, self.cand_since = label, t
        stable = p >= self.conf and t - self.cand_since >= self.hold

        if label == NONE_LABEL:
            if stable:
                self.last_confirmed = None   # none 이 안정적으로 나오면 같은 인 다시 허용
            return None
        if stable and label != self.last_confirmed:
            self.last_confirmed = label
            return label
        return None

    def top(self):
        i = int(self.avg.argmax())
        return self.labels[i], float(self.avg[i])

    def hold_progress(self, t):
        label, p = self.top()
        if label == NONE_LABEL or p < self.conf or label == self.last_confirmed:
            return 0.0
        return min(1.0, (t - self.cand_since) / self.hold)


# ============================================================
# 3) 술법 매칭
# ============================================================
class JutsuMatcher:
    def __init__(self, jutsus, timeout):
        self.jutsus, self.timeout = jutsus, timeout
        self.seq, self.last_t = [], None

    def expire(self, t):
        if self.seq and t - self.last_t > self.timeout:
            self.seq = []

    def add(self, seal, t):
        """인 추가. 술법이 완성되면 그 key 반환."""
        self.expire(t)
        self.seq.append(seal)
        self.last_t = t
        for key, j in self.jutsus.items():
            n = len(j["seals"])
            if self.seq[-n:] == j["seals"]:
                self.seq = []
                return key
        return None

    def progress(self, key):
        """현재 순서의 끝부분이 술법의 앞부분과 몇 개 일치하는지."""
        seals = self.jutsus[key]["seals"]
        for k in range(min(len(seals), len(self.seq)), 0, -1):
            if self.seq[-k:] == seals[:k]:
                return k
        return 0

    def remaining(self, t):
        return max(0.0, self.timeout - (t - self.last_t)) if self.seq else 0.0


# ============================================================
# 4) 이펙트 (원본 프레임 좌표에 그린 뒤 화면을 거울 반전)
# ============================================================
class FireEffect:
    duration, emit_time = 3.2, 2.4

    def __init__(self, now, scale):
        self.t0 = self.t_prev = now
        self.s = scale
        self.p = np.zeros((0, 6))  # x, y, vx, vy, age, life

    def step(self, img, ctx, now):
        dt = min(now - self.t_prev, 0.2)
        self.t_prev = now
        el = now - self.t0
        ox, oy = ctx["fire"]
        if el < self.emit_time:
            n = int(900 * dt) + 10
            ang = rng.uniform(0, 2 * np.pi, n)
            spd = rng.uniform(120, 420, n) * self.s
            new = np.column_stack([
                np.full(n, ox) + rng.normal(0, 10, n), np.full(n, oy) + rng.normal(0, 10, n),
                np.cos(ang) * spd, np.sin(ang) * spd * 0.7 - 60 * self.s,
                np.zeros(n), rng.uniform(0.6, 1.2, n)])
            self.p = np.vstack([self.p, new])
        p = self.p
        p[:, 0] += p[:, 2] * dt
        p[:, 1] += p[:, 3] * dt
        p[:, 3] -= 140 * self.s * dt          # 불꽃은 위로
        p[:, 4] += dt
        self.p = p[p[:, 4] < p[:, 5]]

        # 불꽃 색은 덮어 그리기(alpha), 빛은 더하기(glow)
        # 속도를 위해 1/2 해상도로 그리고 블러한 뒤 확대 (불꽃은 흐릿해서 화질 차이가 거의 없음)
        H, W = img.shape[:2]
        q = 0.5
        color = np.zeros((int(H * q), int(W * q), 3), np.uint8)
        alpha = np.zeros(color.shape[:2], np.float32)
        for x, y, _, _, age, life in self.p[np.argsort(-self.p[:, 4])]:   # 오래된 것부터 → 새 불꽃이 위
            f = age / life
            r = max(1, int((10 + 70 * f ** 0.6) * self.s * q))
            if f < 0.2:
                col = (120, 245, 255)         # 노랑-흰 (BGR)
            elif f < 0.5:
                col = (20, 160, 255)          # 주황
            else:
                col = (20, 50, 220)           # 빨강
            c = (int(x * q), int(y * q))
            cv2.circle(color, c, r, col, -1)
            cv2.circle(alpha, c, r, float(0.9 * (1 - f) ** 0.8), -1)
        color = cv2.GaussianBlur(color, (0, 0), 4 * self.s * q)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 6 * self.s * q)
        glow = cv2.GaussianBlur((color * alpha[..., None] * 0.5).astype(np.uint8), (0, 0), 18 * self.s * q)
        color = cv2.resize(color, (W, H))
        alpha = cv2.resize(alpha, (W, H))[..., None]
        glow = cv2.resize(glow, (W, H))
        img[:] = (img * (1 - alpha) + color * alpha).astype(np.uint8)
        img[:] = cv2.add(img, glow)
        return el < self.duration


class LightningEffect:
    duration = 3.5

    def __init__(self, now, scale):
        self.t0 = now
        self.s = scale

    def _bolt(self, c, length):
        ang = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(ang), np.sin(ang)])
        nrm = np.array([-d[1], d[0]])
        n = rng.integers(6, 10)
        pts = [np.array(c, float)]
        for i in range(1, n + 1):
            base = np.array(c) + d * length * i / n
            pts.append(base + nrm * rng.normal(0, length * 0.08))
        return np.array(pts, np.int32)

    def step(self, img, ctx, now):
        el = now - self.t0
        cx, cy = ctx["right_hand"]
        img[:] = (img * 0.75).astype(np.uint8)        # 주변을 어둡게
        overlay = np.zeros_like(img)
        cv2.circle(overlay, (int(cx), int(cy)), int(rng.uniform(40, 60) * self.s), (255, 190, 110), -1)
        bolts = [self._bolt((cx, cy), rng.uniform(60, 170) * self.s) for _ in range(rng.integers(6, 10))]
        for b in bolts:
            cv2.polylines(overlay, [b], False, (255, 200, 120), max(2, int(6 * self.s)))
        glow = cv2.GaussianBlur(overlay, (0, 0), 9 * self.s)
        img[:] = cv2.add(img, glow)
        img[:] = cv2.add(img, glow)
        for b in bolts:
            cv2.polylines(img, [b], False, (255, 255, 255), max(1, int(2 * self.s)), cv2.LINE_AA)
        cv2.circle(img, (int(cx), int(cy)), int(14 * self.s), (255, 255, 255), -1, cv2.LINE_AA)
        return el < self.duration


class FlashEffect:
    duration = 0.35

    def __init__(self, now, scale):
        self.t0 = now

    def step(self, img, ctx, now):
        a = max(0.0, 1 - (now - self.t0) / self.duration) * 0.8
        img[:] = cv2.addWeighted(img, 1 - a, np.full_like(img, 255), a, 0)
        return now - self.t0 < self.duration


EFFECTS = {"fire": FireEffect, "lightning": LightningEffect}


# ============================================================
# 5) HUD (한글은 PIL 로)
# ============================================================
class TextRenderer:
    def __init__(self):
        self.fonts = {}
        self.path = next((p for p in FONT_PATHS if os.path.exists(p)), None)

    def font(self, size):
        if size not in self.fonts:
            self.fonts[size] = (ImageFont.truetype(self.path, size) if self.path
                                else ImageFont.load_default())
        return self.fonts[size]

    def draw(self, img, items):
        """items: [(text, (x, y), size, (r, g, b), anchor)]  anchor: PIL 앵커 (예 'la', 'mm')"""
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for text, xy, size, color, anchor in items:
            d.text(xy, text, font=self.font(size), fill=color, anchor=anchor,
                   stroke_width=max(2, size // 12), stroke_fill=(0, 0, 0))
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_hud(img, tr, det, matcher, status, fps, t, banner, show_help):
    H, W = img.shape[:2]
    s = H / 480
    items = []

    # 현재 인식 + 확률 막대 + 확정 진행도
    label, p = det.top()
    items.append((f"인식: {SEAL_KR.get(label, label)}  {p:.0%}", (int(12 * s), int(10 * s)), int(22 * s),
                  (255, 255, 255), "la"))
    x0, y0, bw, bh = int(12 * s), int(42 * s), int(180 * s), int(8 * s)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (60, 60, 60), -1)
    cv2.rectangle(img, (x0, y0), (x0 + int(bw * p), y0 + bh), (230, 125, 37), -1)
    hp = det.hold_progress(t)
    if hp > 0:
        cv2.rectangle(img, (x0, y0 + bh + 3), (x0 + int(bw * hp), y0 + bh + 6), (148, 184, 0), -1)

    # 확정된 인 순서
    seq = " → ".join(SEAL_KR[x] for x in matcher.seq) if matcher.seq else "(인을 맺으세요)"
    items.append((seq, (int(12 * s), int(62 * s)), int(20 * s), (255, 230, 120), "la"))
    if matcher.seq:
        items.append((f"{matcher.remaining(t):.1f}s", (int(12 * s), int(88 * s)), int(14 * s),
                      (200, 200, 200), "la"))

    # 술법 진행도 (오른쪽 위)
    yy = int(10 * s)
    for key, j in matcher.jutsus.items():
        k, n = matcher.progress(key), len(j["seals"])
        items.append((j["name"], (W - int(12 * s), yy), int(16 * s),
                      (255, 255, 255) if k else (170, 170, 170), "ra"))
        nxt = SEAL_KR[j["seals"][k]] if k < n else ""
        dots = "●" * k + "○" * (n - k)
        items.append((f"{dots}  다음: {nxt}" if k else dots, (W - int(12 * s), yy + int(20 * s)),
                      int(13 * s), (255, 200, 80) if k else (140, 140, 140), "ra"))
        yy += int(46 * s)

    # 손 상태 + fps (아래)
    col = {"detected": (0, 220, 0), "estimated": (255, 200, 0), "missing": (255, 60, 60)}
    items.append(("L", (int(12 * s), H - int(30 * s)), int(16 * s), col[status[0]], "la"))
    items.append(("R", (int(30 * s), H - int(30 * s)), int(16 * s), col[status[1]], "la"))
    items.append((f"{fps:.1f} fps", (int(52 * s), H - int(28 * s)), int(13 * s), (200, 200, 200), "la"))
    if show_help:
        items.append(("R: 초기화  S: 스켈레톤  Q: 종료", (W - int(12 * s), H - int(28 * s)), int(13 * s),
                      (200, 200, 200), "ra"))

    # 술법 이름 배너
    if banner is not None:
        name, t_start = banner
        if t - t_start < 2.5:
            items.append((name, (W // 2, int(H * 0.2)), int(40 * s), (255, 240, 200), "mm"))

    tr.draw(img, items)


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Real-time hand sign recognition")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--backend", default="msmf", choices=["msmf", "dshow"],
                   help="웹캠 드라이버. 이 PC 에선 dshow 가 8.3fps 로 묶이고 msmf 는 30fps")
    p.add_argument("--video", nargs="+", help="웹캠 대신 영상 파일(들)을 순서대로 입력")
    p.add_argument("--video-fps", type=float, default=None,
                   help="영상 입력 시 시간 = 프레임번호 / 이 값 (재생 속도와 무관하게 원래 타이밍으로 판정)")
    p.add_argument("--start-frame", type=int, default=0, help="영상 입력 시작 프레임")
    p.add_argument("--end-frame", type=int, default=None, help="영상 입력 끝 프레임")
    p.add_argument("--mirrored", action="store_true", help="입력 영상이 좌우 반전돼 있으면 (p5 녹화본) 되돌림")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--conf", type=float, default=0.8, help="인 확정 최소 확률")
    p.add_argument("--hold", type=float, default=0.3, help="인 확정까지 유지 시간(초)")
    p.add_argument("--window", type=int, default=3, help="확률 평균 프레임 수")
    p.add_argument("--timeout", type=float, default=5.0, help="다음 인까지 허용 시간(초)")
    p.add_argument("--max-gap", type=float, default=0.75, help="손이 안 잡혀도 추정으로 버틸 시간(초)")
    p.add_argument("--anchor-move", type=float, default=0.5)
    p.add_argument("--min-cutoff", type=float, default=1.5)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--record", help="화면을 mp4 로 저장")
    p.add_argument("--headless", action="store_true", help="창 없이 실행 (테스트용)")
    return p.parse_args()


def frame_source(args):
    """(frame) 제너레이터: 웹캠 또는 영상 파일들."""
    if args.video:
        for path in args.video:
            cap = cv2.VideoCapture(path)
            i = 0
            while args.end_frame is None or i < args.end_frame:
                if i < args.start_frame:
                    ok = cap.grab()
                    i += 1
                    if not ok:
                        break
                    continue
                ok, frame = cap.read()
                i += 1
                if not ok:
                    break
                yield cv2.flip(frame, 1) if args.mirrored else frame
            cap.release()
        return
    reader = LatestFrameReader(open_camera(args.camera, args.width, args.height, args.backend))
    try:
        while True:
            frame = reader.read()
            if frame is None:
                break
            yield frame
    finally:
        reader.stop()


def open_camera(index, width, height, backend="msmf"):
    """웹캠 열기. msmf 실패 시 dshow 로 재시도."""
    order = [backend, "dshow" if backend == "msmf" else "msmf"] if os.name == "nt" else ["any"]
    api = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}
    for name in order:
        cap = cv2.VideoCapture(index, api[name])
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"Camera backend: {name}")
            return cap
        cap.release()
    raise RuntimeError(f"웹캠을 열 수 없습니다 (camera={index})")


class LatestFrameReader:
    """
    별도 스레드에서 카메라를 계속 읽고 '가장 최근 프레임'만 보관.
    처리(약 16fps)가 카메라(30fps)보다 느려도 프레임이 쌓여 화면이 늦어지지 않는다.
    """

    def __init__(self, cap):
        self.cap = cap
        self.frame, self.seq, self.last_seq = None, 0, 0
        self.cond = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            with self.cond:
                if not ok:
                    self.running = False
                else:
                    self.frame, self.seq = f, self.seq + 1
                self.cond.notify_all()

    def read(self, timeout=2.0):
        """이전에 준 것보다 새로운 프레임이 올 때까지 기다렸다가 반환."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq > self.last_seq or not self.running, timeout)
            if self.seq == self.last_seq:
                return None
            self.last_seq = self.seq
            return self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def lm_array(lms, n, cols=3):
    if lms is None:
        return None
    if cols == 3:
        return np.array([[l.x, l.y, l.z] for l in lms.landmark], np.float64)
    return np.array([[l.x, l.y, l.z, l.visibility] for l in lms.landmark], np.float64)


def main():
    args = parse_args()
    try:
        # Windows 콘솔/로그에서 한글 깨짐 방지 + 줄 단위 출력 (로그 파일로 보낼 때도 바로 기록되게)
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except (AttributeError, ValueError):
        pass

    ckpt = torch.load(args.model, map_location="cpu")
    labels, use_arm = ckpt["labels"], ckpt["use_arm"]
    model = HandSignMLP(ckpt["feature_dim"], len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    none_onehot = np.eye(len(labels))[labels.index(NONE_LABEL)]
    print(f"Model: {Path(args.model).name} | {len(labels)} classes | arm features: {use_arm}")
    for j in JUTSUS.values():
        print(f"  {j['name']}: " + " → ".join(SEAL_KR[s] for s in j["seals"]))

    holistic = mp_holistic.Holistic(model_complexity=args.complexity, smooth_landmarks=True,
                                    refine_face_landmarks=False,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
    trackers = [HandTracker(s, args.max_gap, args.anchor_move, args.min_cutoff, args.beta) for s in range(2)]
    det = SealDetector(labels, args.window, args.conf, args.hold)
    matcher = JutsuMatcher(JUTSUS, args.timeout)
    tr = TextRenderer()

    effects, banner, effect_fps = [], None, []
    show_skel = True
    writer = None
    fps, t_prev = None, time.time()
    t_start = t_prev

    try:
        for frame_i, frame in enumerate(frame_source(args)):
            if args.video and args.video_fps:
                t = t_start + (args.start_frame + frame_i) / args.video_fps   # 영상 기준 시간
            else:
                t = time.time()
            inst = 1.0 / max(t - t_prev, 1e-6)
            t_prev = t
            fps = inst if fps is None else 0.9 * fps + 0.1 * inst

            H, W = frame.shape[:2]
            ar = W / H
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

            # ---- 좌표 (① 비율 보정) ----
            pose_n = lm_array(res.pose_landmarks, 33, cols=4)
            pose = np.zeros((33, 4)) if pose_n is None else pose_n.copy()
            pose[:, 0] *= ar
            pose[:, 2] *= ar
            hands_n = [lm_array(res.left_hand_landmarks, 21), lm_array(res.right_hand_landmarks, 21)]
            clean = np.full((1, 2, 21, 3), np.nan)
            status = []
            for s in range(2):
                h = None
                if hands_n[s] is not None:
                    h = hands_n[s].copy()
                    h[:, 0] *= ar
                    h[:, 2] *= ar
                out, st = trackers[s].update(h, pose, t)
                status.append(st)
                if out is not None:
                    clean[0, s] = out

            # ---- 분류 ----
            mask = ~np.isnan(clean[:, :, 0, 0])
            if mask.all():
                norm, center, size = pp.normalize(clean)
                arm = pp.normalize_arm(pose[None], center, size) if use_arm else None
                x = frame_features(norm.reshape(1, 42, 3), mask, arm)
                with torch.no_grad():
                    probs = torch.softmax(model(torch.from_numpy(x)), 1)[0].numpy()
            else:
                probs = none_onehot   # 두 손이 다 있어야 인으로 판단

            matcher.expire(t)
            seal = det.update(probs, t)
            if seal is not None:
                print(f"[{t - t_start:6.1f}s] seal: {SEAL_KR[seal]} ({seal})")
                done = matcher.add(seal, t)
                if done is not None:
                    j = JUTSUS[done]
                    print(f"[{t - t_start:6.1f}s] *** {j['name']} ***  (frame {frame_i})")
                    scale = H / 480
                    effects += [FlashEffect(t, scale), EFFECTS[j["effect"]](t, scale)]
                    banner = (j["name"], t)

            # ---- 이펙트 위치 (원본 픽셀 좌표) ----
            def px(i):
                return pose_n[i, 0] * W, pose_n[i, 1] * H
            if pose_n is not None:
                mouth = np.mean([px(9), px(10)], axis=0)
                hands_c = np.mean([px(15), px(16)], axis=0)
                # 불꽃: 입과 손 사이 (입이 화면 밖이어도 보이도록), 화면 안으로 제한
                fire = mouth * 0.6 + hands_c * 0.4
                fire = (float(np.clip(fire[0], 0, W)), float(np.clip(fire[1], H * 0.15, H * 0.85)))
                rh = (tuple(hands_n[1][:, :2].mean(0) * [W, H]) if hands_n[1] is not None else px(16))
            else:
                fire, rh = (W / 2, H / 2), (W / 2, H / 2)
            ctx = {"fire": fire, "right_hand": rh}

            view = frame.copy()
            if show_skel:
                for lm, c in ((res.left_hand_landmarks, (255, 160, 0)), (res.right_hand_landmarks, (0, 160, 255))):
                    mp_draw.draw_landmarks(view, lm, mp_holistic.HAND_CONNECTIONS,
                                           mp_draw.DrawingSpec(c, 1, 2), mp_draw.DrawingSpec(c, 1, 1))
            had_effects = bool(effects)
            effects = [e for e in effects if e.step(view, ctx, t)]
            # 이펙트 동안의 fps 기록 (끝날 때 로그로 출력)
            if had_effects:
                effect_fps.append(inst)
                if not effects:
                    print(f"[{t - t_start:6.1f}s] effect fps: min {min(effect_fps):.1f}  "
                          f"avg {np.mean(effect_fps):.1f}")
                    effect_fps = []
            view = cv2.flip(view, 1)
            draw_hud(view, tr, det, matcher, status, fps, t, banner, not args.headless)

            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(5.0, round(fps)), (W, H))
                writer.write(view)

            if not args.headless:
                cv2.imshow("Hand Sign Jutsu", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    matcher.seq = []
                    det.last_confirmed = None
                    print(f"[{t - t_start:6.1f}s] --- reset (R) ---")   # 로그 구간 표시용
                if key == ord("s"):
                    show_skel = not show_skel
    finally:
        holistic.close()
        if writer is not None:
            writer.release()
            print(f"Recorded → {args.record}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
