"""
p0_record_hand_signs.py
웹캠으로 인(hand sign)을 녹화해서 키포인트 + 원본 영상을 인 이름별 폴더에 저장한다.

실행 (final_project 폴더에서):
    python scripts/p0_record_hand_signs.py
    python scripts/p0_record_hand_signs.py --signs none snake ram monkey --duration 5

조작:
    A / D      : 이전 / 다음 인 선택
    SPACE      : 카운트다운 후 녹화 시작 (녹화 중 다시 누르면 즉시 중지 후 저장)
    (시험용 데이터는 --out data/handsign_test 로 따로 저장 → 학습에 섞이지 않음)
    U          : 방금 저장한 클립 삭제 (실행 취소, 여러 번 가능)
    Q / ESC    : 종료

저장 위치:
    data/handsign_keypoints/<인>/<인>_<날짜시간>.npz
    data/handsign_videos/<인>/<인>_<날짜시간>.mp4   (--no-video 로 끌 수 있음)

npz 내용:
    hands      (T, 42, 3)  0~20 = 왼손, 21~41 = 오른손 (사람 기준 좌/우, MediaPipe Holistic)
                           x, y 는 이미지 기준 0~1 정규화 좌표 (y 아래쪽이 +), z 는 손목 기준 상대 깊이
                           검출 실패한 손은 0 으로 채움 → hand_mask 로 구분
    hand_mask  (T, 2)      [왼손 검출 여부, 오른손 검출 여부]
    pose       (T, 33, 4)  전신 이미지 좌표 (x, y, z, visibility)  - 손이 가려질 때 손목/팔꿈치 보조용
    pose_world (T, 33, 4)  전신 3D 월드 좌표 (미터, 골반 중심)
    timestamps (T,)        녹화 시작 기준 초
    fps, img_size [W, H], label
    data       (T*75, 6)   기존 코드와 같은 long format [frame, landmark, x, y, z, visibility]
                           landmark 0~41 = 손, 42~74 = 전신

주의: x 는 이미지 너비, y 는 높이로 정규화돼 있어 가로세로 비율이 다르다.
      거리/각도 계산 전에 x 에 (W / H) 를 곱해서 비율을 맞출 것.
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KEYPOINT_DIR = ROOT / "data" / "handsign_keypoints"
VIDEO_DIR = ROOT / "data" / "handsign_videos"

# 12지 인 + 인이 아닌 상태(none)
# 쥐, 소, 호랑이, 토끼, 용, 뱀, 말, 양, 원숭이, 닭, 개, 돼지
DEFAULT_SIGNS = [
    "none", "rat", "ox", "tiger", "hare", "dragon", "snake",
    "horse", "ram", "monkey", "bird", "dog", "boar",
]

N_HAND = 21
N_POSE = 33
MIN_FRAMES = 10          # 이보다 짧은 클립은 버림
WARN_DETECTION = 0.7     # 두 손 검출률이 이보다 낮으면 경고

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Record hand-sign keypoints from webcam")
    p.add_argument("--camera", type=int, default=0, help="웹캠 번호")
    p.add_argument("--signs", nargs="+", default=DEFAULT_SIGNS, help="녹화할 인 이름 목록")
    p.add_argument("--duration", type=float, default=4.0, help="클립 하나 녹화 시간(초)")
    p.add_argument("--countdown", type=float, default=2.0, help="녹화 전 카운트다운(초)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-video", action="store_true", help="원본 mp4 저장 안 함")
    p.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                   help="전신 모델 크기 (0 = 가장 빠름). 손 모델에는 영향 없음")
    p.add_argument("--out", default=None,
                   help="저장 폴더 (기본: data/handsign_keypoints). 시험용은 data/handsign_test 처럼 따로")
    return p.parse_args()


def hand_to_array(lms):
    if lms is None:
        return np.zeros((N_HAND, 3), np.float32), False
    return np.array([[l.x, l.y, l.z] for l in lms.landmark], np.float32), True


def pose_to_array(lms):
    if lms is None:
        return np.zeros((N_POSE, 4), np.float32)
    return np.array([[l.x, l.y, l.z, l.visibility] for l in lms.landmark], np.float32)


def count_clips(label):
    d = KEYPOINT_DIR / label
    return len(list(d.glob("*.npz"))) if d.exists() else 0


def to_long_format(hands, hand_mask, pose):
    """(T,42,3)+(T,2)+(T,33,4) → (T*75, 6) [frame, landmark, x, y, z, visibility]"""
    T = hands.shape[0]
    hand_vis = np.repeat(hand_mask.astype(np.float32), N_HAND, axis=1)   # (T,42)
    xyz = np.concatenate([hands, pose[:, :, :3]], axis=1)                # (T,75,3)
    vis = np.concatenate([hand_vis, pose[:, :, 3]], axis=1)              # (T,75)
    n = xyz.shape[1]
    frame = np.repeat(np.arange(T), n)
    landmark = np.tile(np.arange(n), T)
    return np.column_stack(
        [frame, landmark, xyz.reshape(-1, 3), vis.reshape(-1)]
    ).astype(np.float32)


def open_camera(index, width, height):
    """
    Windows 에선 MSMF 를 먼저 시도 (이 PC 에서 DSHOW 는 8.3fps 로 묶이고 MSMF 는 30fps).
    실패하면 DSHOW 로 재시도.
    """
    backends = [("msmf", cv2.CAP_MSMF), ("dshow", cv2.CAP_DSHOW)] if os.name == "nt" else [("any", cv2.CAP_ANY)]
    for name, api in backends:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"Camera backend: {name}")
            return cap
        cap.release()
    raise RuntimeError(f"웹캠을 열 수 없습니다 (camera={index})")


def put_text(img, text, org, scale=0.8, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ------------------------------------------------------------
# Recorder
# ------------------------------------------------------------
class ClipRecorder:
    def __init__(self, save_video, cam_fps, img_size):
        self.save_video = save_video
        self.cam_fps = cam_fps
        self.img_size = img_size  # (W, H)
        self.history = []         # 저장한 (npz, mp4) 경로 → undo 용
        self._reset()

    def _reset(self):
        self.label = None
        self.hands, self.mask, self.pose, self.pose_world, self.ts = [], [], [], [], []
        self.t0 = None
        self.writer = None
        self.stem = None

    def start(self, label, loop_fps=None):
        """loop_fps: 실제 처리 속도. MediaPipe 때문에 카메라 fps 보다 느리므로
        이 값으로 영상을 저장해야 재생 속도가 실제와 같다."""
        self._reset()
        self.label = label
        self.t0 = time.time()
        self.stem = f"{label}_{datetime.now():%Y%m%d_%H%M%S}"
        if self.save_video:
            vdir = VIDEO_DIR / label
            vdir.mkdir(parents=True, exist_ok=True)
            self.video_path = vdir / f"{self.stem}.mp4"
            fps = loop_fps if loop_fps and loop_fps > 1 else self.cam_fps
            self.writer = cv2.VideoWriter(
                str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                fps, self.img_size,
            )

    def add(self, frame_bgr, left, l_ok, right, r_ok, pose, pose_world):
        self.hands.append(np.concatenate([left, right], axis=0))
        self.mask.append([l_ok, r_ok])
        self.pose.append(pose)
        self.pose_world.append(pose_world)
        self.ts.append(time.time() - self.t0)
        if self.writer is not None:
            self.writer.write(frame_bgr)

    def elapsed(self):
        return time.time() - self.t0

    def finish(self):
        """저장하고 결과 메시지 반환"""
        if self.writer is not None:
            self.writer.release()

        T = len(self.hands)
        video_path = self.video_path if self.save_video else None

        if T < MIN_FRAMES:
            if video_path is not None and video_path.exists():
                video_path.unlink()
            self._reset()
            return f"Too short ({T} frames) - discarded", (0, 0, 255)

        hands = np.asarray(self.hands, np.float32)
        mask = np.asarray(self.mask, bool)
        pose = np.asarray(self.pose, np.float32)
        pose_world = np.asarray(self.pose_world, np.float32)
        ts = np.asarray(self.ts, np.float32)
        fps = (T - 1) / max(ts[-1] - ts[0], 1e-6)

        kdir = KEYPOINT_DIR / self.label
        kdir.mkdir(parents=True, exist_ok=True)
        npz_path = kdir / f"{self.stem}.npz"
        np.savez(
            npz_path,
            data=to_long_format(hands, mask, pose),
            hands=hands, hand_mask=mask, pose=pose, pose_world=pose_world,
            timestamps=ts, fps=np.float32(fps),
            img_size=np.array(self.img_size, np.int32), label=self.label,
        )
        self.history.append((npz_path, video_path))

        both = mask.all(axis=1).mean()
        print(f"[saved] {npz_path.name} | {T} frames | {fps:.1f} fps | "
              f"L {mask[:, 0].mean():.0%}  R {mask[:, 1].mean():.0%}  both {both:.0%}")

        label = self.label
        self._reset()
        if label != "none" and both < WARN_DETECTION:
            return f"Saved, but both-hands detected only {both:.0%} (U = undo)", (0, 165, 255)
        return f"Saved {npz_path.name} ({T} frames, both hands {both:.0%})", (0, 255, 0)

    def undo(self):
        if not self.history:
            return "Nothing to undo", (0, 165, 255)
        npz_path, video_path = self.history.pop()
        for p in (npz_path, video_path):
            if p is not None and p.exists():
                p.unlink()
        print(f"[deleted] {npz_path.name}")
        return f"Deleted {npz_path.name}", (0, 165, 255)


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
def main():
    global KEYPOINT_DIR, VIDEO_DIR
    args = parse_args()
    if args.out:
        out = Path(args.out)
        KEYPOINT_DIR = out if out.is_absolute() else ROOT / out
        VIDEO_DIR = KEYPOINT_DIR.parent / (KEYPOINT_DIR.name + "_videos")
        print(f"Saving to {KEYPOINT_DIR}  (videos: {VIDEO_DIR})")
    signs = args.signs
    idx = 0

    cap = open_camera(args.camera, args.width, args.height)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("웹캠에서 프레임을 읽을 수 없습니다")
    H, W = frame.shape[:2]
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if cam_fps <= 1:
        cam_fps = 30.0
    print(f"Camera: {W}x{H} @ {cam_fps:.0f}fps")
    print("A/D: change sign | SPACE: record | U: undo | Q: quit")

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=args.complexity,
        smooth_landmarks=True,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    rec = ClipRecorder(not args.no_video, cam_fps, (W, H))

    state = "idle"          # idle → countdown → recording → idle
    t_state = 0.0
    message, msg_color, msg_until = "", (255, 255, 255), 0.0
    loop_fps, t_prev = None, time.time()   # 실제 처리 fps (지수이동평균)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t_now = time.time()
            inst = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            loop_fps = inst if loop_fps is None else 0.9 * loop_fps + 0.1 * inst

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

            left, l_ok = hand_to_array(res.left_hand_landmarks)
            right, r_ok = hand_to_array(res.right_hand_landmarks)
            pose = pose_to_array(res.pose_landmarks)
            pose_world = pose_to_array(res.pose_world_landmarks)

            now = time.time()

            # ---- state transitions ----
            if state == "countdown" and now - t_state >= args.countdown:
                state = "recording"
                rec.start(signs[idx], loop_fps)

            if state == "recording":
                rec.add(frame, left, l_ok, right, r_ok, pose, pose_world)  # 원본(그리기 전) 프레임 저장
                if rec.elapsed() >= args.duration:
                    message, msg_color = rec.finish()
                    msg_until = now + 3
                    state = "idle"

            # ---- drawing (원본 위에 그린 뒤 거울 모드로 반전) ----
            view = frame.copy()
            mp_draw.draw_landmarks(view, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec((120, 120, 120), 1, 1),
                                   mp_draw.DrawingSpec((120, 120, 120), 1, 1))
            mp_draw.draw_landmarks(view, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((255, 160, 0), 2, 3),
                                   mp_draw.DrawingSpec((255, 220, 120), 2, 1))
            mp_draw.draw_landmarks(view, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 160, 255), 2, 3),
                                   mp_draw.DrawingSpec((120, 220, 255), 2, 1))
            view = cv2.flip(view, 1)

            # ---- HUD ----
            label = signs[idx]
            put_text(view, f"[{idx + 1}/{len(signs)}] {label}   clips: {count_clips(label)}",
                     (20, 40), 1.0, (0, 255, 255))
            put_text(view, "L hand", (20, 80), 0.7, (0, 255, 0) if l_ok else (0, 0, 255))
            put_text(view, "R hand", (120, 80), 0.7, (0, 255, 0) if r_ok else (0, 0, 255))
            put_text(view, f"{loop_fps:.1f} fps", (220, 80), 0.7, (200, 200, 200))

            if state == "countdown":
                remain = args.countdown - (now - t_state)
                put_text(view, f"{int(np.ceil(remain))}", (W // 2 - 30, H // 2), 4.0, (0, 255, 255), 6)
                put_text(view, f"Get ready: {label}", (W // 2 - 150, H // 2 + 70), 1.0)
            elif state == "recording":
                cv2.circle(view, (W - 40, 40), 15, (0, 0, 255), -1)
                put_text(view, f"REC {rec.elapsed():.1f}/{args.duration:.0f}s",
                         (W - 260, 50), 0.9, (0, 0, 255))
            else:
                put_text(view, "A/D: sign  SPACE: record  U: undo  Q: quit",
                         (20, H - 20), 0.7, (200, 200, 200))

            if now < msg_until:
                put_text(view, message, (20, 120), 0.7, msg_color)

            cv2.imshow("Hand Sign Recorder", view)

            # ---- keys ----
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == 32:  # SPACE
                if state == "idle":
                    state, t_state = "countdown", now
                elif state == "countdown":
                    state = "idle"
                elif state == "recording":
                    message, msg_color = rec.finish()
                    msg_until = now + 3
                    state = "idle"
            elif state == "idle" and key == ord("a"):
                idx = (idx - 1) % len(signs)
            elif state == "idle" and key == ord("d"):
                idx = (idx + 1) % len(signs)
            elif state == "idle" and key == ord("u"):
                message, msg_color = rec.undo()
                msg_until = now + 3
    finally:
        if state == "recording":
            rec.finish()
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()

    print("\nClip counts:")
    for s in signs:
        print(f"  {s:8s}: {count_clips(s)}")


if __name__ == "__main__":
    main()
