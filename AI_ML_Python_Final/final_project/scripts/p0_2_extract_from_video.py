"""
p0_2_extract_from_video.py
이미 찍어 둔 영상(예: p5 실시간 세션 녹화본)에서 구간을 잘라 키포인트를 뽑고,
p0_record_hand_signs.py 와 같은 형식의 클립(npz)으로 저장한다.
주 용도: 인을 맺지 않은 평소 행동 구간 → none 클래스 추가 (오인식 줄이기)

실행 (final_project 폴더에서):
    python scripts/p0_2_extract_from_video.py data/output/demo.mp4 --label none ^
        --fps 8.31 --start 0 --end 2280 --mirrored --stride 2 --clip-sec 10

옵션:
    --fps       영상 속 실제 진행 속도 (p5 녹화본은 파일에 적힌 fps 가 실제와 다를 수 있음)
    --mirrored  p5 녹화본처럼 좌우 반전된 화면이면 다시 뒤집어서 분석 (좌/우 손 라벨 보존)
    --stride    N 프레임마다 1개만 사용 (연속 프레임은 거의 같아서 중복이 많음)
    --clip-sec  저장할 클립 길이 (p2 가 클립 단위로 train/val 을 나누므로 너무 길지 않게)

주의: p5 녹화본에는 손 스켈레톤 선과 HUD 가 그려져 있다. 손 위의 선은 얇아서 영향이 작겠지만 0 은 아니다.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from p0_record_hand_signs import KEYPOINT_DIR, hand_to_array, pose_to_array, to_long_format  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--label", default="none")
    p.add_argument("--fps", type=float, required=True, help="영상의 실제 진행 fps")
    p.add_argument("--start", type=float, default=0.0, help="시작(초)")
    p.add_argument("--end", type=float, default=None, help="끝(초)")
    p.add_argument("--mirrored", action="store_true")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--clip-sec", type=float, default=10.0)
    p.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--prefix", default=None, help="저장 파일 이름 앞부분 (기본: <label>_<영상이름>)")
    return p.parse_args()


def save_clip(out_dir, stem, buf, fps, img_size, label):
    hands = np.asarray(buf["hands"], np.float32)
    mask = np.asarray(buf["mask"], bool)
    pose = np.asarray(buf["pose"], np.float32)
    np.savez(
        out_dir / f"{stem}.npz",
        data=to_long_format(hands, mask, pose),
        hands=hands, hand_mask=mask, pose=pose,
        pose_world=np.asarray(buf["pose_world"], np.float32),
        timestamps=np.asarray(buf["ts"], np.float32), fps=np.float32(fps),
        img_size=np.array(img_size, np.int32), label=label,
    )
    return len(hands), mask.any(1).mean(), mask.all(1).mean()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {args.video}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_start = int(args.start * args.fps)
    f_end = min(n_total, int(args.end * args.fps)) if args.end is not None else n_total
    per_clip = max(1, int(args.clip_sec * args.fps / args.stride))
    prefix = args.prefix or f"{args.label}_{Path(args.video).stem}"
    out_dir = KEYPOINT_DIR / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.video}: {n_total} frames | using {f_start}~{f_end} (stride {args.stride}) "
          f"→ ~{(f_end - f_start) // args.stride} frames, {per_clip} per clip → {out_dir}")

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=args.complexity, smooth_landmarks=True,
        refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    buf = {k: [] for k in ("hands", "mask", "pose", "pose_world", "ts")}
    img_size, n_clips, n_frames = None, 0, 0
    t0 = time.time()

    i = 0
    while i < f_end:
        ok = cap.grab()                       # 디코딩만 (stride 로 건너뛸 프레임은 retrieve 안 함)
        if not ok:
            break
        if i >= f_start and (i - f_start) % args.stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if args.mirrored:
                frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            img_size = (W, H)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)
            left, l_ok = hand_to_array(res.left_hand_landmarks)
            right, r_ok = hand_to_array(res.right_hand_landmarks)
            buf["hands"].append(np.concatenate([left, right], 0))
            buf["mask"].append([l_ok, r_ok])
            buf["pose"].append(pose_to_array(res.pose_landmarks))
            buf["pose_world"].append(pose_to_array(res.pose_world_landmarks))
            buf["ts"].append((i - f_start) / args.fps)

            if len(buf["hands"]) >= per_clip:
                t_clip = buf["ts"][0] + args.start
                stem = f"{prefix}_{int(t_clip):05d}s"
                T, any_r, both_r = save_clip(out_dir, stem, buf, args.fps / args.stride, img_size, args.label)
                n_clips += 1
                n_frames += T
                el = time.time() - t0
                done = (i - f_start) / max(1, f_end - f_start)
                print(f"  [{done:5.1%}] {stem}  any hand {any_r:4.0%}  both {both_r:4.0%}  "
                      f"(elapsed {el / 60:.1f} min, ETA {el / max(done, 1e-6) * (1 - done) / 60:.1f} min)",
                      flush=True)
                buf = {k: [] for k in buf}
        i += 1

    if len(buf["hands"]) >= per_clip // 2:
        stem = f"{prefix}_{int(buf['ts'][0] + args.start):05d}s"
        T, _, _ = save_clip(out_dir, stem, buf, args.fps / args.stride, img_size, args.label)
        n_clips += 1
        n_frames += T

    holistic.close()
    cap.release()
    print(f"\nSaved {n_clips} clips ({n_frames} frames) → {out_dir}")


if __name__ == "__main__":
    main()
