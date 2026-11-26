"""
handsign_features.py
인 분류 모델의 입력 특징 계산. 학습(p2/p3)과 실시간 인식이 반드시 같은 함수를 써야 한다.

입력: norm (N,42,3)  p1_3 의 정규화 좌표 (0~20 왼손, 21~41 오른손, 결측 = 0)
      mask (N,2)     손별 유효 여부
      arm  (N,10,2)  (선택) 전신 모델의 팔 점들, 같은 기준으로 정규화 (ARM_POINTS 순서, 결측 = 0)
출력: (N, feature_dim(use_arm)) float32
      [손 좌표 126] + [손 유효 여부 2] + [왼손 손끝 ↔ 오른손 손끝 거리 25] (+ [팔 좌표 20])
"""

import numpy as np

N_HAND = 21
FINGER_TIPS = [4, 8, 12, 16, 20]   # 엄지, 검지, 중지, 약지, 새끼 끝

# MediaPipe Pose 인덱스 (사람 기준 좌/우)
POSE_ELBOW = (13, 14)
POSE_WRIST = (15, 16)
# 팔꿈치 L/R, 손목 L/R, 새끼 L/R, 검지 L/R, 엄지 L/R
ARM_POINTS = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

HAND_FEATURE_DIM = 42 * 3 + 2 + len(FINGER_TIPS) ** 2   # 153
ARM_FEATURE_DIM = len(ARM_POINTS) * 2                    # 20


def feature_dim(use_arm):
    return HAND_FEATURE_DIM + (ARM_FEATURE_DIM if use_arm else 0)


def frame_features(norm, mask, arm=None):
    norm = np.asarray(norm, np.float32)
    mask = np.asarray(mask, bool)
    N = norm.shape[0]

    coords = norm.reshape(N, -1)                                   # (N,126)

    left_tips = norm[:, FINGER_TIPS, :]                            # (N,5,3)
    right_tips = norm[:, [N_HAND + i for i in FINGER_TIPS], :]     # (N,5,3)
    dist = np.linalg.norm(left_tips[:, :, None, :] - right_tips[:, None, :, :], axis=-1)  # (N,5,5)
    both = mask.all(axis=1)
    dist[~both] = 0.0

    parts = [coords, mask.astype(np.float32), dist.reshape(N, -1)]
    if arm is not None:
        parts.append(np.asarray(arm, np.float32).reshape(N, -1))   # (N,20)
    return np.concatenate(parts, axis=1).astype(np.float32)


def augment(norm, mask, rng, arm=None, max_rot_deg=15.0, scale_range=(0.9, 1.1), noise_std=0.02):
    """
    학습용 증강 (정규화 좌표 기준). 손과 팔에 같은 회전/크기를 적용한다.
    z축(화면 평면) 회전 + 크기 + 가우시안 노이즈. 좌우 반전은 인 모양을 바꾸므로 하지 않는다.
    결측(0)은 0 으로 유지. 반환: (norm_aug, arm_aug 또는 None)
    """
    N = norm.shape[0]
    theta = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg, N))
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros((N, 3, 3), np.float32)
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1
    scale = rng.uniform(*scale_range, (N, 1, 1)).astype(np.float32)

    out = np.einsum("nij,nkj->nki", R, norm) * scale
    out = out + rng.normal(0, noise_std, out.shape).astype(np.float32)
    hand_valid = np.repeat(mask, N_HAND, axis=1)                   # (N,42)
    out[~hand_valid] = 0.0

    arm_out = None
    if arm is not None:
        arm = np.asarray(arm, np.float32)
        arm_valid = np.any(arm != 0, axis=-1)                      # (N,10)
        arm_out = np.einsum("nij,nkj->nki", R[:, :2, :2], arm) * scale
        arm_out = arm_out + rng.normal(0, noise_std, arm_out.shape).astype(np.float32)
        arm_out[~arm_valid] = 0.0
        arm_out = arm_out.astype(np.float32)
    return out.astype(np.float32), arm_out
