# 웹캠 손 인(印) 인식 — 모션 데이터 획득 · 후처리

AI/ML 기말 프로젝트 · Assignment 02 *3D Motion Data Post-Processing*

웹캠으로 12지 인(쥐 · 소 · 호랑이 · … · 돼지)을 맺으면 MediaPipe Holistic 으로 손 42점 + 전신 33점을 추출하고,
후처리한 뒤 MLP 로 인을 분류한다. 정해진 순서로 인을 맺으면 술법 이펙트(화둔 · 치도리)가 실시간으로 나타난다.

## 결과 요약

| 항목 | 결과 |
|---|---|
| 제공된 후처리 (발 기준 높이 보정) | 상반신 촬영이라 발 visibility 평균 0.004 → **손목 상하 떨림 +528%** (105/105 클립 악화) |
| 손 전용 후처리: 두 손 유효 프레임 | 가장 낮은 인 **65% → 84%**, 12개 인 중 9개 100% |
| 손 전용 후처리: 떨림 | 평균 **−59%** (인별 46~69%) |
| 인식 정확도 (프레임) | 검증 96.5% (학습과 같은 세션) / **시험 65.5%** (별도 세션 · 저조도: 밝기 12~33, 학습 58~87) |
| 실시간 | 딴짓 중 오인식 3.6 → 0 회/분 (none 데이터 보강 후), 웹캠 드라이버 교체로 8.3 → 약 15 fps |

그래프와 리포트 CSV 는 `data/*/_report/` 참고.

## 폴더 구조

```
AI_ML_UnityProject/
├── AI_ML_Python_Final/final_project/
│   ├── environment.yml
│   ├── scripts/
│   │   ├── p0_record_hand_signs.py          # 웹캠 녹화 (인 이름별 클립)
│   │   ├── p0_2_extract_from_video.py       # 녹화 영상 → 키포인트 (세션 영상에서 none 추출)
│   │   ├── p1_3_handsign_post_processing.py # 손 전용 후처리 + 전후 비교 그래프
│   │   ├── p1_4_provided_postprocess_pose.py# 제공된 높이 보정을 전신 3D 에 적용
│   │   ├── p2_handsign_dataset.py           # 프레임 단위 데이터셋 (클립 이름 해시로 train/val 고정)
│   │   ├── p3_handsign_train.py             # MLP 학습 (증강 + hand dropout)
│   │   ├── p5_handsign_realtime.py          # 실시간 인식 + 술법 이펙트
│   │   ├── p6_handsign_evaluate.py          # 시험 세트로 여러 모델 채점
│   │   ├── p7_export_unity_hands.py         # Unity 시각화용 CSV
│   │   └── p1_create_raw_keypoints.py, p1_2_post_processing.py, p2_create_dataset.py,
│   │       p3_train_mlp.py, p4_predict.py, ex2_mnist.py   # 수업 제공 참고 코드 (수정 없음)
│   ├── utils/handsign_features.py           # 특징 계산 (학습 · 실시간 공용)
│   ├── checkpoints/handsign_mlp*.pth        # 학습된 모델 (기본 = v4)
│   ├── data/
│   │   ├── handsign_keypoints/<인>/*.npz     # 녹화 원본 키포인트 (학습용)
│   │   ├── handsign_test/<인>/*.npz          # 시험 세트 (학습 미사용)
│   │   ├── handsign_heldout/none/*.npz       # 시험용 none
│   │   └── */_report/                        # 그래프 · 리포트 CSV
└── HandSignViewer/                           # Unity 6 시각화 프로젝트 (원본 vs 후처리 손 재생)
```

## 설치

```bash
cd AI_ML_Python_Final/final_project
conda env create -f environment.yml      # Python 3.10, mediapipe 0.10.21, torch ...
conda activate ai_ml_final
```

## 실행 순서

모든 명령은 `AI_ML_Python_Final/final_project` 폴더에서 실행.

| 단계 | 명령 | 결과 |
|---|---|---|
| 1. 녹화 | `python scripts/p0_record_hand_signs.py` | `data/handsign_keypoints/<인>/*.npz` (A/D 인 선택, SPACE 녹화) |
| 1-b. 영상에서 추출 | `python scripts/p0_2_extract_from_video.py <영상> --label none --fps 8.32 --mirrored` | 같은 형식의 npz |
| 2. 제공 후처리 적용 | `python scripts/p1_4_provided_postprocess_pose.py` | `data/handsign_processed/_report/provided/` |
| 3. 손 전용 후처리 | `python scripts/p1_3_handsign_post_processing.py` | `data/handsign_processed/` + 그래프 |
| 4. 데이터셋 | `python scripts/p2_handsign_dataset.py` | `data/handsign_dataset/` |
| 5. 학습 | `python scripts/p3_handsign_train.py` | `checkpoints/handsign_mlp.pth` |
| 6. 시험 세트 채점 | `python scripts/p6_handsign_evaluate.py` | `data/handsign_test/_report/` |
| 7. 실시간 | `python scripts/p5_handsign_realtime.py` | 웹캠 창 (R 초기화, S 스켈레톤, Q 종료) |
| 8. Unity 용 CSV | `python scripts/p7_export_unity_hands.py` | `HandSignViewer/Assets/StreamingAssets/HandSign/` |

녹화본에 포함된 데이터로 3~6 단계를 바로 다시 실행할 수 있다 (2~4 단계 중간 결과는 저장소에 없음).

### 실시간 술법

| 술법 | 인 순서 |
|---|---|
| 화둔 · 호화구의 술 | 뱀 → 양 → 원숭이 → 돼지 → 말 → 호랑이 |
| 치도리 | 소 → 토끼 → 원숭이 |

인을 0.3초 이상 (확률 0.8 이상) 유지하면 확정, 다음 인은 5초 안에. 옵션: `--conf`, `--hold`, `--timeout`, `--complexity 0`, `--record out.mp4`.

## 후처리 파이프라인 (손 전용)

1. **비율 보정** — x, z × W/H
2. **튐 제거** — 앞뒤 프레임 대비 손 크기 0.5배 이상 이탈하면 결측 처리
3. **선형 보간** — 0.75초 이하 결측 구간
4. **조건부 팔 기반 추정** — 전신 손목이 손 크기 0.5배 이상 움직였을 때만 그만큼 이동 (실험상 정지 상태에선 '직전 값 유지'가 더 정확)
5. **One-Euro 필터** — 과거 프레임만 사용 → 실시간에도 같은 처리
6. **정규화** — 손목 중점 = 원점, 손 크기 = 1

## Unity 시각화 (HandSignViewer)

Unity Hub 에서 `HandSignViewer` 폴더를 추가 → `Assets/Scenes/HandSignViewer.unity` 열고 Play.
왼쪽 = 원본(주황, 검출 안 된 손은 안 그림), 오른쪽 = 후처리(청록, 채운 손은 노랑). ← / → 클립, Space 정지.
자세한 내용은 [HandSignViewer/README.md](HandSignViewer/README.md).

## 저장소에 포함하지 않은 것

- **영상 전부** (얼굴 포함): 녹화 mp4, 실시간 세션 녹화본
- 수업 제공 Unity 프로젝트 (`AI_ML_Unity_Final`, 수정 없음)
- 다시 만들 수 있는 중간 결과: `data/handsign_processed/<인>/`, `data/handsign_dataset/*.npz`
- 보고서(PPT)와 그 생성 스크립트 · 캡처 (별도 제출)

## 한계

- 아주 어두우면 (화면 밝기 14/255) 밝기 보정 후에도 두 손 검출 1% — MediaPipe 의 조명 의존
- 검증 세트가 학습과 같은 세션이라 성능이 과대평가됨 (96.5% vs 시험 65.5%) → 다른 조건의 데이터 추가 필요
- 손이 깊게 맞물리는 인(돼지 등)은 전신 손목도 틀어져 팔 기반 추정이 빗나갈 수 있음
