# HandSignViewer

손 인(印) 데이터의 **원본(MediaPipe)** 과 **후처리 결과**를 Unity 에서 나란히 재생하는 시각화 프로젝트.
손은 별도 에셋 없이 도형으로 만든다 (관절 = 구, 뼈 = 원기둥, 손바닥 = 면).

- Unity 6 (6000.6.0f1), Built-in Render Pipeline, 기존 Input Manager
- 수업 프로젝트(`AI_ML_Unity_Final`)와 분리된 별도 프로젝트. 스크립트는 일반 C# 이라 2022.3 에서도 동작

## 사용법

1. 손 데이터 내보내기 (Python, `AI_ML_Python_Final/final_project` 폴더에서)
   ```
   python scripts/p7_export_unity_hands.py
   ```
   → `Assets/StreamingAssets/HandSign/<클립>_raw.csv`, `<클립>_clean.csv`
2. Unity Hub 에서 이 폴더(`HandSignViewer`)를 프로젝트로 추가해 열기
3. `Assets/Scenes/HandSignViewer.unity` 를 열고 ▶ Play
   (장면이 없으면 메뉴 **HandSign → Create Scene**)

| 키 | 동작 |
|---|---|
| ← / → | 클립 변경 |
| Space | 일시정지 |
| , / . | 한 프레임 이동 |

## 화면 읽는 법

- 왼쪽 **RAW (주황)**: MediaPipe 가 검출한 그대로. 검출 안 된 손은 그리지 않음 (`MISSING`)
- 오른쪽 **POST-PROCESSED (청록)**: 튐 제거 · 보간 · 조건부 팔 추정 · One-Euro 스무딩 결과
  - **노란색** 손 = 원본에서 사라졌던 손을 보간 / 팔 기반 추정으로 채운 프레임 (`FILLED`)
- 좌표: 클립 전체의 관절 영역 중심 = 원점, 클립 손 크기(손목~중지 뿌리, 3D) 중앙값 = 9cm 로 정규화.
  손끝이 카메라를 향하는 인도 크기가 부풀지 않도록 z 를 포함한 길이를 사용

## 파일

| 파일 | 내용 |
|---|---|
| `Assets/Scripts/HandModel.cs` | 21개 관절로 도형 손 만들기 / 매 프레임 자세 갱신 |
| `Assets/Scripts/HandSignPlayer.cs` | CSV 로드, 원본/후처리 나란히 재생, 라벨 |
| `Assets/Editor/HandSignSetup.cs` | 장면 자동 생성, batchmode 스크린샷 (`-executeMethod HandSignSetup.CaptureBatch -captureDir <폴더> -captures "clip:frame,..."`) |
