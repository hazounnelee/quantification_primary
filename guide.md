# 1차 입자 두께 측정 가이드

## 파일 구성

| 파일 | 역할 |
|------|------|
| `core.py` | SAM2 추론 인프라 (모델 로드, 타일링, 포인트 샘플링, 중복 제거) |
| `measure.py` | 1차 입자 분석 메인 스크립트 (침상/판상 분류, 두께 측정, 결과 저장) |

---

## 핵심 선택: particle_type × magnification

실행 전 반드시 두 가지를 결정합니다.

### particle_type (입자 형태)

| 값 | 입자 형태 | 전구체 특성 | 측정 모드 |
|----|-----------|------------|---------|
| `acicular` | 침상 (바늘 모양) | **대입경** 전구체 — 긴 needle이 방사형 배열 | LSD 직접 측정 |
| `plate` | 판상 (판 모양) | **소입경** 전구체 — 납작한 flake 구조 | SAM2 segmentation |

> 침상은 대입경에서만, 판상은 소입경에서만 나타나므로 하나를 선택하면 나머지 파라미터가 자동 최적화됩니다.

### magnification (배율)

| 값 | 배율 | 이미지 특성 | ROI 전략 |
|----|------|------------|---------|
| `20k` | 20,000× | 구형 2차 입자 전체가 화면에 들어옴 | 구(sphere) 자동 검출 → cap 영역 |
| `50k` | 50,000× | 입자 표면 클로즈업 (위아래 잘림 가능) | center crop (85%) |

---

## 프리셋 파라미터 요약

`--particle_type`과 `--magnification`을 지정하면 아래 값이 자동 설정됩니다.  
CLI에서 개별 파라미터를 명시하면 preset보다 그 값이 우선합니다.

| 파라미터 | acicular/20k | acicular/50k | plate/20k | plate/50k |
|----------|-------------|-------------|----------|----------|
| `measure_mode` | **lsd** | **lsd** | sam2 | sam2 |
| `scale_pixels` | 276 | 184 | 276 | 184 |
| `scale_um` | 50 | 10 | 50 | 10 |
| `particle_mode` | acicular | acicular | auto | auto |
| `auto_detect_sphere` | True | False | True | False |
| `sphere_cap_fraction` | 0.65 | — | 0.65 | — |
| `center_crop_ratio` | 0.60 | 0.85 | 0.60 | 0.85 |
| `tile_size` | 192 | 192 | 192 | 192 |
| `stride` | 96 | 96 | 96 | 96 |
| `points_per_tile` | 120 | 150 | 100 | 100 |
| `point_min_distance` | 8 | 5 | 10 | 8 |
| `area_threshold` | 80 | 20 | 300 | 150 |

---

## 알고리즘 전체 흐름

### 침상 모드 (LSD)

```
SEM 이미지 입력
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1 : ROI 추출                                          │
│                                                             │
│  20k (auto_detect_sphere=True)                              │
│    Gaussian blur → Otsu threshold → morphology              │
│    → 최대 contour → 최소 외접원                              │
│    → 구 상단 cap 영역 (sphere_cap_fraction=0.65)            │
│    검출 실패 시 center crop으로 자동 fallback                │
│                                                             │
│  50k (center crop)                                          │
│    이미지 중앙 85% 직사각형 사용                             │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2 : LSD 침상 탐지                                     │
│                                                             │
│  CLAHE(clip=2.0) + GaussianBlur(3×3) → 전처리              │
│  cv2.createLineSegmentDetector(0).detect()                  │
│    → 선분 전체 (Image 7 기준 2089개)                         │
│                                                             │
│  1차 필터 (길이 + 대략적 AR)                                 │
│    length < 20px → 제거                                     │
│    lsd_width / length ≥ 0.60 → 제거 (너무 뚱뚱한 선분)      │
│                                                             │
│  중복 제거 (같은 침상에서 나온 복수 선분)                    │
│    center 거리 < 12px AND 각도 차이 < 25° → 짧은 것 제거    │
│    (Image 7: 2089 → 758 → 662 → 최종 217개)                 │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3 : 수직 프로파일 두께 측정                           │
│                                                             │
│  각 선분마다:                                               │
│    전역 Otsu threshold 계산 (전처리된 ROI 기준)             │
│    선분을 따라 7개 샘플 포인트 (t=0.2..0.8)                 │
│    각 포인트에서 수직 방향으로 ±0.5µm 스캔                   │
│    → intensity profile > Otsu threshold → 밝은 구간 탐지    │
│    → 스캔 중심에 가장 가까운 구간 선택 (이 침상의 엣지)      │
│    → 구간 폭 = 이 포인트에서의 두께                          │
│    7개 샘플의 중앙값 = 침상 두께                             │
│                                                             │
│  aspect_ratio = thickness / length                          │
│    AR < 0.40 → 침상 (acicular)                              │
│    AR ≥ 0.40 → 판상/fragment → particle_type=acicular면 버림│
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4 : 결과 저장                                         │
│                                                             │
│  01_input.png          원본 이미지                          │
│  02_input_roi.png      분석 ROI                             │
│  03_overlay_roi.png    색상 오버레이 (ROI 기준)              │
│  04_overlay_full.png   전체 이미지에 ROI 표시               │
│  05_opencv_candidates  LSD 탐지 결과                        │
│  06_sphere_detection   구 검출 결과 (20k)                   │
│  objects.csv           전체 입자 측정값 (두께, 장축, AR)    │
│  acicular.csv          침상 입자 분리 저장                   │
│  thickness_dist.png    두께 분포 histogram                  │
│  summary.json          통계 요약 + 분석 설정 기록           │
│  objects.json          입자별 상세 측정값                   │
│  debug.json            LSD 탐지 디버그 정보                 │
└─────────────────────────────────────────────────────────────┘
```

### 판상 모드 (SAM2)

판상 모드는 기존 SAM2 + point prompt 방식을 사용합니다.  
`--model` 과 `--model_cfg` 가 필요합니다.

---

## 사용 예시

### 침상 입자 / 20000배 (구형 전체 이미지)

```bash
python measure.py \
  --input sem_20k.jpg \
  --particle_type acicular \
  --magnification 20k \
  --output_dir out_acicular_20k
```

> SAM2 불필요 — LSD 모드는 `--model` 없이 실행됩니다.

### 침상 입자 / 50000배 (표면 클로즈업)

```bash
python measure.py \
  --input sem_50k.jpg \
  --particle_type acicular \
  --magnification 50k \
  --output_dir out_acicular_50k
```

### 판상 입자 / 50000배 (SAM2 필요)

```bash
python measure.py \
  --input sem_plate_50k.jpg \
  --particle_type plate \
  --magnification 50k \
  --model model/sam2.1_hiera_base_plus.pt \
  --model_cfg model/sam2.1_hiera_t.yaml
```

### 스케일 바 수동 지정 (preset과 다를 경우)

```bash
# 실제 이미지: 21,000배, 스케일 바 206px = 5µm
python measure.py \
  --input sem_21k.jpg \
  --particle_type acicular \
  --magnification 20k \
  --scale_pixels 206 \
  --scale_um 5 \
  --output_dir out_acicular_21k
```

### 배치 처리

```bash
python measure.py \
  --input ./samples/ \
  --particle_type acicular \
  --magnification 20k \
  --output_dir out_batch
```

---

## 스케일 바 설정

| 배율 | Preset 기본값 | 실측 예시 | 비고 |
|------|-------------|---------|------|
| 20k  | 276px = 50µm | 206px = 5µm (21,000× FW 24.8µm) | 기기/배율 따라 달라짐 |
| 50k  | 184px = 10µm | 311px = 2µm (79,000× FW 6.58µm) | 반드시 스케일 바 직접 측정 |

실제 이미지의 스케일 바 픽셀 수를 측정한 뒤 `--scale_pixels`와 `--scale_um`으로 지정하면 정확한 µm 환산이 가능합니다.

---

## 성능 비교

| 방식 | Image 6 (20k) | Image 7 (50k) | 처리 시간 |
|------|-------------|-------------|---------|
| SAM2 (이전) | 4개 | 10개 | 35–280초 |
| **LSD (현재)** | **78개** | **217개** | **< 2초** |

---

## core.py 주요 구성 요소

| 클래스/함수 | 역할 |
|-------------|------|
| `Sam2AspectRatioConfig` | 추론 파라미터 dataclass |
| `Sam2AspectRatioService` | SAM2 모델 초기화, 타일 추론, mask 후처리 |
| `enhance_image_texture()` | CLAHE + gradient + Laplacian 조합으로 texture 강화 |
| `sample_interest_points()` | Shi-Tomasi 코너 검출 + contour centroid fallback |
| `calculate_binary_iou()` | mask 기반 IoU 계산 |
| `calculate_box_iou()` | bounding box IoU 계산 |
| `create_processing_tiles()` | ROI를 겹치는 타일로 분할 |
| `collect_input_groups()` | 단일 파일 / IMG_ID 폴더 배치 입력 처리 |
