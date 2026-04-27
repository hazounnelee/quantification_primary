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

| 값 | 입자 형태 | 전구체 특성 |
|----|-----------|------------|
| `acicular` | 침상 (바늘 모양) | **대입경** 전구체 — 긴 needle이 방사형 배열 |
| `plate` | 판상 (판 모양) | **소입경** 전구체 — 납작한 flake 구조 |

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
| `scale_pixels` | 276 | 184 | 276 | 184 |
| `scale_um` | 50 | 10 | 50 | 10 |
| `particle_mode` | acicular | acicular | auto | auto |
| `auto_detect_sphere` | True | False | True | False |
| `center_crop_ratio` | 0.60 | 0.85 | 0.60 | 0.85 |
| `tile_size` | 256 | 192 | 256 | 192 |
| `stride` | 128 | 96 | 128 | 96 |
| `points_per_tile` | 120 | 150 | 100 | 100 |
| `point_min_distance` | 8 | 5 | 10 | 8 |
| `area_threshold` | 200 | 80 | 300 | 150 |

---

## 알고리즘 전체 흐름

```
SEM 이미지 입력
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1 : ROI 추출                                          │
│                                                             │
│  20k 이미지 (--auto_detect_sphere)                          │
│    Gaussian blur → Otsu threshold                           │
│    → morphology close/open → 최대 contour                   │
│    → 최소 외접원 → 상단 cap 영역 추출                         │
│    검출 실패 시 center crop으로 자동 fallback                │
│                                                             │
│  50k 이미지 (center crop)                                   │
│    이미지 중앙 85% 직사각형 사용                             │
│    (입자가 화면 전체를 채우므로 더 넓게)                      │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2 : SAM2 Segmentation                                 │
│                                                             │
│  [particle_mode = acicular]  ← 침상 전용 hybrid             │
│                                                             │
│  ① OpenCV 침상 후보 탐지                                    │
│     CLAHE → adaptive threshold (block=35, C=4)             │
│     → 외곽 contour 탐지                                     │
│     → elongation 필터: minAreaRect AR < 0.60               │
│     → 패딩 8% 적용 → 후보 bbox 목록                         │
│                                                             │
│  ② SAM2 box prompt                                         │
│     후보 bbox를 SAM2에 직접 전달 (배치 16개씩)              │
│     → 각 bbox당 정밀 binary mask 생성                       │
│                                                             │
│  ③ 후보 부족(< 3개) 시 point prompt fallback               │
│     ROI를 tile_size 타일로 분할                             │
│     각 타일: CLAHE + Laplacian → Shi-Tomasi 코너 검출       │
│     → SAM2 point prompt (배치 32개씩)                       │
│                                                             │
│  [particle_mode = auto]  ← 판상용, 범용                     │
│  ROI를 tile_size 타일로 분할                                │
│  각 타일: Shi-Tomasi 코너 → SAM2 point prompt               │
└─────────────────────────────────────────────────────────────┘
       │  binary masks (N × H × W)
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3 : 중복 제거                                         │
│                                                             │
│  1차) bbox IoU ≥ 0.85 → 중복 mask 제거                     │
│  2차) mask IoU ≥ 0.60 → 픽셀 단위 중복 제거                │
│  ROI 경계 margin(8px) 이내 bbox → 가장자리 효과 제거        │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4 : 두께 측정 (각 mask 독립 처리)                     │
│                                                             │
│  cv2.minAreaRect(contour)                                   │
│    → 회전 보정된 최소 외접 직사각형                          │
│    → 단축 = thickness (두께)                                │
│    → 장축 = long axis (길이)                                │
│                                                             │
│  aspect_ratio = thickness / long_axis  (0 < AR ≤ 1)        │
│    AR < 0.40 → 침상 (acicular)   ─ 파란색 overlay          │
│    AR ≥ 0.40 → 판상 (plate)      ─ 초록색 overlay          │
│    면적 < area_threshold → fragment ─ 주황색 overlay        │
│                                                             │
│  픽셀 → µm 변환:                                            │
│    µm = px × (scale_um / scale_pixels)                     │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5 : 결과 저장                                         │
│                                                             │
│  01_input.png          원본 이미지                          │
│  02_input_roi.png      분석 ROI                             │
│  03_overlay_roi.png    색상 오버레이 (ROI 기준)              │
│  04_overlay_full.png   전체 이미지에 ROI 표시               │
│  05_opencv_candidates  OpenCV 침상 탐지 결과 (acicular 모드) │
│  06_sphere_detection   구 검출 결과 (20k + auto_detect)     │
│  objects.csv           전체 입자 측정값 (두께, 장축, AR 등) │
│  acicular.csv / plate.csv  형태별 분리 저장                 │
│  thickness_dist.png    두께 분포 histogram                  │
│  summary.json          통계 요약 + 분석 설정 기록           │
│  debug.json            타일/포인트/bbox 디버그 정보         │
└─────────────────────────────────────────────────────────────┘
```

---

## 사용 예시

### 침상 입자 / 20000배 (구형 전체 이미지) — 권장

```bash
python measure.py \
  --input sem_20k.jpg \
  --particle_type acicular \
  --magnification 20k \
  --model model/sam2.1_hiera_base_plus.pt \
  --model_cfg model/sam2.1_hiera_t.yaml \
  --output_dir out_acicular_20k
```

### 침상 입자 / 50000배 (표면 클로즈업)

```bash
python measure.py \
  --input sem_50k.jpg \
  --particle_type acicular \
  --magnification 50k \
  --model model/sam2.1_hiera_base_plus.pt \
  --model_cfg model/sam2.1_hiera_t.yaml \
  --output_dir out_acicular_50k
```

### 판상 입자 / 50000배

```bash
python measure.py \
  --input sem_plate_50k.jpg \
  --particle_type plate \
  --magnification 50k \
  --model model/sam2.1_hiera_base_plus.pt \
  --model_cfg model/sam2.1_hiera_t.yaml
```

### 개별 파라미터 override (preset 기반 + 수동 조정)

```bash
python measure.py \
  --input sem_20k.jpg \
  --particle_type acicular \
  --magnification 20k \
  --sphere_cap_fraction 0.55 \    # preset(0.45)를 0.55로 override
  --acicular_threshold 0.35 \     # 더 엄격한 침상 기준
  --model model/sam2.1_hiera_base_plus.pt \
  --model_cfg model/sam2.1_hiera_t.yaml
```

### 배치 처리

```bash
python measure.py \
  --input ./samples/ \            # IMG_ID 서브폴더 구조 지원
  --particle_type acicular \
  --magnification 20k \
  --model model/sam2.1_hiera_base_plus.pt \
  --model_cfg model/sam2.1_hiera_t.yaml \
  --output_dir out_batch
```

---

## 스케일 바 설정 (preset에 자동 포함)

| 배율 | scale_pixels | scale_um | 환산 |
|------|-------------|----------|------|
| 20k (`--magnification 20k`) | 276 | 50 | 1 px ≈ 0.181 µm |
| 50k (`--magnification 50k`) | 184 | 10 | 1 px ≈ 0.054 µm |

실제 이미지의 스케일 바가 다를 경우 `--scale_pixels`와 `--scale_um`으로 직접 지정합니다.

---

## 촬영 팁

| 배율 | 권장 ROI | 두께 측정 원리 |
|------|---------|--------------|
| 20k | 구 상단 cap — 입자가 측면으로 배열 | minAreaRect 단축 = 두께 |
| 50k | 표면 균일 영역 | minAreaRect 단축 = 두께 |

- 구 적도 부근: 입자가 측면 정면으로 → 길이와 두께 모두 측정 가능
- 구 상단(pole): 입자 단면이 보임 → 단면 폭 ≈ 두께 (직접 측정 가능)

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
