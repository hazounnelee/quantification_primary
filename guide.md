# 1차 입자 두께 측정 가이드

## 패키지 구조

```
primary/
├── primary_measure.py          CLI 진입점 (여기만 실행)
├── configs/
│   ├── __init__.py             get_analysis_preset() 제공
│   └── presets.yaml            4종 프리셋 정의
├── core/
│   └── schema.py               모든 dataclass 정의
├── models/
│   └── __init__.py             SAM2 모델 lazy-load (LSD 모드에서는 import 안 함)
├── services/
│   ├── sam2_service.py         SAM2 추론 인프라 (타일링, 포인트 샘플링, 중복 제거)
│   └── primary_particle.py     1차 입자 파이프라인 (ROI 추출, LSD/SAM2 분기, 저장)
└── utils/
    ├── image.py                구 검출, center crop, CLAHE/texture 강화, 타일 분할
    ├── iou.py                  binary IoU, box IoU
    ├── lsd.py                  LSD 탐지 + 수직 프로파일 두께 측정
    ├── metrics.py              단위 변환, 통계, JSON 직렬화
    └── io.py                   입력 파일/폴더 수집, 출력 디렉토리 생성
```

실행은 항상 `primary_measure.py`를 통해 한다.

---

## 핵심 선택: particle_type × magnification

실행 전 반드시 두 가지를 결정한다.

### particle_type (입자 형태)

| 값 | 입자 형태 | 전구체 특성 | 측정 모드 |
|----|-----------|------------|---------|
| `acicular` | 침상 (바늘 모양) | **대입경** 전구체 — 긴 needle이 방사형 배열 | LSD 직접 측정 |
| `plate` | 판상 (판 모양) | **소입경** 전구체 — 납작한 flake 구조 | SAM2 segmentation |

> 침상은 대입경에서만, 판상은 소입경에서만 나타나므로 하나를 선택하면 나머지 파라미터가 자동 최적화된다.

### magnification (배율)

| 값 | 배율 | 이미지 특성 | ROI 전략 |
|----|------|------------|---------|
| `20k` | 20,000× | 구형 2차 입자 전체가 화면에 들어옴 | 구(sphere) 자동 검출 → cap 영역 |
| `50k` | 50,000× | 입자 표면 클로즈업 (위아래 잘림 가능) | center crop (85%) |

---

## 프리셋 파라미터 요약

`--particle_type`과 `--magnification`을 지정하면 아래 값이 자동 설정된다.  
CLI에서 개별 파라미터를 명시하면 preset보다 그 값이 우선한다.

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
│  [utils/image.py]                                           │
│                                                             │
│  20k (auto_detect_sphere=True)                              │
│    Gaussian blur → Otsu threshold → morphology close/open   │
│    → 최대 contour → 최소 외접원 (cx, cy, r)                 │
│    → 구 상단 cap 영역 y∈[cy-r, cy-r + 2r×cap_fraction]     │
│      sphere_cap_fraction=0.65이면 구 직경의 65% 영역        │
│    검출 실패 시 center crop으로 자동 fallback                │
│                                                             │
│  50k (center crop)                                          │
│    이미지 중앙 85% 직사각형 사용                             │
│                                                             │
│  출력: arr_inputRoiBgr (분석 대상 BGR 이미지)               │
│        dict_roi {x_min, y_min, x_max, y_max, width, height} │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2 : 전처리 + Otsu 이진화                              │
│  [utils/lsd.py — detect_acicular_lsd()]                     │
│                                                             │
│  CLAHE(clipLimit=2.0, tileGrid=8×8)                         │
│    → 대비 균일화 (밝기 편차 제거)                            │
│  GaussianBlur(3×3)                                           │
│    → 노이즈 제거                                             │
│  Otsu threshold                                             │
│    → float_otsu_thresh 계산 (전체 ROI 기준 단일 임계값)      │
│    → arr_otsu_binary (밝은 픽셀 = 침상 위치)                 │
│                                                             │
│  📷 lsd_01_preprocessed.png  (CLAHE+blur 결과)              │
│  📷 lsd_02_otsu_thresh.png   (Otsu 이진화 결과)              │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3 : LSD 원본 탐지                                     │
│                                                             │
│  cv2.createLineSegmentDetector(0).detect(arr_blur)          │
│    → arr_lines: N×1×4 (x1,y1,x2,y2)                        │
│    → arr_widths: LSD가 추정한 선분 폭 (필터용으로만 사용)    │
│  Image 7 기준: 약 2089개 탐지                                │
│                                                             │
│  📷 lsd_03_raw_detections.png  (노란색 선 = 전체 LSD 결과)  │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4 : 1차 필터 (길이 + 대략적 AR)                       │
│                                                             │
│  length < 20px → 제거 (너무 짧은 노이즈 선분)               │
│  lsd_width / length ≥ 0.60 → 제거 (뚱뚱한 선분, 침상 아님) │
│                                                             │
│  Image 7 기준: 2089 → 758개                                 │
│                                                             │
│  📷 lsd_04_after_filter.png  (하늘색 선 = 필터 후 선분)     │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5 : 중복 제거 (같은 침상에서 나온 복수 선분 통합)      │
│                                                             │
│  선분을 길이 내림차순 정렬 (긴 것 우선)                      │
│  이미 accepted된 선분과 비교:                                │
│    center 거리 < 12px AND 각도 차이 < 25° → 제거 (중복)     │
│    → 남은 것: 각 침상을 대표하는 하나의 선분                 │
│                                                             │
│  Image 7 기준: 758 → 662개 (→ 최종 217개)                   │
│  (662는 중복제거 후, 217은 두께 측정 통과 후)                │
│                                                             │
│  📷 lsd_05_after_dedup.png  (주황색 선 = 중복제거 후 선분)  │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6 : 수직 프로파일 두께 측정                           │
│  [utils/lsd.py — measure_perpendicular_thickness()]         │
│                                                             │
│  각 선분마다:                                               │
│    선분을 따라 7개 샘플 포인트 (t=0.2, 0.25, ..., 0.8)      │
│    각 포인트에서 수직 방향으로 ±0.5µm 스캔                   │
│      → intensity profile 추출                               │
│      → profile > float_otsu_thresh → 밝은 구간 탐지         │
│      → 스캔 중심에 가장 가까운 밝은 구간 선택 (이 침상)      │
│        (가장 가까운 구간을 선택하는 이유: 밀집 이미지에서     │
│         인접 침상 엣지로 스캔이 넘어가는 것을 방지)          │
│      → 구간 폭 = 이 포인트에서의 두께 (pixels)              │
│    7개 샘플의 중앙값 = 침상 두께                             │
│    두께 < 2px → 유효하지 않음 → 제거                        │
│                                                             │
│  aspect_ratio = thickness_px / length_px                    │
│    AR < 0.40 → 침상 (acicular)                              │
│    AR ≥ 0.40 → 판상/fragment                               │
│    particle_type=acicular이면 AR≥0.40 선분 제거             │
│                                                             │
│  ROI 엣지 근처 선분 (margin=8px) → 제거                     │
│                                                             │
│  Image 7 기준: 최종 217개 침상 탐지                         │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 7 : 결과 저장                                         │
│  [services/primary_particle.py — save_primary_outputs()]    │
│                                                             │
│  01_input.png              원본 이미지                      │
│  02_input_roi.png          분석 ROI (잘라낸 영역)           │
│  03_overlay_roi.png        색상 오버레이 (ROI 기준)          │
│                              침상=초록, 판상=파랑,           │
│                              fragment=주황                   │
│  04_overlay_full.png       전체 이미지에 ROI 표시            │
│  05_opencv_candidates.png  LSD 최종 탐지 선분               │
│                              초록=침상, 주황=판상            │
│  06_sphere_detection.png   구 검출 결과 (20k만 생성)         │
│  lsd_01_preprocessed.png   CLAHE+blur 전처리 결과           │
│  lsd_02_otsu_thresh.png    Otsu 이진화 (밝은 픽셀=침상)     │
│  lsd_03_raw_detections.png LSD 원본 전체 선분 (노란색)      │
│  lsd_04_after_filter.png   길이/AR 필터 후 (하늘색)         │
│  lsd_05_after_dedup.png    중복제거 후 (주황색)              │
│  objects.csv               전체 입자 측정값 (두께, 장축, AR) │
│  acicular.csv              침상 입자 분리 저장               │
│  thickness_dist.png        두께 분포 histogram               │
│  summary.json              통계 요약 + 분석 설정 기록        │
│  objects.json              입자별 상세 측정값                │
│  debug.json                LSD 탐지 디버그 정보              │
└─────────────────────────────────────────────────────────────┘
```

#### 단계별 이미지 색상 의미 요약

| 파일 | 색상 | 의미 |
|------|------|------|
| `lsd_01_preprocessed.png` | 그레이스케일 | CLAHE + Gaussian blur 적용 결과 |
| `lsd_02_otsu_thresh.png` | 흰색 = bright | Otsu threshold로 이진화된 밝은 픽셀 (침상 위치) |
| `lsd_03_raw_detections.png` | 노란색 선 | LSD가 탐지한 모든 선분 (필터 전) |
| `lsd_04_after_filter.png` | 하늘색 선 | 길이≥20px, AR<0.6 통과 선분 |
| `lsd_05_after_dedup.png` | 주황색 선 | 중복 제거 후 대표 선분 |
| `05_opencv_candidates.png` | 초록/주황 선 | 두께 측정 후 최종 침상(초록)/판상(주황) |
| `03_overlay_roi.png` | 초록/파랑/주황 면 | 최종 mask: 침상(초록), 판상(파랑), fragment(주황) |

### 판상 모드 (SAM2)

판상 모드는 SAM2 + point prompt 방식을 사용한다.  
`--model` 과 `--model_cfg` 가 필요하다.

---

## 사용 예시

### 침상 입자 / 20000배 (구형 전체 이미지)

```bash
python primary_measure.py \
  --input sem_20k.jpg \
  --particle_type acicular \
  --magnification 20k \
  --output_dir out_acicular_20k
```

> SAM2 불필요 — LSD 모드는 `--model` 없이 실행된다.

### 침상 입자 / 50000배 (표면 클로즈업)

```bash
python primary_measure.py \
  --input sem_50k.jpg \
  --particle_type acicular \
  --magnification 50k \
  --output_dir out_acicular_50k
```

### 판상 입자 / 50000배 (SAM2 필요)

```bash
python primary_measure.py \
  --input sem_plate_50k.jpg \
  --particle_type plate \
  --magnification 50k \
  --model model/sam2.1_hiera_base_plus.pt \
  --model_cfg model/sam2.1_hiera_t.yaml
```

### 스케일 바 수동 지정 (preset과 다를 경우)

```bash
# 실제 이미지: 21,000배, 스케일 바 206px = 5µm
python primary_measure.py \
  --input sem_21k.jpg \
  --particle_type acicular \
  --magnification 20k \
  --scale_pixels 206 \
  --scale_um 5 \
  --output_dir out_acicular_21k
```

### 배치 처리

```bash
python primary_measure.py \
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

실제 이미지의 스케일 바 픽셀 수를 측정한 뒤 `--scale_pixels`와 `--scale_um`으로 지정하면 정확한 µm 환산이 가능하다.

---

## 성능 비교

| 방식 | Image 6 (20k) | Image 7 (50k) | 처리 시간 |
|------|-------------|-------------|---------|
| SAM2 (이전) | 4개 | 10개 | 35–280초 |
| **LSD (현재)** | **78개** | **217개** | **< 2초** |

---

## 주요 모듈 참조

### services/primary_particle.py

| 함수/클래스 | 역할 |
|-------------|------|
| `PrimaryParticleService` | 전체 파이프라인 오케스트레이션 |
| `extract_inference_roi()` | ROI 추출 (구 검출 or center crop) |
| `process_primary()` | LSD/SAM2 분기 후 분석 실행 |
| `save_primary_outputs()` | 이미지·CSV·JSON·histogram 저장 |
| `run_primary_particle_analysis()` | 단일/배치 진입점 함수 |

### utils/lsd.py

| 함수 | 역할 |
|------|------|
| `detect_acicular_lsd()` | LSD 탐지 전체 파이프라인, step 이미지 반환 |
| `measure_perpendicular_thickness()` | 선분 수직 방향 intensity scan으로 두께 반환 |

### utils/image.py

| 함수 | 역할 |
|------|------|
| `detect_sphere_roi()` | 구형 2차 입자 검출 → cap ROI 좌표 반환 |
| `compute_center_roi()` | 중앙 crop 좌표 계산 |
| `enhance_image_texture()` | CLAHE + gradient + Laplacian texture 강화 |
| `sample_interest_points()` | Shi-Tomasi 코너 검출 (SAM2 point prompt용) |
| `create_processing_tiles()` | ROI를 겹치는 타일로 분할 |
