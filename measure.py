#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Primary Particle Thickness - SAM2 기반 1차 입자 침상/판상 분류 및 두께 측정
=============================================================================
2차전지 전구체 SEM 이미지(20000배 / 50000배)에서 1차 입자를 segmentation하고,
침상(acicular) / 판상(plate) 으로 분류하여 두께를 측정한다.

처리 흐름:
1. 원본 SEM 이미지 로드 → 자동 중앙 crop (또는 명시적 ROI)
2. SAM2 타일 추론으로 1차 입자 segmentation
3. cv2.minAreaRect 기반 두께(단축) / 장축 측정
4. aspect ratio 기준 침상 / 판상 / fragment 분류
5. 결과 저장 (objects.csv, acicular.csv, plate.csv, overlay, thickness histogram,
              summary.json, objects.json, debug.json, 개별 mask)
6. 배치 입력 시 IMG_ID 집계 및 batch_summary.json 생성
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import typing as tp
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from core import (
        Sam2AspectRatioConfig,
        Sam2AspectRatioService,
        convert_pixels_to_micrometers,
        calculate_mean_from_optional_values,
        calculate_percentage,
        collect_input_groups,
        build_image_output_dir,
        calculate_box_iou,
        calculate_binary_iou,
        CONST_SCALE_PIXELS,
        CONST_SCALE_MICROMETERS,
        CONST_SMALL_PARTICLE_SCALE_PIXELS,
        CONST_SMALL_PARTICLE_SCALE_MICROMETERS,
        CONST_DEFAULT_SMALL_PARTICLE,
        CONST_BBOX_EDGE_MARGIN,
        CONST_TILE_EDGE_MARGIN,
        CONST_ROI_X_MIN,
        CONST_ROI_Y_MIN,
        CONST_ROI_X_MAX,
        CONST_ROI_Y_MAX,
        CONST_MASK_BINARIZE_THRESHOLD,
        CONST_MIN_VALID_MASK_AREA,
        CONST_MASK_MORPH_KERNEL_SIZE,
        CONST_MASK_MORPH_OPEN_ITERATIONS,
        CONST_MASK_MORPH_CLOSE_ITERATIONS,
        CONST_DEFAULT_IMAGE_SIZE,
        CONST_DEFAULT_RETINA_MASKS,
        CONST_DEFAULT_SAVE_INDIVIDUAL_MASKS,
        CONST_DEFAULT_TILE_SIZE,
        CONST_DEFAULT_TILE_STRIDE,
        CONST_DEFAULT_POINTS_PER_TILE,
        CONST_DEFAULT_POINT_MIN_DISTANCE,
        CONST_DEFAULT_POINT_QUALITY_LEVEL,
        CONST_DEFAULT_POINT_BATCH_SIZE,
        CONST_DEFAULT_DEDUP_IOU,
        CONST_DEFAULT_BBOX_DEDUP_IOU,
        CONST_DEFAULT_USE_POINT_PROMPTS,
    )
except ImportError as e:
    print(
        f"[ERROR] core.py 를 import할 수 없습니다: {e}\n"
        "같은 디렉터리에 있는지 확인하세요.",
        file=sys.stderr,
    )
    sys.exit(1)


# =========================================================
# 1차 입자 분석 전용 상수
# =========================================================

# 침상(acicular) / 판상(plate) 분류 기준
# aspect_ratio = thickness_px / long_axis_px  (0 < x <= 1)
# aspect_ratio <  이 값 : 침상
# aspect_ratio >= 이 값 : 판상
CONST_ACICULAR_THRESHOLD: float = 0.40


def _json_default(obj: tp.Any) -> tp.Any:
    """numpy scalar/array → Python native type (JSON 직렬화 호환)."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# 유효 1차 입자 최소 면적 (미만이면 fragment)
# 2차 입자 기본값(1500)보다 훨씬 작게 설정
CONST_PRIMARY_PARTICLE_AREA_THRESHOLD: float = 200.0

# 자동 중앙 crop 비율 (이미지 중앙의 이 비율 영역을 사용)
CONST_CENTER_CROP_RATIO: float = 0.60

# 침상+판상 입자 목표 수 (미달 시 경고)
CONST_TARGET_PARTICLE_COUNT: int = 10

# ---- 구형(sphere) 2차 입자 자동 검출 ----
# acicular 모드에서 구형 2차 입자를 검출해 cap 영역을 ROI로 사용
CONST_SPHERE_CAP_FRACTION: float = 0.45   # 구 지름 기준 cap 높이 비율
CONST_SPHERE_MORPH_KERNEL: int = 15       # 구 마스크 전처리 morphology kernel 크기
CONST_SPHERE_MIN_RADIUS_RATIO: float = 0.15  # 이미지 단변 기준 최소 구 반지름 비율

# 1차 입자용 SAM2 추론 파라미터 (더 촘촘한 타일/포인트)
CONST_PRIMARY_TILE_SIZE: int = 256
CONST_PRIMARY_TILE_STRIDE: int = 128
CONST_PRIMARY_POINTS_PER_TILE: int = 120
CONST_PRIMARY_POINT_MIN_DISTANCE: int = 8

# ---- 침상 hybrid mode (OpenCV → SAM2 box prompt) ----
# OpenCV adaptive threshold 파라미터
CONST_ACICULAR_ADAPT_BLOCK_SIZE: int = 35   # 홀수여야 함; 대략 입자 크기의 2~3배 픽셀
CONST_ACICULAR_ADAPT_C: int = 4             # 밝기 보정값 (양수 → bright-on-dark 기준)

# 후보 contour 필터링
CONST_ACICULAR_CANDIDATE_MIN_AREA: float = 60.0     # px² 미만 제거 (노이즈)
CONST_ACICULAR_CANDIDATE_MAX_AREA: float = 25000.0  # px² 초과 제거 (2차 입자 등)
CONST_ACICULAR_CANDIDATE_AR_SCREEN: float = 0.60    # minAreaRect AR < 이 값만 후보로 선택

# SAM2 box prompt 파라미터
CONST_ACICULAR_BBOX_PAD_RATIO: float = 0.08   # bbox를 이 비율만큼 확장 후 SAM2에 전달
CONST_ACICULAR_BOX_PROMPT_BATCH: int = 16     # 한 번의 SAM2 호출에 묶을 bbox 수
CONST_ACICULAR_FALLBACK_THRESHOLD: int = 3    # OpenCV 후보가 이 수 미만이면 point fallback

# ---- LSD (Line Segment Detector) 직접 측정 모드 ----
CONST_LSD_MIN_LENGTH_PX: int = 20        # 최소 segment 길이 (px)
CONST_LSD_DEDUP_DIST_PX: int = 12        # 이 거리 이내 + 비슷한 각도 → 중복으로 제거
CONST_LSD_DEDUP_ANGLE_DEG: float = 25.0  # 중복 판정 각도 차이 기준 (도)
CONST_LSD_PERP_THRESH_RATIO: float = 0.35  # 수직 프로파일에서 엣지 감지 임계값 비율
CONST_LSD_PERP_N_SAMPLES: int = 7        # segment를 따라 수직 프로파일 샘플링 횟수


# =========================================================
# 배율 × 입경 Preset (--particle_type / --magnification 조합)
# =========================================================
#
# 침상(acicular) = 대입경(large)  → 20000배 or 50000배
# 판상(plate)    = 소입경(small)  → 20000배 or 50000배
#
# 20000배: 구형 2차 입자 전체가 화면에 들어옴  → sphere auto-detect 가능
# 50000배: 입자 표면 클로즈업 (위아래 잘림)    → center crop 사용
#
# 아래 키는 argparse dest 이름과 일치 (set_defaults 에 직접 전달됨)

DICT_PRESETS: tp.Dict[tp.Tuple[str, str], tp.Dict[str, tp.Any]] = {
    # ── 침상 / 20000배 ────────────────────────────────────────
    ("acicular", "20k"): {
        "scale_pixels":        276.0,
        "scale_um":            50.0,
        "particle_mode":       "acicular",
        "measure_mode":        "lsd",   # LSD 직접 측정 (SAM2 불필요)
        "auto_detect_sphere":  True,
        "sphere_cap_fraction": 0.65,
        "auto_center_crop":    True,
        "center_crop_ratio":   0.60,
        "tile_size":           192,
        "stride":              96,
        "points_per_tile":     120,
        "point_min_distance":  8,
        "area_threshold":      80.0,
    },
    # ── 침상 / 50000배 ────────────────────────────────────────
    ("acicular", "50k"): {
        "scale_pixels":        184.0,
        "scale_um":            10.0,
        "particle_mode":       "acicular",
        "measure_mode":        "lsd",   # LSD 직접 측정
        "auto_detect_sphere":  False,
        "auto_center_crop":    True,
        "center_crop_ratio":   0.85,
        "tile_size":           192,
        "stride":              96,
        "points_per_tile":     150,
        "point_min_distance":  5,
        "area_threshold":      20.0,
    },
    # ── 판상 / 20000배 ────────────────────────────────────────
    ("plate", "20k"): {
        "scale_pixels":        276.0,
        "scale_um":            50.0,
        "particle_mode":       "auto",  # 판상은 elongation 낮음 → acicular hybrid 불필요
        "auto_detect_sphere":  True,
        "sphere_cap_fraction": 0.65,
        "auto_center_crop":    True,
        "center_crop_ratio":   0.60,
        "tile_size":           192,
        "stride":              96,
        "points_per_tile":     100,
        "point_min_distance":  10,
        "area_threshold":      300.0,
    },
    # ── 판상 / 50000배 ────────────────────────────────────────
    ("plate", "50k"): {
        "scale_pixels":        184.0,
        "scale_um":            10.0,
        "particle_mode":       "auto",
        "auto_detect_sphere":  False,
        "auto_center_crop":    True,
        "center_crop_ratio":   0.85,
        "tile_size":           192,
        "stride":              96,
        "points_per_tile":     100,
        "point_min_distance":  8,
        "area_threshold":      150.0,
    },
}


def get_analysis_preset(
    str_particleType: str,
    str_magnification: str,
) -> tp.Dict[str, tp.Any]:
    """배율 + 입자 형태 조합에 맞는 파라미터 preset을 반환한다.

    Returns:
        argparse dest 이름을 키로 하는 파라미터 dict. 매칭 preset이 없으면 빈 dict.
    """
    return dict(DICT_PRESETS.get((str_particleType, str_magnification), {}))


# =========================================================
# Config / Dataclass
# =========================================================


@dataclass
class PrimaryParticleConfig(Sam2AspectRatioConfig):
    """1차 입자 분석 전용 설정. Sam2AspectRatioConfig 를 확장한다."""

    float_acicularThreshold: float = CONST_ACICULAR_THRESHOLD
    bool_autoCenterCrop: bool = True
    float_centerCropRatio: float = CONST_CENTER_CROP_RATIO
    int_targetParticleCount: int = CONST_TARGET_PARTICLE_COUNT
    # "auto": 기존 tiled point prompt / "acicular": OpenCV→SAM2 box prompt hybrid
    str_particleMode: str = "auto"
    # 구형 2차 입자 자동 검출 → cap 영역 ROI 사용 (acicular 모드 전용)
    bool_autoDetectSphere: bool = False
    float_sphereCapFraction: float = CONST_SPHERE_CAP_FRACTION
    # 분석 컨텍스트 (summary.json 기록용)
    str_particleType: str = "unknown"   # "acicular" | "plate" | "unknown"
    str_magnification: str = "unknown"  # "20k" | "50k" | "unknown"
    # 측정 방법 선택
    str_measureMode: str = "sam2"       # "sam2": SAM2 기반 | "lsd": LSD 직접 측정 (빠름, 대량)


@dataclass
class PrimaryParticleMeasurement:
    """1차 입자 단일 마스크 측정 결과."""

    int_index: int
    str_category: str            # "acicular" | "plate" | "fragment"
    int_maskArea: int
    float_confidence: tp.Optional[float]
    int_bboxX: int
    int_bboxY: int
    int_bboxWidth: int
    int_bboxHeight: int
    float_centroidX: float
    float_centroidY: float
    # minAreaRect 기반 단/장축 측정 (회전 보정)
    float_thicknessPx: float     # 단축 = 두께 [pixel]
    float_longAxisPx: float      # 장축 [pixel]
    float_minRectAngle: float    # 회전 각도 [degree]
    float_thicknessUm: float     # 두께 [µm]
    float_longAxisUm: float      # 장축 [µm]
    float_aspectRatio: float     # thickness / long_axis  (0 < x <= 1)
    # H/V span (참조용 — 기존 secondary 파이프라인 호환)
    int_longestHorizontal: int
    int_longestVertical: int
    float_longestHorizontalUm: float
    float_longestVerticalUm: float


@dataclass
class PrimaryParticleResult:
    """단일 이미지 처리 결과."""

    list_objects: tp.List[PrimaryParticleMeasurement]
    dict_summary: tp.Dict[str, tp.Any]


# =========================================================
# Service
# =========================================================


class PrimaryParticleService(Sam2AspectRatioService):
    """1차 입자 segmentation, 측정, 저장 서비스."""

    def __init__(self, obj_config: PrimaryParticleConfig) -> None:
        super().__init__(obj_config)
        self.obj_primary_config: PrimaryParticleConfig = obj_config

    def validate_inputs(self) -> None:
        """LSD 모드에서는 모델 파일 불필요 → 이미지와 config만 확인한다."""
        if self.obj_primary_config.str_measureMode == "lsd":
            if not self.obj_config.path_input.exists():
                raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {self.obj_config.path_input}")
            return
        super().validate_inputs()

    # ----------------------------------------------------------
    # ROI: 구형 2차 입자 검출 + cap 영역 추출
    # ----------------------------------------------------------

    def detect_sphere_and_extract_cap(
        self,
        arr_imageBgr: np.ndarray,
    ) -> tp.Tuple[tp.Optional[tp.Tuple[int, int, int, int]], tp.Optional[np.ndarray]]:
        """이미지에서 구형 2차 입자를 검출하고 top-cap ROI 좌표를 반환한다.

        Cap 영역은 구 상단 `float_sphereCapFraction` 비율에 해당하는 직사각형이다.
        이 영역의 1차 침상 입자는 전자빔에 대해 측면 방향으로 배열되어 두께 측정에 유리하다.

        Args:
            arr_imageBgr: 원본 BGR 이미지.

        Returns:
            `((x1, y1, x2, y2), arr_debugSphere)` 또는 검출 실패 시 `(None, None)`.
            `arr_debugSphere`는 구 마스크 (저장·시각화용 uint8).
        """
        arr_gray = cv2.cvtColor(arr_imageBgr, cv2.COLOR_BGR2GRAY)
        int_h, int_w = arr_gray.shape[:2]

        # 큰 blur → 구 경계 강조, 입자 텍스처 제거
        arr_blur = cv2.GaussianBlur(arr_gray, (21, 21), 0)

        # Otsu threshold로 구 vs 배경 분리
        _, arr_thresh = cv2.threshold(arr_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # foreground가 50% 초과이면 극성 반전 (배경이 밝은 이미지 대응)
        if float(arr_thresh.sum()) / (255.0 * int_h * int_w) > 0.5:
            arr_thresh = cv2.bitwise_not(arr_thresh)

        # 큰 kernel morphology close → 구 내부 빈 틈 제거
        int_k = CONST_SPHERE_MORPH_KERNEL
        arr_kernel = np.ones((int_k, int_k), np.uint8)
        arr_closed = cv2.morphologyEx(arr_thresh, cv2.MORPH_CLOSE, arr_kernel, iterations=3)
        arr_opened = cv2.morphologyEx(arr_closed, cv2.MORPH_OPEN, arr_kernel, iterations=2)

        # 가장 큰 contour = 구
        list_cnts, _ = cv2.findContours(
            arr_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not list_cnts:
            return None, None

        arr_sphereCnt = max(list_cnts, key=cv2.contourArea)
        float_sphereArea = float(cv2.contourArea(arr_sphereCnt))

        # 너무 작으면 (이미지의 2% 미만) 구로 간주하지 않음
        if float_sphereArea < int_h * int_w * 0.02:
            return None, None

        # 최소 외접원 → 중심 + 반지름
        (float_cx, float_cy), float_r = cv2.minEnclosingCircle(arr_sphereCnt)
        int_cx = int(float_cx)
        int_cy = int(float_cy)
        int_r = int(float_r)

        # 너무 작은 반지름이면 reject
        if int_r < int(min(int_h, int_w) * CONST_SPHERE_MIN_RADIUS_RATIO):
            return None, None

        # Cap 영역 (구 상단 cap_fraction 비율)
        float_cap = float(np.clip(self.obj_primary_config.float_sphereCapFraction, 0.1, 1.0))
        int_y1 = max(0, int_cy - int_r)
        int_y2 = min(int_h, int_y1 + int(int_r * 2 * float_cap))
        int_x1 = max(0, int_cx - int_r)
        int_x2 = min(int_w, int_cx + int_r)

        if int_x2 <= int_x1 or int_y2 <= int_y1:
            return None, None

        print(
            f"[sphere-detect] 구 검출: center=({int_cx},{int_cy}) r={int_r}px  "
            f"cap ROI=({int_x1},{int_y1})-({int_x2},{int_y2})",
            flush=True,
        )

        # debug mask (구 외접원 영역 시각화용)
        arr_debugSphere = np.zeros((int_h, int_w), dtype=np.uint8)
        cv2.circle(arr_debugSphere, (int_cx, int_cy), int_r, 255, 2)
        cv2.rectangle(arr_debugSphere, (int_x1, int_y1), (int_x2, int_y2), 128, 2)

        return (int_x1, int_y1, int_x2, int_y2), arr_debugSphere

    # ----------------------------------------------------------
    # ROI: 자동 중앙 crop
    # ----------------------------------------------------------

    def compute_center_roi(
        self, int_h: int, int_w: int
    ) -> tp.Tuple[int, int, int, int]:
        """이미지 크기에서 중앙 crop ROI 좌표를 계산한다.

        Returns:
            (x0, y0, x1, y1) 형식의 ROI 좌표.
        """
        float_ratio = float(np.clip(self.obj_primary_config.float_centerCropRatio, 0.1, 1.0))
        int_xMargin = int(int_w * (1.0 - float_ratio) / 2.0)
        int_yMargin = int(int_h * (1.0 - float_ratio) / 2.0)
        int_x0 = max(0, int_xMargin)
        int_y0 = max(0, int_yMargin)
        int_x1 = min(int_w, int_w - int_xMargin)
        int_y1 = min(int_h, int_h - int_yMargin)
        return int_x0, int_y0, int_x1, int_y1

    def extract_inference_roi(
        self,
        arr_imageBgr: np.ndarray,
    ) -> tp.Tuple[np.ndarray, tp.Dict[str, int]]:
        """ROI를 추출한다.

        우선순위:
        1. `bool_autoDetectSphere=True` (acicular 모드) → 구 검출 후 cap ROI
        2. `bool_autoCenterCrop=True` → 중앙 crop ROI
        3. 명시적 ROI 좌표 사용
        """
        int_h, int_w = arr_imageBgr.shape[:2]

        # 구 자동 검출 (acicular 모드에서만)
        if (
            self.obj_primary_config.str_particleMode == "acicular"
            and self.obj_primary_config.bool_autoDetectSphere
        ):
            tpl_cap, arr_sphereDbg = self.detect_sphere_and_extract_cap(arr_imageBgr)
            if tpl_cap is not None:
                int_x0, int_y0, int_x1, int_y1 = tpl_cap
                arr_roiBgr = arr_imageBgr[int_y0:int_y1, int_x0:int_x1].copy()
                dict_roi: tp.Dict[str, int] = {
                    "x_min": int_x0,
                    "y_min": int_y0,
                    "x_max": int_x1,
                    "y_max": int_y1,
                    "width": int_x1 - int_x0,
                    "height": int_y1 - int_y0,
                }
                # sphere debug 이미지를 인스턴스 변수로 보관 → save 단계에서 저장
                self._arr_sphereDebug: tp.Optional[np.ndarray] = arr_sphereDbg
                return arr_roiBgr, dict_roi
            else:
                print(
                    "[sphere-detect] 구 검출 실패 → center crop fallback",
                    flush=True,
                )

        self._arr_sphereDebug = None

        if self.obj_primary_config.bool_autoCenterCrop:
            int_x0, int_y0, int_x1, int_y1 = self.compute_center_roi(int_h, int_w)
        else:
            int_x0 = max(0, min(self.obj_config.int_roiXMin, int_w))
            int_y0 = max(0, min(self.obj_config.int_roiYMin, int_h))
            int_x1 = max(int_x0, min(self.obj_config.int_roiXMax, int_w))
            int_y1 = max(int_y0, min(self.obj_config.int_roiYMax, int_h))

        if int_x1 <= int_x0 or int_y1 <= int_y0:
            raise ValueError("유효한 ROI를 만들 수 없습니다. ROI 좌표와 이미지 크기를 확인하세요.")

        arr_roiBgr = arr_imageBgr[int_y0:int_y1, int_x0:int_x1].copy()
        dict_roi = {
            "x_min": int_x0,
            "y_min": int_y0,
            "x_max": int_x1,
            "y_max": int_y1,
            "width": int_x1 - int_x0,
            "height": int_y1 - int_y0,
        }
        return arr_roiBgr, dict_roi

    # ----------------------------------------------------------
    # 측정: minAreaRect 기반 두께 + 침상/판상 분류
    # ----------------------------------------------------------

    def measure_primary_mask(
        self,
        arr_mask: np.ndarray,
        int_index: int,
        float_confidence: tp.Optional[float],
    ) -> tp.Optional[PrimaryParticleMeasurement]:
        """단일 mask를 minAreaRect로 측정하고 침상/판상/fragment로 분류한다.

        Args:
            arr_mask: ROI 좌표계 기준 binary mask.
            int_index: 마스크 인덱스 (식별용).
            float_confidence: SAM2 confidence score. 없으면 None.

        Returns:
            유효하면 PrimaryParticleMeasurement, 아니면 None.
        """
        arr_refined = self.refine_mask_for_area(arr_mask)
        int_maskArea = int(arr_refined.sum())
        if int_maskArea < self.obj_config.int_minValidMaskArea:
            return None

        arr_contour = self.extract_largest_contour(arr_refined)
        if arr_contour is None:
            return None

        int_bx, int_by, int_bw, int_bh = cv2.boundingRect(arr_contour)
        int_roiH, int_roiW = arr_refined.shape[:2]
        if self.is_bbox_near_roi_edge(int_bx, int_by, int_bw, int_bh, int_roiW, int_roiH):
            return None

        # centroid
        obj_moments = cv2.moments(arr_contour)
        if obj_moments["m00"] > 0.0:
            float_cx = float(obj_moments["m10"] / obj_moments["m00"])
            float_cy = float(obj_moments["m01"] / obj_moments["m00"])
        else:
            float_cx = float(int_bx + int_bw / 2.0)
            float_cy = float(int_by + int_bh / 2.0)

        # minAreaRect: 회전 보정된 단/장축 측정
        # contour 점이 5개 이상일 때만 사용 가능
        if len(arr_contour) >= 5:
            tpl_rect = cv2.minAreaRect(arr_contour)
            (_, _), (float_rectW, float_rectH), float_rectAngle = tpl_rect
            float_thicknessPx = float(min(float_rectW, float_rectH))
            float_longAxisPx = float(max(float_rectW, float_rectH))
        else:
            float_thicknessPx = float(min(int_bw, int_bh))
            float_longAxisPx = float(max(int_bw, int_bh))
            float_rectAngle = 0.0

        # 장축이 0에 가까우면 skip
        if float_longAxisPx < 1.0:
            return None

        float_aspectRatio = float_thicknessPx / float_longAxisPx

        # H/V span (참조용)
        int_horizontal = min(
            self.get_longest_span(arr_refined, bool_horizontal=True), int_bw)
        int_vertical = min(
            self.get_longest_span(arr_refined, bool_horizontal=False), int_bh)

        # 분류
        # particle_type 에 따라 목표 형태만 해당 카테고리로 분류하고
        # 나머지(면적 미달 포함)는 fragment 처리 — 분석 대상이 아닌 마스크
        if int_maskArea < int(round(self.obj_config.float_particleAreaThreshold)):
            str_category = "fragment"
        elif float_aspectRatio < self.obj_primary_config.float_acicularThreshold:
            # 침상 형태 (AR < threshold)
            str_category = (
                "acicular"
                if self.obj_primary_config.str_particleType != "plate"
                else "fragment"  # 판상 모드에서 침상은 분석 대상 아님
            )
        else:
            # 판상 형태 (AR >= threshold)
            str_category = (
                "plate"
                if self.obj_primary_config.str_particleType != "acicular"
                else "fragment"  # 침상 모드에서 판상은 분석 대상 아님
            )

        return PrimaryParticleMeasurement(
            int_index=int_index,
            str_category=str_category,
            int_maskArea=int_maskArea,
            float_confidence=float_confidence,
            int_bboxX=int_bx,
            int_bboxY=int_by,
            int_bboxWidth=int_bw,
            int_bboxHeight=int_bh,
            float_centroidX=float_cx,
            float_centroidY=float_cy,
            float_thicknessPx=float_thicknessPx,
            float_longAxisPx=float_longAxisPx,
            float_minRectAngle=float(float_rectAngle),
            float_thicknessUm=self.convert_pixels_to_micrometers(float_thicknessPx),
            float_longAxisUm=self.convert_pixels_to_micrometers(float_longAxisPx),
            float_aspectRatio=float_aspectRatio,
            int_longestHorizontal=int_horizontal,
            int_longestVertical=int_vertical,
            float_longestHorizontalUm=self.convert_pixels_to_micrometers(float(int_horizontal)),
            float_longestVerticalUm=self.convert_pixels_to_micrometers(float(int_vertical)),
        )

    # ----------------------------------------------------------
    # 침상 hybrid: OpenCV 후보 탐지
    # ----------------------------------------------------------

    def detect_acicular_candidates_opencv(
        self,
        arr_roiGray: np.ndarray,
        float_arScreen: float = CONST_ACICULAR_CANDIDATE_AR_SCREEN,
        float_minArea: float = CONST_ACICULAR_CANDIDATE_MIN_AREA,
        float_maxArea: float = CONST_ACICULAR_CANDIDATE_MAX_AREA,
        int_maxCandidates: int = 500,
    ) -> tp.Tuple[tp.List[tp.Tuple[int, int, int, int]], np.ndarray]:
        """connectedComponents + moments 기반으로 침상 후보 bbox를 탐지한다.

        Adaptive threshold 후 erosion으로 인접 입자를 분리하고,
        connectedComponentsWithStats로 개별 blob을 추출한 뒤,
        image moments의 공분산 고유값으로 elongation을 계산한다
        (minAreaRect/fitEllipse보다 픽셀 분포를 직접 반영해 정확).

        Returns:
            (list_bboxes, arr_debugMask):
            - list_bboxes: 패딩이 적용된 (x1, y1, x2, y2) 후보 bbox 목록.
            - arr_debugMask: erosion 후 이진 이미지 (시각화·저장용).
        """
        int_roiH, int_roiW = arr_roiGray.shape[:2]

        # 1. 대비 정규화
        obj_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        arr_eq = obj_clahe.apply(arr_roiGray)
        arr_blur = cv2.GaussianBlur(arr_eq, (3, 3), 0)

        # 2. Adaptive threshold — ROI 크기에 맞게 block size 자동 조정
        int_bsAuto = max(11, min(CONST_ACICULAR_ADAPT_BLOCK_SIZE,
                                  int(min(int_roiH, int_roiW) / 25)))
        int_bs = int_bsAuto if int_bsAuto % 2 == 1 else int_bsAuto + 1
        arr_thresh = cv2.adaptiveThreshold(
            arr_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int_bs,
            CONST_ACICULAR_ADAPT_C,
        )
        if float(arr_thresh.sum()) / (255.0 * int_roiH * int_roiW) > 0.55:
            arr_thresh = cv2.bitwise_not(arr_thresh)

        # 3. Erosion — 서로 붙어있는 입자 경계를 끊어 개별 blob으로 분리
        arr_kernelE = np.ones((2, 2), np.uint8)
        arr_eroded = cv2.erode(arr_thresh, arr_kernelE, iterations=2)

        # 4. connectedComponentsWithStats → 개별 blob 추출
        int_numLabels, arr_labels, arr_stats, arr_centroids = cv2.connectedComponentsWithStats(
            arr_eroded, connectivity=8)

        # 5. 각 blob에 대해 면적 필터 + elongation 계산 (image moments 기반)
        list_bboxes: tp.List[tp.Tuple[int, int, int, int]] = []
        arr_debugMask = np.zeros_like(arr_eroded)

        for int_lbl in range(1, int_numLabels):  # 0 = 배경
            int_area = int(arr_stats[int_lbl, cv2.CC_STAT_AREA])
            if int_area < float_minArea or int_area > float_maxArea:
                continue

            int_bx = int(arr_stats[int_lbl, cv2.CC_STAT_LEFT])
            int_by = int(arr_stats[int_lbl, cv2.CC_STAT_TOP])
            int_bw = int(arr_stats[int_lbl, cv2.CC_STAT_WIDTH])
            int_bh = int(arr_stats[int_lbl, cv2.CC_STAT_HEIGHT])

            # elongation: image moments의 공분산 행렬 고유값 비율
            # lambda_min / lambda_max = (단축/장축)² ≈ AR²
            arr_compMask = (arr_labels == int_lbl).astype(np.uint8)
            dict_m = cv2.moments(arr_compMask)
            float_m00 = dict_m["m00"]
            if float_m00 < 1.0:
                continue

            float_mu20 = dict_m["mu20"] / float_m00
            float_mu02 = dict_m["mu02"] / float_m00
            float_mu11 = dict_m["mu11"] / float_m00
            float_disc = float(np.sqrt(max(0.0,
                (float_mu20 - float_mu02) ** 2 + 4.0 * float_mu11 ** 2)))
            float_l1 = (float_mu20 + float_mu02 + float_disc) / 2.0  # 장축 분산
            float_l2 = (float_mu20 + float_mu02 - float_disc) / 2.0  # 단축 분산

            # AR = sqrt(l2/l1): 1에 가까울수록 원형, 0에 가까울수록 길쭉
            float_ar = float(np.sqrt(max(0.0, float_l2) / max(1e-9, float_l1)))
            if float_ar >= float_arScreen:
                continue  # 충분히 길쭉하지 않음

            if self.is_bbox_near_roi_edge(int_bx, int_by, int_bw, int_bh, int_roiW, int_roiH):
                continue

            # bbox 패딩
            int_padX = max(2, int(int_bw * CONST_ACICULAR_BBOX_PAD_RATIO))
            int_padY = max(2, int(int_bh * CONST_ACICULAR_BBOX_PAD_RATIO))
            int_x1 = max(0, int_bx - int_padX)
            int_y1 = max(0, int_by - int_padY)
            int_x2 = min(int_roiW, int_bx + int_bw + int_padX)
            int_y2 = min(int_roiH, int_by + int_bh + int_padY)
            list_bboxes.append((int_x1, int_y1, int_x2, int_y2))
            arr_debugMask[arr_labels == int_lbl] = 255

            if len(list_bboxes) >= int_maxCandidates:
                break

        return list_bboxes, arr_debugMask

    # ----------------------------------------------------------
    # 침상 hybrid: SAM2 box prompt 추론
    # ----------------------------------------------------------

    def predict_with_box_prompts(
        self,
        arr_roi: np.ndarray,
        list_boxes: tp.List[tp.Tuple[int, int, int, int]],
    ) -> tp.Tuple[np.ndarray, tp.Optional[np.ndarray], tp.Dict[str, tp.Any]]:
        """SAM2 bounding box prompt 방식으로 ROI 전체에 대해 mask를 추론한다.

        Args:
            arr_roi: ROI BGR 이미지.
            list_boxes: (x1, y1, x2, y2) 형식의 bbox 목록.

        Returns:
            (arr_masks, arr_scores, dict_debug) — predict_tiled_point_prompts와 동일 형식.
        """
        if self.obj_model is None:
            self.initialize_model()

        int_roiH, int_roiW = arr_roi.shape[:2]

        dict_predictCommon: tp.Dict[str, tp.Any] = {
            "imgsz": self.obj_config.int_imgSize,
            "retina_masks": self.obj_config.bool_retinaMasks,
            "verbose": False,
        }
        if self.obj_config.str_device:
            dict_predictCommon["device"] = self.obj_config.str_device

        list_keptMasks: tp.List[np.ndarray] = []
        list_keptScores: tp.List[tp.Optional[float]] = []
        list_keptBboxes: tp.List[tp.Tuple[int, int, int, int]] = []
        int_acceptedCount = 0
        int_bboxDedupRejected = 0

        int_bs = max(1, CONST_ACICULAR_BOX_PROMPT_BATCH)
        for int_start in range(0, len(list_boxes), int_bs):
            list_batch = list_boxes[int_start:int_start + int_bs]
            list_bboxForSam = [[x1, y1, x2, y2] for (x1, y1, x2, y2) in list_batch]

            try:
                list_results = self.obj_model(  # type: ignore[misc]
                    source=arr_roi,
                    bboxes=list_bboxForSam,
                    **dict_predictCommon,
                )
            except Exception as exc:
                print(f"[WARN] box prompt batch 실패 (skip): {exc}", flush=True)
                continue

            if not list_results:
                continue

            obj_result = list_results[0]
            if obj_result.masks is None or obj_result.masks.data is None:
                continue

            arr_batchMasks = obj_result.masks.data.detach().cpu().numpy()
            arr_batchScores = None
            if obj_result.boxes is not None and obj_result.boxes.conf is not None:
                arr_batchScores = obj_result.boxes.conf.detach().cpu().numpy()

            for int_mi, arr_rawMask in enumerate(arr_batchMasks):
                arr_mask = (
                    arr_rawMask > self.obj_config.float_maskBinarizeThreshold
                ).astype(np.uint8)
                if int(arr_mask.sum()) < self.obj_config.int_minValidMaskArea:
                    continue

                arr_contour = self.extract_largest_contour(arr_mask)
                if arr_contour is None:
                    continue

                int_bx, int_by, int_bw, int_bh = cv2.boundingRect(arr_contour)
                if self.is_bbox_near_roi_edge(
                        int_bx, int_by, int_bw, int_bh, int_roiW, int_roiH):
                    continue

                tuple_box = (int_bx, int_by, int_bw, int_bh)
                bool_bboxDup = any(
                    calculate_box_iou(tuple_box, prev) >= self.obj_config.float_bboxDedupIou
                    for prev in list_keptBboxes
                )
                if bool_bboxDup:
                    int_bboxDedupRejected += 1
                    continue

                bool_maskDup = any(
                    calculate_binary_iou(arr_mask, prev) >= self.obj_config.float_dedupIou
                    for prev in list_keptMasks
                )
                if bool_maskDup:
                    continue

                int_acceptedCount += 1
                list_keptMasks.append(arr_mask)
                list_keptBboxes.append(tuple_box)
                float_score: tp.Optional[float] = None
                if arr_batchScores is not None and int_mi < len(arr_batchScores):
                    float_score = float(arr_batchScores[int_mi])
                list_keptScores.append(float_score)

        arr_masks = (
            np.stack(list_keptMasks, axis=0).astype(np.uint8)
            if list_keptMasks
            else np.empty((0, int_roiH, int_roiW), dtype=np.uint8)
        )
        arr_scores: tp.Optional[np.ndarray] = None
        if list_keptScores:
            arr_scores = np.array(
                [np.nan if x is None else float(x) for x in list_keptScores],
                dtype=np.float32,
            )
        dict_debug: tp.Dict[str, tp.Any] = {
            "num_tiles": 1,
            "num_candidate_boxes": len(list_boxes),
            "num_candidate_points": 0,
            "num_accepted_masks": int_acceptedCount,
            "num_bbox_dedup_rejected": int_bboxDedupRejected,
        }
        return arr_masks, arr_scores, dict_debug

    def _merge_mask_results(
        self,
        arr_masksA: np.ndarray,
        arr_scoresA: tp.Optional[np.ndarray],
        arr_masksB: np.ndarray,
        arr_scoresB: tp.Optional[np.ndarray],
        int_roiH: int,
        int_roiW: int,
    ) -> tp.Tuple[np.ndarray, tp.Optional[np.ndarray]]:
        """두 mask 배열을 IoU dedup하며 병합한다 (A 우선)."""
        list_masks: tp.List[np.ndarray] = list(arr_masksA) if arr_masksA.shape[0] > 0 else []
        list_scores: tp.List[tp.Optional[float]] = (
            [float(x) if not np.isnan(x) else None for x in arr_scoresA]
            if arr_scoresA is not None else [None] * len(list_masks)
        )

        for int_i, arr_m in enumerate(arr_masksB):
            if any(
                calculate_binary_iou(arr_m, prev) >= self.obj_config.float_dedupIou
                for prev in list_masks
            ):
                continue
            list_masks.append(arr_m)
            float_sb: tp.Optional[float] = None
            if arr_scoresB is not None and int_i < len(arr_scoresB):
                float_sb = float(arr_scoresB[int_i]) if not np.isnan(arr_scoresB[int_i]) else None
            list_scores.append(float_sb)

        arr_merged = (
            np.stack(list_masks, axis=0).astype(np.uint8)
            if list_masks
            else np.empty((0, int_roiH, int_roiW), dtype=np.uint8)
        )
        arr_mergedScores: tp.Optional[np.ndarray] = None
        if list_scores:
            arr_mergedScores = np.array(
                [np.nan if x is None else float(x) for x in list_scores],
                dtype=np.float32,
            )
        return arr_merged, arr_mergedScores

    # ----------------------------------------------------------
    # 침상 hybrid: 메인 파이프라인
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # LSD 직접 측정 모드
    # ----------------------------------------------------------

    def _measure_perpendicular_thickness(
        self,
        arr_gray: np.ndarray,
        arr_global_thresh: float,
        float_x1: float,
        float_y1: float,
        float_x2: float,
        float_y2: float,
    ) -> float:
        """침상 선분 방향에 수직으로 강도 프로파일을 샘플링해 실제 폭(두께)을 반환한다.

        빽빽한 이미지에서 여러 침상이 스캔 범위에 들어올 수 있으므로,
        스캔 **중심에 가장 가까운** 연속 밝은 영역만 선택한다.
        (중심 = LSD 선분 위치 = 이 침상의 엣지)

        Args:
            arr_gray: ROI grayscale 이미지.
            arr_global_thresh: 전체 ROI Otsu threshold.
            float_x1, float_y1, float_x2, float_y2: 선분 끝점.

        Returns:
            두께 (pixel). 유효 샘플이 없으면 0.0.
        """
        float_dx = float_x2 - float_x1
        float_dy = float_y2 - float_y1
        float_length = float(np.sqrt(float_dx ** 2 + float_dy ** 2))
        if float_length < 1.0:
            return 0.0

        float_ux = float_dx / float_length
        float_uy = float_dy / float_length
        float_px = -float_uy   # 수직 방향
        float_py = float_ux

        int_roiH, int_roiW = arr_gray.shape[:2]
        # 수직 스캔 범위: 예상 침상 폭의 2~3배 정도만 스캔
        # (너무 크면 인접 침상까지 포함되어 두께가 과대 추정됨)
        float_px_per_um = self.obj_config.float_scalePixels / self.obj_config.float_scaleMicrometers
        int_half_scan = max(15, int(0.5 * float_px_per_um))  # ±0.5µm
        int_center = int_half_scan  # 배열 중심 인덱스
        arr_scan = np.arange(-int_half_scan, int_half_scan + 1, dtype=np.float32)

        list_widths: tp.List[float] = []
        for float_t in np.linspace(0.2, 0.8, CONST_LSD_PERP_N_SAMPLES):
            float_sx = float_x1 + float_t * float_dx
            float_sy = float_y1 + float_t * float_dy

            arr_xs = np.clip(float_sx + float_px * arr_scan, 0, int_roiW - 1).astype(np.int32)
            arr_ys = np.clip(float_sy + float_py * arr_scan, 0, int_roiH - 1).astype(np.int32)
            arr_profile = arr_gray[arr_ys, arr_xs].astype(np.float32)

            # 전역 Otsu threshold로 밝은 영역(침상)을 판정
            arr_above = arr_profile > arr_global_thresh
            if not arr_above.any():
                continue

            # 연속된 밝은 구간(regions) 추출
            list_regions: tp.List[tp.Tuple[int, int]] = []
            bool_in = False
            int_start = 0
            for int_k, bool_v in enumerate(arr_above.tolist()):
                if bool_v and not bool_in:
                    bool_in = True
                    int_start = int_k
                elif not bool_v and bool_in:
                    bool_in = False
                    list_regions.append((int_start, int_k - 1))
            if bool_in:
                list_regions.append((int_start, len(arr_above) - 1))

            if not list_regions:
                continue

            # 스캔 중심(= 이 침상의 엣지 위치)에 가장 가까운 밝은 구간 선택
            def _dist_to_center(tpl: tp.Tuple[int, int]) -> int:
                return min(abs(tpl[0] - int_center), abs(tpl[1] - int_center))

            tpl_best = min(list_regions, key=_dist_to_center)
            int_width = tpl_best[1] - tpl_best[0] + 1
            if int_width > 1:
                list_widths.append(float(int_width))

        return float(np.median(list_widths)) if list_widths else 0.0

    def analyze_with_lsd(
        self,
        arr_roiGray: np.ndarray,
        arr_roiBgr: np.ndarray,
    ) -> tp.Tuple[tp.List[PrimaryParticleMeasurement], tp.List[np.ndarray], np.ndarray]:
        """LSD(Line Segment Detector)로 침상을 탐지하고 수직 프로파일로 두께를 측정한다.

        SAM2 없이 순수 OpenCV로 수행:
        1. LSD → 모든 선분 검출
        2. 길이·종횡비 필터로 침상 후보 선별
        3. 중복 선분 제거 (동일 침상에서 나온 복수 검출)
        4. 각 선분에 수직으로 강도 프로파일 샘플링 → 실제 두께
        5. 각 침상의 직사각형 mask 생성 (overlay용)

        Returns:
            (list_objects, list_masks, arr_debug):
            - list_objects: PrimaryParticleMeasurement 목록.
            - list_masks: 각 침상의 직사각형 binary mask 목록.
            - arr_debug: LSD segment가 그려진 BGR 디버그 이미지.
        """
        int_roiH, int_roiW = arr_roiGray.shape[:2]

        # 전처리
        obj_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr_eq = obj_clahe.apply(arr_roiGray)
        arr_blur = cv2.GaussianBlur(arr_eq, (3, 3), 0)

        # 전역 Otsu threshold (수직 프로파일 측정에 사용)
        float_otsu_thresh, _ = cv2.threshold(arr_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # LSD 검출
        obj_lsd = cv2.createLineSegmentDetector(0)
        arr_lines, arr_widths, _, _ = obj_lsd.detect(arr_blur)

        list_objects: tp.List[PrimaryParticleMeasurement] = []
        list_masks: tp.List[np.ndarray] = []

        if arr_lines is None:
            return list_objects, list_masks, arr_roiBgr.copy()

        # 1차 필터: 길이 + 대략적 AR
        list_cands: tp.List[tp.Dict[str, float]] = []
        float_ar_loose = min(self.obj_primary_config.float_acicularThreshold + 0.20, 0.65)
        for int_i, arr_line in enumerate(arr_lines):
            float_x1, float_y1, float_x2, float_y2 = arr_line[0]
            float_len = float(np.sqrt((float_x2 - float_x1) ** 2 + (float_y2 - float_y1) ** 2))
            if float_len < CONST_LSD_MIN_LENGTH_PX:
                continue
            float_lsd_w = float(arr_widths[int_i][0]) if arr_widths is not None else 5.0
            if float_len > 0 and float_lsd_w / float_len >= float_ar_loose:
                continue
            float_angle = float(np.degrees(np.arctan2(float_y2 - float_y1, float_x2 - float_x1)) % 180)
            list_cands.append({
                "x1": float_x1, "y1": float_y1, "x2": float_x2, "y2": float_y2,
                "length": float_len, "lsd_w": float_lsd_w, "angle": float_angle,
            })

        # 중복 제거: 유사 위치+방향의 선분 제거 (긴 것 우선)
        list_cands.sort(key=lambda d: d["length"], reverse=True)
        list_accepted: tp.List[tp.Dict[str, float]] = []
        for dict_c in list_cands:
            float_cx = (dict_c["x1"] + dict_c["x2"]) / 2.0
            float_cy = (dict_c["y1"] + dict_c["y2"]) / 2.0
            bool_dup = False
            for dict_p in list_accepted:
                float_pcx = (dict_p["x1"] + dict_p["x2"]) / 2.0
                float_pcy = (dict_p["y1"] + dict_p["y2"]) / 2.0
                float_dist = float(np.sqrt((float_cx - float_pcx) ** 2 + (float_cy - float_pcy) ** 2))
                float_adiff = abs(dict_c["angle"] - dict_p["angle"])
                float_adiff = min(float_adiff, 180.0 - float_adiff)
                if float_dist < CONST_LSD_DEDUP_DIST_PX and float_adiff < CONST_LSD_DEDUP_ANGLE_DEG:
                    bool_dup = True
                    break
            if not bool_dup:
                list_accepted.append(dict_c)

        # 수직 프로파일 두께 측정 + 분류
        for int_idx, dict_c in enumerate(list_accepted):
            float_x1 = dict_c["x1"]
            float_y1 = dict_c["y1"]
            float_x2 = dict_c["x2"]
            float_y2 = dict_c["y2"]
            float_len = dict_c["length"]

            float_thickness = self._measure_perpendicular_thickness(
                arr_blur, float_otsu_thresh, float_x1, float_y1, float_x2, float_y2)
            if float_thickness < 2.0:
                continue

            float_ar = float_thickness / float_len

            # 분류
            if float_ar < self.obj_primary_config.float_acicularThreshold:
                str_category = (
                    "acicular"
                    if self.obj_primary_config.str_particleType != "plate"
                    else "fragment"
                )
            else:
                str_category = (
                    "plate"
                    if self.obj_primary_config.str_particleType != "acicular"
                    else "fragment"
                )

            # particle_type 명시 시 목표 아닌 것 버림
            if (self.obj_primary_config.str_particleType in ("acicular", "plate")
                    and str_category == "fragment"):
                continue

            # 침상을 감싸는 직사각형 마스크
            float_ux = (float_x2 - float_x1) / max(float_len, 1.0)
            float_uy = (float_y2 - float_y1) / max(float_len, 1.0)
            float_px_dir = -float_uy
            float_py_dir = float_ux
            float_half_t = float_thickness / 2.0

            arr_corners = np.float32([
                [float_x1 - float_half_t * float_px_dir, float_y1 - float_half_t * float_py_dir],
                [float_x1 + float_half_t * float_px_dir, float_y1 + float_half_t * float_py_dir],
                [float_x2 + float_half_t * float_px_dir, float_y2 + float_half_t * float_py_dir],
                [float_x2 - float_half_t * float_px_dir, float_y2 - float_half_t * float_py_dir],
            ])

            int_bx, int_by, int_bw, int_bh = cv2.boundingRect(arr_corners.astype(np.int32))
            int_bx = max(0, int_bx)
            int_by = max(0, int_by)
            int_bw = min(int_bw, int_roiW - int_bx)
            int_bh = min(int_bh, int_roiH - int_by)

            if self.is_bbox_near_roi_edge(int_bx, int_by, int_bw, int_bh, int_roiW, int_roiH):
                continue

            arr_mask = np.zeros((int_roiH, int_roiW), dtype=np.uint8)
            cv2.fillPoly(arr_mask, [arr_corners.astype(np.int32).reshape(-1, 1, 2)], 1)

            float_cx = (float_x1 + float_x2) / 2.0
            float_cy = (float_y1 + float_y2) / 2.0

            list_objects.append(PrimaryParticleMeasurement(
                int_index=int_idx,
                str_category=str_category,
                int_maskArea=int(arr_mask.sum()),
                float_confidence=None,
                int_bboxX=int_bx,
                int_bboxY=int_by,
                int_bboxWidth=int_bw,
                int_bboxHeight=int_bh,
                float_centroidX=float_cx,
                float_centroidY=float_cy,
                float_thicknessPx=float_thickness,
                float_longAxisPx=float_len,
                float_minRectAngle=dict_c["angle"],
                float_thicknessUm=self.convert_pixels_to_micrometers(float_thickness),
                float_longAxisUm=self.convert_pixels_to_micrometers(float_len),
                float_aspectRatio=float_ar,
                int_longestHorizontal=int_bw,
                int_longestVertical=int_bh,
                float_longestHorizontalUm=self.convert_pixels_to_micrometers(float(int_bw)),
                float_longestVerticalUm=self.convert_pixels_to_micrometers(float(int_bh)),
            ))
            list_masks.append(arr_mask)

        # 디버그 이미지: 수락된 segment 표시
        arr_debug = arr_roiBgr.copy()
        for dict_c in list_accepted:
            cv2.line(arr_debug,
                     (int(dict_c["x1"]), int(dict_c["y1"])),
                     (int(dict_c["x2"]), int(dict_c["y2"])),
                     (0, 255, 255), 1, cv2.LINE_AA)

        print(
            f"[LSD] 선분 {len(arr_lines)}개 → 필터 {len(list_cands)}개 → "
            f"중복제거 {len(list_accepted)}개 → 측정성공 {len(list_objects)}개",
            flush=True,
        )
        return list_objects, list_masks, arr_debug

    def _extract_blob_centroids(
        self,
        list_bboxes: tp.List[tp.Tuple[int, int, int, int]],
    ) -> tp.List[tp.Tuple[int, int]]:
        """bbox 목록에서 centroid 좌표 목록을 계산한다."""
        return [
            (int_x1 + (int_x2 - int_x1) // 2, int_y1 + (int_y2 - int_y1) // 2)
            for (int_x1, int_y1, int_x2, int_y2) in list_bboxes
        ]

    def predict_with_guided_point_prompts(
        self,
        arr_roi: np.ndarray,
        list_points: tp.List[tp.Tuple[int, int]],
    ) -> tp.Tuple[np.ndarray, tp.Optional[np.ndarray], tp.Dict[str, tp.Any]]:
        """OpenCV 검출 centroid를 SAM2 point prompt로 전달한다.

        box prompt는 작은 입자에서 SAM2가 박스 전체를 채우는 과분할을 일으키지만,
        point prompt는 입자 중심만 알려주어 SAM2가 자연스럽게 경계를 찾는다.
        """
        if self.obj_model is None:
            self.initialize_model()

        int_roiH, int_roiW = arr_roi.shape[:2]

        dict_predictCommon: tp.Dict[str, tp.Any] = {
            "imgsz": self.obj_config.int_imgSize,
            "retina_masks": self.obj_config.bool_retinaMasks,
            "verbose": False,
        }
        if self.obj_config.str_device:
            dict_predictCommon["device"] = self.obj_config.str_device

        list_keptMasks: tp.List[np.ndarray] = []
        list_keptScores: tp.List[tp.Optional[float]] = []
        list_keptBboxes: tp.List[tp.Tuple[int, int, int, int]] = []
        int_acceptedCount = 0
        int_bboxDedupRejected = 0

        int_bs = max(1, self.obj_config.int_pointBatchSize)
        for int_start in range(0, len(list_points), int_bs):
            list_chunk = list_points[int_start:int_start + int_bs]
            try:
                list_results = self.obj_model(  # type: ignore[misc]
                    source=arr_roi,
                    points=[[int_px, int_py] for (int_px, int_py) in list_chunk],
                    labels=[1] * len(list_chunk),
                    **dict_predictCommon,
                )
            except Exception as exc:
                print(f"[WARN] guided point prompt batch 실패: {exc}", flush=True)
                continue

            if not list_results:
                continue
            obj_result = list_results[0]
            if obj_result.masks is None or obj_result.masks.data is None:
                continue

            arr_batchMasks = obj_result.masks.data.detach().cpu().numpy()
            arr_batchScores = None
            if obj_result.boxes is not None and obj_result.boxes.conf is not None:
                arr_batchScores = obj_result.boxes.conf.detach().cpu().numpy()

            for int_mi, arr_rawMask in enumerate(arr_batchMasks):
                arr_mask = (arr_rawMask > self.obj_config.float_maskBinarizeThreshold).astype(np.uint8)
                if int(arr_mask.sum()) < self.obj_config.int_minValidMaskArea:
                    continue
                arr_contour = self.extract_largest_contour(arr_mask)
                if arr_contour is None:
                    continue
                int_bx, int_by, int_bw, int_bh = cv2.boundingRect(arr_contour)
                if self.is_bbox_near_roi_edge(int_bx, int_by, int_bw, int_bh, int_roiW, int_roiH):
                    continue
                tuple_box = (int_bx, int_by, int_bw, int_bh)
                if any(calculate_box_iou(tuple_box, prev) >= self.obj_config.float_bboxDedupIou
                       for prev in list_keptBboxes):
                    int_bboxDedupRejected += 1
                    continue
                if any(calculate_binary_iou(arr_mask, prev) >= self.obj_config.float_dedupIou
                       for prev in list_keptMasks):
                    continue
                int_acceptedCount += 1
                list_keptMasks.append(arr_mask)
                list_keptBboxes.append(tuple_box)
                float_sc: tp.Optional[float] = None
                if arr_batchScores is not None and int_mi < len(arr_batchScores):
                    float_sc = float(arr_batchScores[int_mi])
                list_keptScores.append(float_sc)

        arr_masks = (
            np.stack(list_keptMasks, axis=0).astype(np.uint8)
            if list_keptMasks else np.empty((0, int_roiH, int_roiW), dtype=np.uint8)
        )
        arr_scores: tp.Optional[np.ndarray] = None
        if list_keptScores:
            arr_scores = np.array(
                [np.nan if x is None else float(x) for x in list_keptScores],
                dtype=np.float32,
            )
        return arr_masks, arr_scores, {
            "num_tiles": 1,
            "num_guided_points": len(list_points),
            "num_candidate_points": len(list_points),
            "num_accepted_masks": int_acceptedCount,
            "num_bbox_dedup_rejected": int_bboxDedupRejected,
        }

    def process_acicular_hybrid(
        self,
        arr_inputRoiBgr: np.ndarray,
    ) -> tp.Tuple[np.ndarray, tp.Optional[np.ndarray], tp.Dict[str, tp.Any],
                  tp.Optional[np.ndarray]]:
        """OpenCV centroid → SAM2 guided point prompt hybrid 파이프라인.

        OpenCV로 침상 후보 blob의 centroid를 추출하고,
        이를 SAM2 point prompt로 전달해 정밀 mask를 얻는다.
        box prompt 대신 point prompt를 쓰는 이유: 작은 입자에서 box는
        SAM2가 박스 전체를 채우는 과분할을 일으키지만 point는 그렇지 않다.

        Returns:
            (arr_masks, arr_scores, dict_debug, arr_opencvDebugMask)
        """
        arr_roiGray = cv2.cvtColor(arr_inputRoiBgr, cv2.COLOR_BGR2GRAY)
        int_roiH, int_roiW = arr_inputRoiBgr.shape[:2]

        float_arScreen = min(
            CONST_ACICULAR_CANDIDATE_AR_SCREEN,
            self.obj_primary_config.float_acicularThreshold + 0.25,
        )
        list_bboxes, arr_opencvMask = self.detect_acicular_candidates_opencv(
            arr_roiGray,
            float_arScreen=float_arScreen,
            float_minArea=max(8.0, self.obj_config.float_particleAreaThreshold * 0.5),
        )

        print(
            f"[acicular-hybrid] OpenCV 후보 침상: {len(list_bboxes)}개", flush=True)

        if len(list_bboxes) >= CONST_ACICULAR_FALLBACK_THRESHOLD:
            # OpenCV centroid → SAM2 guided point prompts
            list_centroids = self._extract_blob_centroids(list_bboxes)
            arr_masks, arr_scores, dict_debug = self.predict_with_guided_point_prompts(
                arr_inputRoiBgr, list_centroids)
            dict_debug["opencv_candidates"] = len(list_bboxes)
            dict_debug["mode"] = "opencv_guided_points"

            # 결과 부족 시 tiled point prompt로 보완
            if len(arr_masks) < CONST_ACICULAR_FALLBACK_THRESHOLD:
                print("[acicular-hybrid] guided point 결과 부족 → tiled point 보완",
                      flush=True)
                arr_ptMasks, arr_ptScores, dict_ptDebug = self.predict_tiled_point_prompts(
                    arr_inputRoiBgr)
                arr_masks, arr_scores = self._merge_mask_results(
                    arr_masks, arr_scores, arr_ptMasks, arr_ptScores,
                    int_roiH=int_roiH, int_roiW=int_roiW,
                )
                dict_debug["fallback_point_masks"] = int(arr_ptMasks.shape[0])
                dict_debug["mode"] = "opencv_guided+tiled_fallback"
        else:
            print("[acicular-hybrid] OpenCV 후보 부족 → tiled point fallback",
                  flush=True)
            arr_masks, arr_scores, dict_debug = self.predict_tiled_point_prompts(
                arr_inputRoiBgr)
            dict_debug["opencv_candidates"] = len(list_bboxes)
            dict_debug["mode"] = "tiled_point_fallback"
            arr_opencvMask = None

        return arr_masks, arr_scores, dict_debug, arr_opencvMask

    # ----------------------------------------------------------
    # 시각화
    # ----------------------------------------------------------

    def create_primary_overlay(
        self,
        arr_imageBgr: np.ndarray,
        list_objects: tp.List[PrimaryParticleMeasurement],
        list_masks: tp.List[np.ndarray],
    ) -> np.ndarray:
        """침상(파랑)/판상(초록)/fragment(주황) 색으로 overlay를 생성한다."""
        # BGR 색상 (mask fill / contour+label)
        dict_fill = {
            "acicular": (230, 80,  80),
            "plate":    (60,  220, 60),
            "fragment": (0,   165, 255),
        }
        dict_edge = {
            "acicular": (180, 40,  40),
            "plate":    (0,   180, 0),
            "fragment": (0,   120, 200),
        }

        arr_overlay = arr_imageBgr.copy()

        # 반투명 색칠
        for obj_m, arr_mask in zip(list_objects, list_masks):
            tpl_c = dict_fill.get(obj_m.str_category, (128, 128, 128))
            arr_overlay[arr_mask > 0] = (
                arr_overlay[arr_mask > 0].astype(np.float32) * 0.5
                + np.array(tpl_c, dtype=np.float32) * 0.5
            ).astype(np.uint8)

        # contour + bbox + label
        for obj_m, arr_mask in zip(list_objects, list_masks):
            arr_contour = self.extract_largest_contour(arr_mask)
            if arr_contour is None:
                continue

            tpl_ec = dict_edge.get(obj_m.str_category, (128, 128, 128))
            cv2.drawContours(arr_overlay, [arr_contour], -1, tpl_ec, 1)
            cv2.rectangle(
                arr_overlay,
                (obj_m.int_bboxX, obj_m.int_bboxY),
                (
                    obj_m.int_bboxX + obj_m.int_bboxWidth,
                    obj_m.int_bboxY + obj_m.int_bboxHeight,
                ),
                tpl_ec, 1,
            )

            int_lx = obj_m.int_bboxX
            int_ly = max(14, obj_m.int_bboxY - 4)

            if obj_m.str_category == "fragment":
                str_label = f"F{obj_m.int_index} A={obj_m.int_maskArea}"
            else:
                str_prefix = "Ac" if obj_m.str_category == "acicular" else "Pl"
                str_label = (
                    f"{str_prefix}{obj_m.int_index} "
                    f"t={obj_m.float_thicknessUm:.2f}um "
                    f"AR={obj_m.float_aspectRatio:.2f}"
                )

            cv2.putText(
                arr_overlay, str_label,
                (int_lx, int_ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                tpl_ec, 1, cv2.LINE_AA,
            )

        return arr_overlay

    # ----------------------------------------------------------
    # 통계
    # ----------------------------------------------------------

    def build_primary_summary(
        self,
        list_objects: tp.List[PrimaryParticleMeasurement],
    ) -> tp.Dict[str, tp.Any]:
        """침상/판상별 두께·종횡비 통계를 포함한 summary dict 를 생성한다."""

        list_acicular = [o for o in list_objects if o.str_category == "acicular"]
        list_plate = [o for o in list_objects if o.str_category == "plate"]
        list_fragment = [o for o in list_objects if o.str_category == "fragment"]

        def _stats(
            list_vals: tp.List[float],
        ) -> tp.Dict[str, tp.Optional[float]]:
            if not list_vals:
                return {"mean": None, "median": None, "std": None, "min": None, "max": None}
            arr = np.array(list_vals, dtype=np.float32)
            return {
                "mean":   float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std":    float(np.std(arr)),
                "min":    float(np.min(arr)),
                "max":    float(np.max(arr)),
            }

        float_um_per_px = convert_pixels_to_micrometers(
            1.0,
            float_scalePixels=self.obj_config.float_scalePixels,
            float_scaleMicrometers=self.obj_config.float_scaleMicrometers,
        )

        return {
            "input_path": str(self.obj_config.path_input),
            "output_dir": str(self.obj_config.path_outputDir),
            "model_weights_path": str(self.obj_config.path_modelWeights),
            "particle_type": str(self.obj_primary_config.str_particleType),
            "magnification": str(self.obj_primary_config.str_magnification),
            "scale_pixels": float(self.obj_config.float_scalePixels),
            "scale_micrometers": float(self.obj_config.float_scaleMicrometers),
            "micrometers_per_pixel": float(float_um_per_px),
            "acicular_threshold": float(self.obj_primary_config.float_acicularThreshold),
            "particle_mode": str(self.obj_primary_config.str_particleMode),
            "auto_center_crop": bool(self.obj_primary_config.bool_autoCenterCrop),
            "center_crop_ratio": float(self.obj_primary_config.float_centerCropRatio),
            "particle_area_threshold": float(self.obj_config.float_particleAreaThreshold),
            "num_total_objects": len(list_objects),
            "num_acicular": len(list_acicular),
            "num_plate": len(list_plate),
            "num_fragment": len(list_fragment),
            "acicular_thickness_um": _stats([o.float_thicknessUm for o in list_acicular]),
            "acicular_long_axis_um": _stats([o.float_longAxisUm for o in list_acicular]),
            "acicular_aspect_ratio": _stats([o.float_aspectRatio for o in list_acicular]),
            "plate_thickness_um": _stats([o.float_thicknessUm for o in list_plate]),
            "plate_long_axis_um": _stats([o.float_longAxisUm for o in list_plate]),
            "plate_aspect_ratio": _stats([o.float_aspectRatio for o in list_plate]),
            "all_primary_thickness_um": _stats(
                [o.float_thicknessUm for o in list_acicular + list_plate]
            ),
        }

    # ----------------------------------------------------------
    # 저장
    # ----------------------------------------------------------

    def save_thickness_histogram(
        self,
        list_objects: tp.List[PrimaryParticleMeasurement],
        path_output: Path,
    ) -> None:
        """침상/판상별 두께 분포 histogram 을 PNG로 저장한다."""
        list_ac = [o.float_thicknessUm for o in list_objects if o.str_category == "acicular"]
        list_pl = [o.float_thicknessUm for o in list_objects if o.str_category == "plate"]

        str_lot = self.obj_config.path_input.resolve().parent.name or "UnknownLot"
        obj_fig, obj_ax = plt.subplots(figsize=(10, 6), dpi=100)
        try:
            obj_ax.set_title(f"{str_lot} — Primary Particle Thickness", fontsize=18)
            obj_ax.set_xlabel("Thickness (µm)", fontsize=14)
            obj_ax.set_ylabel("Count", fontsize=14)
            obj_ax.tick_params(labelsize=12)

            bool_hasData = False
            for list_vals, str_label, str_color in [
                (list_ac, "Acicular", "#5588ff"),
                (list_pl, "Plate",    "#44cc44"),
            ]:
                if list_vals:
                    bool_hasData = True
                    arr_v = np.array(list_vals, dtype=np.float32)
                    int_bins = int(np.clip(np.sqrt(len(arr_v)), 5, 20))
                    float_mean = float(np.mean(arr_v))
                    obj_ax.hist(
                        arr_v, bins=int_bins, alpha=0.65,
                        label=str_label, color=str_color,
                        edgecolor="#333333", linewidth=0.8,
                    )
                    obj_ax.axvline(float_mean, linestyle="--", linewidth=1.5, color=str_color)
                    float_ymax = obj_ax.get_ylim()[1]
                    obj_ax.text(
                        float_mean, float_ymax * 0.95,
                        f"  {str_label[:2]} mean: {float_mean:.3f} µm",
                        color=str_color, fontsize=11, va="top",
                    )

            if bool_hasData:
                obj_ax.legend(fontsize=12)
                obj_ax.grid(axis="y", linestyle="--", alpha=0.3)
            else:
                obj_ax.text(
                    0.5, 0.5, "No primary particle data",
                    ha="center", va="center",
                    transform=obj_ax.transAxes, fontsize=13, color="#666666",
                )

            obj_fig.tight_layout()
            obj_fig.savefig(str(path_output), bbox_inches="tight")
        finally:
            plt.close(obj_fig)

    def save_primary_outputs(
        self,
        arr_inputBgr: np.ndarray,
        arr_inputRoiBgr: np.ndarray,
        arr_overlayRoi: np.ndarray,
        list_objects: tp.List[PrimaryParticleMeasurement],
        list_masks: tp.List[np.ndarray],
        dict_summary: tp.Dict[str, tp.Any],
        dict_roi: tp.Dict[str, int],
        dict_debug: tp.Dict[str, tp.Any],
        arr_opencvDebugMask: tp.Optional[np.ndarray] = None,
    ) -> None:
        """이미지, CSV, JSON, histogram 등 1차 입자 분석 산출물을 저장한다."""
        self.obj_config.path_outputDir.mkdir(parents=True, exist_ok=True)

        # overlay_full: 원본 이미지 위에 ROI overlay를 덮는다
        arr_overlayFull = arr_inputBgr.copy()
        arr_overlayFull[
            dict_roi["y_min"]:dict_roi["y_max"],
            dict_roi["x_min"]:dict_roi["x_max"],
        ] = arr_overlayRoi
        cv2.rectangle(
            arr_overlayFull,
            (dict_roi["x_min"], dict_roi["y_min"]),
            (dict_roi["x_max"], dict_roi["y_max"]),
            (255, 255, 0), 2,
        )

        cv2.imwrite(str(self.obj_config.path_outputDir / "01_input.png"), arr_inputBgr)
        cv2.imwrite(str(self.obj_config.path_outputDir / "02_input_roi.png"), arr_inputRoiBgr)
        cv2.imwrite(str(self.obj_config.path_outputDir / "03_overlay_roi.png"), arr_overlayRoi)
        cv2.imwrite(str(self.obj_config.path_outputDir / "04_overlay_full.png"), arr_overlayFull)

        # OpenCV debug mask (침상 hybrid 모드에서만 생성)
        if arr_opencvDebugMask is not None:
            cv2.imwrite(
                str(self.obj_config.path_outputDir / "05_opencv_candidates.png"),
                arr_opencvDebugMask * 255,
            )

        # Sphere detection debug (자동 구 검출 모드에서만 생성)
        arr_sphereDbg: tp.Optional[np.ndarray] = getattr(self, "_arr_sphereDebug", None)
        if arr_sphereDbg is not None:
            cv2.imwrite(
                str(self.obj_config.path_outputDir / "06_sphere_detection.png"),
                arr_sphereDbg,
            )

        # objects.csv (전체)
        path_csvAll = self.obj_config.path_outputDir / "objects.csv"
        with path_csvAll.open("w", newline="", encoding="utf-8-sig") as obj_f:
            if list_objects:
                obj_writer = csv.DictWriter(
                    obj_f, fieldnames=list(asdict(list_objects[0]).keys()))
                obj_writer.writeheader()
                for obj_m in list_objects:
                    obj_writer.writerow(asdict(obj_m))

        # acicular.csv / plate.csv
        for str_cat, str_fname in [("acicular", "acicular.csv"), ("plate", "plate.csv")]:
            list_rows = [asdict(o) for o in list_objects if o.str_category == str_cat]
            path_csv = self.obj_config.path_outputDir / str_fname
            with path_csv.open("w", newline="", encoding="utf-8-sig") as obj_f:
                if list_rows:
                    obj_writer = csv.DictWriter(
                        obj_f, fieldnames=list(list_rows[0].keys()))
                    obj_writer.writeheader()
                    for dict_row in list_rows:
                        obj_writer.writerow(dict_row)

        # thickness 분포 histogram
        self.save_thickness_histogram(
            list_objects,
            self.obj_config.path_outputDir / "thickness_dist.png",
        )

        # JSON
        with (self.obj_config.path_outputDir / "summary.json").open(
                "w", encoding="utf-8") as obj_f:
            json.dump(dict_summary, obj_f, ensure_ascii=False, indent=2,
                      default=_json_default)
        with (self.obj_config.path_outputDir / "objects.json").open(
                "w", encoding="utf-8") as obj_f:
            json.dump(
                [asdict(o) for o in list_objects], obj_f, ensure_ascii=False, indent=2,
                default=_json_default)
        with (self.obj_config.path_outputDir / "debug.json").open(
                "w", encoding="utf-8") as obj_f:
            json.dump(dict_debug, obj_f, ensure_ascii=False, indent=2,
                      default=_json_default)

        if not self.obj_config.bool_saveIndividualMasks:
            return

        # 개별 mask png (acicular_masks / plate_masks / fragment_masks)
        for str_cat in ("acicular", "plate", "fragment"):
            (self.obj_config.path_outputDir / f"{str_cat}_masks").mkdir(
                parents=True, exist_ok=True)

        for obj_m, arr_mask in zip(list_objects, list_masks):
            str_fname = f"{obj_m.str_category}_{obj_m.int_index:04d}.png"
            path_maskDir = self.obj_config.path_outputDir / f"{obj_m.str_category}_masks"
            cv2.imwrite(str(path_maskDir / str_fname), arr_mask.astype(np.uint8) * 255)

    # ----------------------------------------------------------
    # 메인 파이프라인
    # ----------------------------------------------------------

    def process_primary(self) -> PrimaryParticleResult:
        """단일 이미지에 대한 1차 입자 분석 파이프라인을 실행한다."""
        arr_inputBgr = self.load_image_bgr()
        arr_inputRoiBgr, dict_roi = self.extract_inference_roi(arr_inputBgr)

        arr_opencvDebugMask: tp.Optional[np.ndarray] = None
        dict_debug: tp.Dict[str, tp.Any] = {}

        # ── LSD 직접 측정 모드 (SAM2 불필요) ────────────────────
        if self.obj_primary_config.str_measureMode == "lsd":
            arr_roiGray = cv2.cvtColor(arr_inputRoiBgr, cv2.COLOR_BGR2GRAY)
            list_objects, list_validMasks, arr_opencvDebugMask = self.analyze_with_lsd(
                arr_roiGray, arr_inputRoiBgr)

            int_primaryCount = sum(
                1 for o in list_objects if o.str_category in ("acicular", "plate"))
            if int_primaryCount < self.obj_primary_config.int_targetParticleCount:
                print(
                    f"[WARNING] 침상+판상 입자 수({int_primaryCount})가 목표치"
                    f"({self.obj_primary_config.int_targetParticleCount}) 미만입니다.",
                    flush=True,
                )

            arr_overlay = self.create_primary_overlay(arr_inputRoiBgr, list_objects, list_validMasks)
            dict_summary = self.build_primary_summary(list_objects)
            dict_summary["roi"] = dict_roi
            dict_summary["measure_mode"] = "lsd"
            dict_summary["particle_mode"] = self.obj_primary_config.str_particleMode

            self.save_primary_outputs(
                arr_inputBgr, arr_inputRoiBgr, arr_overlay,
                list_objects, list_validMasks,
                dict_summary, dict_roi, dict_debug,
                arr_opencvDebugMask=arr_opencvDebugMask,
            )
            return PrimaryParticleResult(list_objects=list_objects, dict_summary=dict_summary)

        # ── SAM2 모드 (기존 파이프라인) ─────────────────────────
        if self.obj_primary_config.str_particleMode == "acicular":
            arr_masks, arr_scores, dict_debug, arr_opencvDebugMask = (
                self.process_acicular_hybrid(arr_inputRoiBgr))
            dict_debug["particle_mode"] = "acicular"
        else:
            arr_masks, arr_scores, dict_debug = self.predict_tiled_point_prompts(arr_inputRoiBgr)
            dict_debug["particle_mode"] = "auto"

        list_objects = []
        list_validMasks = []

        for int_index, arr_mask in enumerate(arr_masks):
            float_conf: tp.Optional[float] = None
            if arr_scores is not None and int_index < len(arr_scores):
                float_conf = float(arr_scores[int_index])

            obj_m = self.measure_primary_mask(arr_mask, int_index, float_conf)
            if obj_m is None:
                continue
            if (self.obj_primary_config.str_particleType in ("acicular", "plate")
                    and obj_m.str_category == "fragment"):
                continue
            list_objects.append(obj_m)
            list_validMasks.append(self.refine_mask_for_area(arr_mask).astype(np.uint8))

        int_primaryCount = sum(
            1 for o in list_objects if o.str_category in ("acicular", "plate"))
        if int_primaryCount < self.obj_primary_config.int_targetParticleCount:
            print(
                f"[WARNING] 침상+판상 입자 수({int_primaryCount})가 목표치"
                f"({self.obj_primary_config.int_targetParticleCount}) 미만입니다. "
                "ROI 조정 또는 파라미터 변경을 고려하세요.",
                flush=True,
            )

        arr_overlay = self.create_primary_overlay(arr_inputRoiBgr, list_objects, list_validMasks)
        dict_summary = self.build_primary_summary(list_objects)
        dict_summary["roi"] = dict_roi
        dict_summary["measure_mode"] = "sam2"
        dict_summary["particle_mode"] = self.obj_primary_config.str_particleMode
        dict_summary["num_tiles"] = dict_debug.get("num_tiles")
        dict_summary["num_candidate_points"] = dict_debug.get("num_candidate_points")
        dict_summary["num_accepted_masks"] = dict_debug.get("num_accepted_masks")
        dict_summary["num_bbox_dedup_rejected"] = dict_debug.get("num_bbox_dedup_rejected")
        if "opencv_candidates" in dict_debug:
            dict_summary["opencv_candidates"] = dict_debug["opencv_candidates"]

        self.save_primary_outputs(
            arr_inputBgr, arr_inputRoiBgr, arr_overlay,
            list_objects, list_validMasks,
            dict_summary, dict_roi, dict_debug,
            arr_opencvDebugMask=arr_opencvDebugMask,
        )

        return PrimaryParticleResult(list_objects=list_objects, dict_summary=dict_summary)


# =========================================================
# 배치 집계
# =========================================================


def build_primary_img_id_summary(
    str_imgId: str,
    path_outputRoot: Path,
    list_fileSummaries: tp.List[tp.Dict[str, tp.Any]],
) -> tp.Dict[str, tp.Any]:
    """동일 IMG_ID 그룹의 1차 입자 summary 들을 집계한다."""

    def _mean_stat(str_key: str, str_stat: str) -> tp.Optional[float]:
        list_vals = [
            d[str_key][str_stat]
            for d in list_fileSummaries
            if isinstance(d.get(str_key), dict)
            and d[str_key].get(str_stat) is not None
        ]
        return calculate_mean_from_optional_values(list_vals)

    return {
        "img_id": str_imgId,
        "output_dir": str(path_outputRoot / str_imgId),
        "num_images": len(list_fileSummaries),
        "num_total_objects": int(
            sum(d.get("num_total_objects", 0) for d in list_fileSummaries)),
        "num_acicular": int(
            sum(d.get("num_acicular", 0) for d in list_fileSummaries)),
        "num_plate": int(
            sum(d.get("num_plate", 0) for d in list_fileSummaries)),
        "num_fragment": int(
            sum(d.get("num_fragment", 0) for d in list_fileSummaries)),
        "acicular_thickness_um_mean": _mean_stat("acicular_thickness_um", "mean"),
        "acicular_long_axis_um_mean": _mean_stat("acicular_long_axis_um", "mean"),
        "acicular_aspect_ratio_mean": _mean_stat("acicular_aspect_ratio", "mean"),
        "plate_thickness_um_mean": _mean_stat("plate_thickness_um", "mean"),
        "plate_long_axis_um_mean": _mean_stat("plate_long_axis_um", "mean"),
        "plate_aspect_ratio_mean": _mean_stat("plate_aspect_ratio", "mean"),
        "all_primary_thickness_um_mean": _mean_stat("all_primary_thickness_um", "mean"),
        "files": list_fileSummaries,
    }


def build_primary_batch_summary(
    path_input: Path,
    path_outputDir: Path,
    list_groupSummaries: tp.List[tp.Dict[str, tp.Any]],
) -> tp.Dict[str, tp.Any]:
    """1차 입자 배치 전체 통합 summary 를 생성한다."""
    return {
        "input_path": str(path_input),
        "output_dir": str(path_outputDir),
        "num_img_ids": len(list_groupSummaries),
        "num_images": int(
            sum(d.get("num_images", 0) for d in list_groupSummaries)),
        "num_total_objects": int(
            sum(d.get("num_total_objects", 0) for d in list_groupSummaries)),
        "num_acicular": int(
            sum(d.get("num_acicular", 0) for d in list_groupSummaries)),
        "num_plate": int(
            sum(d.get("num_plate", 0) for d in list_groupSummaries)),
        "num_fragment": int(
            sum(d.get("num_fragment", 0) for d in list_groupSummaries)),
        "acicular_thickness_um_mean": calculate_mean_from_optional_values(
            d.get("acicular_thickness_um_mean") for d in list_groupSummaries),
        "plate_thickness_um_mean": calculate_mean_from_optional_values(
            d.get("plate_thickness_um_mean") for d in list_groupSummaries),
        "all_primary_thickness_um_mean": calculate_mean_from_optional_values(
            d.get("all_primary_thickness_um_mean") for d in list_groupSummaries),
        "img_ids": list_groupSummaries,
    }


# =========================================================
# 최상위 실행 함수
# =========================================================


def run_primary_particle_analysis(
    str_input: str,
    str_outputDir: str,
    str_modelConfig: str,
    str_modelWeights: str,
    # center crop
    bool_autoCenterCrop: bool = True,
    float_centerCropRatio: float = CONST_CENTER_CROP_RATIO,
    # ROI (auto_center_crop=False 일 때 사용)
    int_roiXMin: int = CONST_ROI_X_MIN,
    int_roiYMin: int = CONST_ROI_Y_MIN,
    int_roiXMax: int = CONST_ROI_X_MAX,
    int_roiYMax: int = CONST_ROI_Y_MAX,
    int_bboxEdgeMargin: int = CONST_BBOX_EDGE_MARGIN,
    int_tileEdgeMargin: int = CONST_TILE_EDGE_MARGIN,
    # 분류
    float_acicularThreshold: float = CONST_ACICULAR_THRESHOLD,
    float_particleAreaThreshold: float = CONST_PRIMARY_PARTICLE_AREA_THRESHOLD,
    int_targetParticleCount: int = CONST_TARGET_PARTICLE_COUNT,
    # mask 후처리
    float_maskBinarizeThreshold: float = CONST_MASK_BINARIZE_THRESHOLD,
    int_minValidMaskArea: int = CONST_MIN_VALID_MASK_AREA,
    int_maskMorphKernelSize: int = CONST_MASK_MORPH_KERNEL_SIZE,
    int_maskMorphOpenIterations: int = CONST_MASK_MORPH_OPEN_ITERATIONS,
    int_maskMorphCloseIterations: int = CONST_MASK_MORPH_CLOSE_ITERATIONS,
    # SAM2 추론
    int_imgSize: int = CONST_DEFAULT_IMAGE_SIZE,
    int_tileSize: int = CONST_PRIMARY_TILE_SIZE,
    int_stride: int = CONST_PRIMARY_TILE_STRIDE,
    int_pointsPerTile: int = CONST_PRIMARY_POINTS_PER_TILE,
    int_pointMinDistance: int = CONST_PRIMARY_POINT_MIN_DISTANCE,
    float_pointQualityLevel: float = CONST_DEFAULT_POINT_QUALITY_LEVEL,
    int_pointBatchSize: int = CONST_DEFAULT_POINT_BATCH_SIZE,
    float_dedupIou: float = CONST_DEFAULT_DEDUP_IOU,
    float_bboxDedupIou: float = CONST_DEFAULT_BBOX_DEDUP_IOU,
    bool_usePointPrompts: bool = CONST_DEFAULT_USE_POINT_PROMPTS,
    # 스케일
    float_scalePixels: float = CONST_SCALE_PIXELS,
    float_scaleMicrometers: float = CONST_SCALE_MICROMETERS,
    # 기타
    str_device: tp.Optional[str] = None,
    bool_retinaMasks: bool = CONST_DEFAULT_RETINA_MASKS,
    bool_saveIndividualMasks: bool = CONST_DEFAULT_SAVE_INDIVIDUAL_MASKS,
    str_particleMode: str = "auto",
    bool_autoDetectSphere: bool = False,
    float_sphereCapFraction: float = CONST_SPHERE_CAP_FRACTION,
    str_particleType: str = "unknown",
    str_magnification: str = "unknown",
    str_measureMode: str = "sam2",
) -> tp.Dict[str, tp.Any]:
    """외부에서 호출 가능한 최상위 실행 함수.

    Args:
        str_input: 단일 이미지 또는 batch root directory 경로.
        str_outputDir: 결과 저장 root directory 경로.
        str_modelConfig: SAM2 설정 파일 경로.
        str_modelWeights: SAM2 weight 파일 경로.
        (나머지 파라미터는 build_primary_arg_parser 설명 참조)

    Returns:
        단일 입력이면 단일 이미지 summary dict,
        directory 입력이면 batch summary dict.
    """
    path_input = Path(str_input)
    path_outputRoot = Path(str_outputDir)
    list_inputGroups = collect_input_groups(path_input)
    bool_isBatch = path_input.is_dir()

    def _create_config(
        str_groupId: str, path_image: Path
    ) -> PrimaryParticleConfig:
        return PrimaryParticleConfig(
            path_input=path_image,
            path_outputDir=build_image_output_dir(
                path_outputRoot, str_groupId, path_image, bool_isBatch),
            path_modelConfig=Path(str_modelConfig),
            path_modelWeights=Path(str_modelWeights),
            int_roiXMin=int_roiXMin,
            int_roiYMin=int_roiYMin,
            int_roiXMax=int_roiXMax,
            int_roiYMax=int_roiYMax,
            int_bboxEdgeMargin=int_bboxEdgeMargin,
            int_tileEdgeMargin=int_tileEdgeMargin,
            float_particleAreaThreshold=float_particleAreaThreshold,
            float_maskBinarizeThreshold=float_maskBinarizeThreshold,
            int_minValidMaskArea=int_minValidMaskArea,
            int_maskMorphKernelSize=int_maskMorphKernelSize,
            int_maskMorphOpenIterations=int_maskMorphOpenIterations,
            int_maskMorphCloseIterations=int_maskMorphCloseIterations,
            int_imgSize=int_imgSize,
            int_tileSize=int_tileSize,
            int_stride=int_stride,
            int_pointsPerTile=int_pointsPerTile,
            int_pointMinDistance=int_pointMinDistance,
            float_pointQualityLevel=float_pointQualityLevel,
            int_pointBatchSize=int_pointBatchSize,
            float_dedupIou=float_dedupIou,
            float_bboxDedupIou=float_bboxDedupIou,
            bool_usePointPrompts=bool_usePointPrompts,
            bool_smallParticle=False,
            float_scalePixels=float_scalePixels,
            float_scaleMicrometers=float_scaleMicrometers,
            str_device=str_device,
            bool_retinaMasks=bool_retinaMasks,
            bool_saveIndividualMasks=bool_saveIndividualMasks,
            float_acicularThreshold=float_acicularThreshold,
            bool_autoCenterCrop=bool_autoCenterCrop,
            float_centerCropRatio=float_centerCropRatio,
            int_targetParticleCount=int_targetParticleCount,
            str_particleMode=str_particleMode,
            bool_autoDetectSphere=bool_autoDetectSphere,
            float_sphereCapFraction=float_sphereCapFraction,
            str_particleType=str_particleType,
            str_magnification=str_magnification,
            str_measureMode=str_measureMode,
        )

    # 단일 이미지
    if not bool_isBatch:
        str_groupId, list_imagePaths = list_inputGroups[0]
        print(f"[single] processing: {list_imagePaths[0].name}", flush=True)
        obj_service = PrimaryParticleService(_create_config(str_groupId, list_imagePaths[0]))
        obj_result = obj_service.process_primary()
        print(f"[single] done: {list_imagePaths[0].name}", flush=True)
        return obj_result.dict_summary

    # 배치
    path_outputRoot.mkdir(parents=True, exist_ok=True)

    str_firstGroupId, list_firstImages = list_inputGroups[0]
    print(f"[batch] init model: {list_firstImages[0].name}", flush=True)
    obj_sharedService = PrimaryParticleService(
        _create_config(str_firstGroupId, list_firstImages[0]))
    obj_sharedService.initialize_model()

    list_groupSummaries: tp.List[tp.Dict[str, tp.Any]] = []
    int_numGroups = len(list_inputGroups)

    for int_gi, (str_groupId, list_imagePaths) in enumerate(list_inputGroups, start=1):
        print(
            f"[batch][group {int_gi}/{int_numGroups}] IMG_ID={str_groupId} "
            f"({len(list_imagePaths)} images)",
            flush=True,
        )
        list_fileSummaries: tp.List[tp.Dict[str, tp.Any]] = []
        int_numImages = len(list_imagePaths)

        for int_ii, path_image in enumerate(list_imagePaths, start=1):
            print(f"  [image {int_ii}/{int_numImages}] {path_image.name}", flush=True)
            obj_service = PrimaryParticleService(_create_config(str_groupId, path_image))
            obj_service.obj_model = obj_sharedService.obj_model
            obj_service.dict_modelConfig = dict(obj_sharedService.dict_modelConfig)
            obj_result = obj_service.process_primary()
            dict_fs = dict(obj_result.dict_summary)
            dict_fs["img_id"] = str_groupId
            dict_fs["image_name"] = path_image.name
            list_fileSummaries.append(dict_fs)

        dict_groupSummary = build_primary_img_id_summary(
            str_groupId, path_outputRoot, list_fileSummaries)
        path_groupDir = path_outputRoot / str_groupId
        path_groupDir.mkdir(parents=True, exist_ok=True)
        with (path_groupDir / "img_id_summary.json").open("w", encoding="utf-8") as obj_f:
            json.dump(dict_groupSummary, obj_f, ensure_ascii=False, indent=2,
                      default=_json_default)
        list_groupSummaries.append(dict_groupSummary)

        print(
            f"[batch][group done] {str_groupId}  "
            f"acicular={dict_groupSummary['num_acicular']}  "
            f"plate={dict_groupSummary['num_plate']}  "
            f"fragment={dict_groupSummary['num_fragment']}",
            flush=True,
        )

    dict_batchSummary = build_primary_batch_summary(
        path_input, path_outputRoot, list_groupSummaries)
    with (path_outputRoot / "batch_summary.json").open("w", encoding="utf-8") as obj_f:
        json.dump(dict_batchSummary, obj_f, ensure_ascii=False, indent=2,
                  default=_json_default)

    print(
        f"[batch] done: {dict_batchSummary['num_img_ids']} groups, "
        f"{dict_batchSummary['num_images']} images  "
        f"total acicular={dict_batchSummary['num_acicular']} "
        f"plate={dict_batchSummary['num_plate']}",
        flush=True,
    )
    return dict_batchSummary


# =========================================================
# CLI
# =========================================================


def build_primary_arg_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 생성한다."""
    str_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    obj_parser = argparse.ArgumentParser(
        description=(
            "SAM2로 1차 입자를 segmentation하고 "
            "침상(acicular)/판상(plate)으로 분류하여 두께를 측정합니다."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 입출력 / 모델
    obj_parser.add_argument(
        "--input", default="img/primary_test.jpg",
        help="입력 이미지 또는 디렉터리 경로")
    obj_parser.add_argument(
        "--output_dir", default=f"out_primary_{str_ts}",
        help="결과 저장 폴더")
    obj_parser.add_argument(
        "--model_cfg", default="model/sam2.1_hiera_t.yaml",
        help="SAM2 YAML 설정 파일 경로")
    obj_parser.add_argument(
        "--model", default="model/sam2.1_hiera_base_plus.pt",
        help="SAM2 가중치 파일 경로")

    # 중앙 crop
    obj_parser.add_argument(
        "--auto_center_crop",
        action=argparse.BooleanOptionalAction, default=True,
        help="자동 중앙 crop 사용 여부. False 이면 --roi_* 파라미터로 직접 지정")
    obj_parser.add_argument(
        "--center_crop_ratio", type=float, default=CONST_CENTER_CROP_RATIO,
        help="중앙 crop 비율 (0.1 ~ 1.0). 예: 0.6 → 이미지 중앙 60%% 사용")

    # ROI (auto_center_crop=False 일 때)
    obj_parser.add_argument("--roi_x_min", type=int, default=CONST_ROI_X_MIN)
    obj_parser.add_argument("--roi_y_min", type=int, default=CONST_ROI_Y_MIN)
    obj_parser.add_argument("--roi_x_max", type=int, default=CONST_ROI_X_MAX)
    obj_parser.add_argument("--roi_y_max", type=int, default=CONST_ROI_Y_MAX)
    obj_parser.add_argument(
        "--bbox_edge_margin", type=int, default=CONST_BBOX_EDGE_MARGIN,
        help="ROI 경계 근처 bbox 제외 margin")
    obj_parser.add_argument(
        "--tile_edge_margin", type=int, default=CONST_TILE_EDGE_MARGIN,
        help="타일 경계 근처 bbox 제외 margin")

    # 분류 기준
    obj_parser.add_argument(
        "--acicular_threshold", type=float, default=CONST_ACICULAR_THRESHOLD,
        help="침상/판상 분류 aspect_ratio 임계값. "
             "aspect_ratio(= 두께/장축) < 이 값 → 침상, >= 이 값 → 판상")
    obj_parser.add_argument(
        "--area_threshold", type=float, default=CONST_PRIMARY_PARTICLE_AREA_THRESHOLD,
        help="유효 1차 입자 최소 면적 (미만이면 fragment)")
    obj_parser.add_argument(
        "--target_particle_count", type=int, default=CONST_TARGET_PARTICLE_COUNT,
        help="목표 침상+판상 입자 수 (미달 시 경고)")

    # 스케일 (SEM 이미지 배율에 맞게 설정 필요)
    obj_parser.add_argument(
        "--scale_pixels", type=float, default=CONST_SCALE_PIXELS,
        help="스케일 기준 pixel 수. 예: 276 (기본 = 276 px = 50 µm)")
    obj_parser.add_argument(
        "--scale_um", type=float, default=CONST_SCALE_MICROMETERS,
        help="스케일 기준 µm 값. 예: 50 (기본 = 276 px = 50 µm). "
             "소입자 스케일 예시: --scale_pixels 184 --scale_um 10")

    # SAM2 추론
    obj_parser.add_argument(
        "--imgsz", type=int, default=CONST_DEFAULT_IMAGE_SIZE,
        help="SAM2 추론 이미지 크기")
    obj_parser.add_argument(
        "--tile_size", type=int, default=CONST_PRIMARY_TILE_SIZE,
        help="ROI 내부 타일 크기 (1차 입자는 작게 설정)")
    obj_parser.add_argument(
        "--stride", type=int, default=CONST_PRIMARY_TILE_STRIDE,
        help="타일 stride")
    obj_parser.add_argument(
        "--points_per_tile", type=int, default=CONST_PRIMARY_POINTS_PER_TILE,
        help="각 타일에서 추출할 후보점 수")
    obj_parser.add_argument(
        "--point_min_distance", type=int, default=CONST_PRIMARY_POINT_MIN_DISTANCE,
        help="후보점 최소 거리 (pixel)")
    obj_parser.add_argument(
        "--point_quality_level", type=float, default=CONST_DEFAULT_POINT_QUALITY_LEVEL,
        help="goodFeaturesToTrack qualityLevel")
    obj_parser.add_argument(
        "--point_batch_size", type=int, default=CONST_DEFAULT_POINT_BATCH_SIZE,
        help="한 번의 SAM2 호출에 묶을 point 수")
    obj_parser.add_argument(
        "--dedup_iou", type=float, default=CONST_DEFAULT_DEDUP_IOU,
        help="mask 기준 중복 제거 IoU threshold")
    obj_parser.add_argument(
        "--bbox_dedup_iou", type=float, default=CONST_DEFAULT_BBOX_DEDUP_IOU,
        help="bbox 기준 1차 중복 제거 IoU threshold")
    obj_parser.add_argument(
        "--use_point_prompts",
        action=argparse.BooleanOptionalAction, default=True,
        help="OpenCV 후보점 기반 point prompt 추론 사용 여부")

    # Mask 후처리
    obj_parser.add_argument(
        "--mask_binarize_threshold", type=float, default=CONST_MASK_BINARIZE_THRESHOLD,
        help="SAM2 raw mask → binary mask 변환 threshold")
    obj_parser.add_argument(
        "--min_valid_mask_area", type=int, default=CONST_MIN_VALID_MASK_AREA,
        help="이 값보다 작은 mask 는 무시")
    obj_parser.add_argument(
        "--mask_morph_kernel_size", type=int, default=CONST_MASK_MORPH_KERNEL_SIZE,
        help="morphology kernel 크기. 0/1 이면 비활성화")
    obj_parser.add_argument(
        "--mask_morph_open_iterations", type=int, default=CONST_MASK_MORPH_OPEN_ITERATIONS)
    obj_parser.add_argument(
        "--mask_morph_close_iterations", type=int, default=CONST_MASK_MORPH_CLOSE_ITERATIONS)

    # 기타
    obj_parser.add_argument(
        "--retina_masks",
        action=argparse.BooleanOptionalAction, default=True)
    obj_parser.add_argument(
        "--save_mask_imgs", "--save_individual_masks",
        dest="save_mask_imgs",
        action=argparse.BooleanOptionalAction, default=True,
        help="개별 mask 이미지 저장 여부")
    obj_parser.add_argument(
        "--device", default=None,
        help="추론 device. 예: cpu, cuda:0")

    # ── 핵심 분석 모드 선택 (preset 적용됨) ──────────────────────
    obj_parser.add_argument(
        "--particle_type", default=None,
        choices=["acicular", "plate"],
        help=(
            "1차 입자 형태 (필수 권장). "
            "'acicular': 침상 입자 = 대입경 전구체 특성. "
            "'plate': 판상 입자 = 소입경 전구체 특성. "
            "지정하면 --magnification 과 함께 최적 파라미터가 자동 설정됨."
        ),
    )
    obj_parser.add_argument(
        "--magnification", default=None,
        choices=["20k", "50k"],
        help=(
            "SEM 이미지 배율. "
            "'20k' (20000배): 구형 2차 입자 전체가 화면에 들어옴. "
            "'50k' (50000배): 입자 표면 클로즈업 (위아래 잘림 가능)."
        ),
    )

    # 입자 형태 모드 (preset으로 자동 설정되지만 개별 override 가능)
    obj_parser.add_argument(
        "--particle_mode", default="auto",
        choices=["auto", "acicular"],
        help=(
            "입자 형태 처리 모드. "
            "'auto': 기존 tiled point prompt 방식. "
            "'acicular': OpenCV elongated-contour 탐지 → SAM2 box prompt hybrid 방식 "
            "(침상 입자에 특화; 탐지가 부족하면 자동으로 point prompt fallback)."
        ),
    )

    # 구형 2차 입자 자동 검출
    obj_parser.add_argument(
        "--auto_detect_sphere",
        action=argparse.BooleanOptionalAction, default=False,
        help=(
            "구형 2차 입자를 자동 검출하여 top-cap 영역을 ROI로 사용한다 (acicular 모드 전용). "
            "2차 입자 전체가 보이는 SEM 이미지(예: 5000~20000배)에 유용하다. "
            "검출 실패 시 center crop으로 fallback."
        ),
    )
    obj_parser.add_argument(
        "--sphere_cap_fraction", type=float, default=CONST_SPHERE_CAP_FRACTION,
        help=(
            "구형 2차 입자 검출 시 ROI로 사용할 cap 높이 비율 (0.1~1.0). "
            f"기본값 {CONST_SPHERE_CAP_FRACTION}: 구 지름의 상단 {int(CONST_SPHERE_CAP_FRACTION*100)}%% 영역."
        ),
    )

    # 측정 방법
    obj_parser.add_argument(
        "--measure_mode", default="sam2",
        choices=["sam2", "lsd"],
        help=(
            "측정 방법. "
            "'sam2'(기본): SAM2 segmentation → minAreaRect 두께. "
            "'lsd': LSD(Line Segment Detector) + 수직 강도 프로파일 직접 측정 "
            "(SAM2 불필요, 매우 빠름, 대량 탐지). "
            "acicular 프리셋은 기본으로 'lsd'를 사용."
        ),
    )

    return obj_parser


def main() -> None:
    """CLI 진입점."""
    # 1차 파싱: --particle_type / --magnification 만 먼저 읽어 preset 결정
    obj_preParser = argparse.ArgumentParser(add_help=False)
    obj_preParser.add_argument("--particle_type", default=None)
    obj_preParser.add_argument("--magnification", default=None)
    obj_preArgs, _ = obj_preParser.parse_known_args()

    # 메인 파서 구성
    obj_parser = build_primary_arg_parser()

    # preset 적용: set_defaults 는 CLI에서 명시적으로 지정한 값에는 영향 없음
    if obj_preArgs.particle_type is not None:
        str_mag = obj_preArgs.magnification or "50k"
        dict_preset = get_analysis_preset(obj_preArgs.particle_type, str_mag)
        if dict_preset:
            obj_parser.set_defaults(**dict_preset)
            print(
                f"[preset] {obj_preArgs.particle_type}/{str_mag} 프리셋 적용 "
                f"({len(dict_preset)}개 파라미터)",
                flush=True,
            )
        else:
            print(
                f"[preset] 알 수 없는 조합: {obj_preArgs.particle_type}/{str_mag} "
                "(기본값 사용)",
                flush=True,
            )

    # 2차 파싱: preset 이 기본값으로 설정된 상태에서 모든 인자 파싱
    obj_args = obj_parser.parse_args()

    dict_summary = run_primary_particle_analysis(
        str_input=obj_args.input,
        str_outputDir=obj_args.output_dir,
        str_modelConfig=obj_args.model_cfg,
        str_modelWeights=obj_args.model,
        bool_autoCenterCrop=obj_args.auto_center_crop,
        float_centerCropRatio=obj_args.center_crop_ratio,
        int_roiXMin=obj_args.roi_x_min,
        int_roiYMin=obj_args.roi_y_min,
        int_roiXMax=obj_args.roi_x_max,
        int_roiYMax=obj_args.roi_y_max,
        int_bboxEdgeMargin=obj_args.bbox_edge_margin,
        int_tileEdgeMargin=obj_args.tile_edge_margin,
        float_acicularThreshold=obj_args.acicular_threshold,
        float_particleAreaThreshold=obj_args.area_threshold,
        int_targetParticleCount=obj_args.target_particle_count,
        float_scalePixels=obj_args.scale_pixels,
        float_scaleMicrometers=obj_args.scale_um,
        float_maskBinarizeThreshold=obj_args.mask_binarize_threshold,
        int_minValidMaskArea=obj_args.min_valid_mask_area,
        int_maskMorphKernelSize=obj_args.mask_morph_kernel_size,
        int_maskMorphOpenIterations=obj_args.mask_morph_open_iterations,
        int_maskMorphCloseIterations=obj_args.mask_morph_close_iterations,
        int_imgSize=obj_args.imgsz,
        int_tileSize=obj_args.tile_size,
        int_stride=obj_args.stride,
        int_pointsPerTile=obj_args.points_per_tile,
        int_pointMinDistance=obj_args.point_min_distance,
        float_pointQualityLevel=obj_args.point_quality_level,
        int_pointBatchSize=obj_args.point_batch_size,
        float_dedupIou=obj_args.dedup_iou,
        float_bboxDedupIou=obj_args.bbox_dedup_iou,
        bool_usePointPrompts=obj_args.use_point_prompts,
        str_device=obj_args.device,
        bool_retinaMasks=obj_args.retina_masks,
        bool_saveIndividualMasks=obj_args.save_mask_imgs,
        str_particleMode=obj_args.particle_mode,
        bool_autoDetectSphere=obj_args.auto_detect_sphere,
        float_sphereCapFraction=obj_args.sphere_cap_fraction,
        str_particleType=obj_args.particle_type or "unknown",
        str_magnification=obj_args.magnification or "unknown",
        str_measureMode=obj_args.measure_mode,
    )

    print("===== 1차 입자 분석 결과 요약 =====")
    print(json.dumps(dict_summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    import time
    float_t0 = time.time()
    main()
    print(f"Elapsed time: {time.time() - float_t0:.4f} seconds")
