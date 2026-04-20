#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM2 Aspect Ratio - particle/fragment classification and particle aspect-ratio measurement
=========================================================================================
이 스크립트는 Ultralytics SAM2를 사용해 이미지 내 객체를 분할하고, 면적 기준으로
particle / fragment를 분류합니다.

처리 흐름:
1. 입력 이미지 및 SAM2 모델 경로 로드
2. SAM2 자동 세그멘테이션 수행
3. 각 마스크의 픽셀 면적 계산
4. 면적 기준으로 particle / fragment 분류
5. particle에 대해 가장 긴 가로 span, 가장 긴 세로 span 측정
6. particle aspect ratio(가로 / 세로) 계산
7. fragment 개수 집계 및 결과 저장
=========================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import typing as tp
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import SAM


# =========================================================
# Const / Tunable Hyperparameters
# =========================================================
# `particle` / `fragment` 분류 기준 면적
CONST_PARTICLE_AREA_THRESHOLD: float = 1500.0

# 길이 단위 환산 배율: 276 pixel = 50 um
CONST_SCALE_PIXELS: float = 276.0
CONST_SCALE_MICROMETERS: float = 50.0
CONST_SMALL_PARTICLE_SCALE_PIXELS: float = 184.0
CONST_SMALL_PARTICLE_SCALE_MICROMETERS: float = 10.0
CONST_DEFAULT_SMALL_PARTICLE: bool = False

# ROI 가장자리와 가까운 bbox를 제외하기 위한 margin
CONST_BBOX_EDGE_MARGIN: int = 8
CONST_TILE_EDGE_MARGIN: int = 8

# 실제 SAM2 추론에 사용할 ROI
CONST_ROI_X_MIN: int = 0
CONST_ROI_Y_MIN: int = 0
CONST_ROI_X_MAX: int = 1024
CONST_ROI_Y_MAX: int = 768

# SAM2 raw mask를 binary mask로 바꿀 때 사용하는 threshold
# 현재 Ultralytics 출력에 맞춰 기존 동작과 동일하게 0.0을 기본값으로 둔다.
CONST_MASK_BINARIZE_THRESHOLD: float = 0.0

# 후처리 이후 이 값보다 작은 마스크는 무시
CONST_MIN_VALID_MASK_AREA: int = 1

# 면적 계산 전 binary mask morphology 파라미터
# 0 또는 1이면 morphology를 적용하지 않는다.
CONST_MASK_MORPH_KERNEL_SIZE: int = 0
CONST_MASK_MORPH_OPEN_ITERATIONS: int = 0
CONST_MASK_MORPH_CLOSE_ITERATIONS: int = 0

# 기본 SAM2 추론 파라미터
CONST_DEFAULT_IMAGE_SIZE: int = 1536
CONST_DEFAULT_RETINA_MASKS: bool = True
CONST_DEFAULT_SAVE_INDIVIDUAL_MASKS: bool = True
CONST_DEFAULT_TILE_SIZE: int = 512
CONST_DEFAULT_TILE_STRIDE: int = 256
CONST_DEFAULT_POINTS_PER_TILE: int = 80
CONST_DEFAULT_POINT_MIN_DISTANCE: int = 14
CONST_DEFAULT_POINT_QUALITY_LEVEL: float = 0.03
CONST_DEFAULT_POINT_BATCH_SIZE: int = 32
CONST_DEFAULT_DEDUP_IOU: float = 0.60
CONST_DEFAULT_BBOX_DEDUP_IOU: float = 0.85
CONST_DEFAULT_USE_POINT_PROMPTS: bool = True

CONST_SUPPORTED_IMAGE_SUFFIXES: tp.Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


@dataclass
class Sam2AspectRatioConfig:
    """SAM2 기반 객체 분류 및 종횡비 측정 설정."""

    path_input: Path
    path_outputDir: Path
    path_modelConfig: Path
    path_modelWeights: Path
    int_roiXMin: int = CONST_ROI_X_MIN
    int_roiYMin: int = CONST_ROI_Y_MIN
    int_roiXMax: int = CONST_ROI_X_MAX
    int_roiYMax: int = CONST_ROI_Y_MAX
    int_bboxEdgeMargin: int = CONST_BBOX_EDGE_MARGIN
    int_tileEdgeMargin: int = CONST_TILE_EDGE_MARGIN
    float_particleAreaThreshold: float = CONST_PARTICLE_AREA_THRESHOLD
    float_maskBinarizeThreshold: float = CONST_MASK_BINARIZE_THRESHOLD
    int_minValidMaskArea: int = CONST_MIN_VALID_MASK_AREA
    int_maskMorphKernelSize: int = CONST_MASK_MORPH_KERNEL_SIZE
    int_maskMorphOpenIterations: int = CONST_MASK_MORPH_OPEN_ITERATIONS
    int_maskMorphCloseIterations: int = CONST_MASK_MORPH_CLOSE_ITERATIONS
    int_imgSize: int = CONST_DEFAULT_IMAGE_SIZE
    int_tileSize: int = CONST_DEFAULT_TILE_SIZE
    int_stride: int = CONST_DEFAULT_TILE_STRIDE
    int_pointsPerTile: int = CONST_DEFAULT_POINTS_PER_TILE
    int_pointMinDistance: int = CONST_DEFAULT_POINT_MIN_DISTANCE
    float_pointQualityLevel: float = CONST_DEFAULT_POINT_QUALITY_LEVEL
    int_pointBatchSize: int = CONST_DEFAULT_POINT_BATCH_SIZE
    float_dedupIou: float = CONST_DEFAULT_DEDUP_IOU
    float_bboxDedupIou: float = CONST_DEFAULT_BBOX_DEDUP_IOU
    bool_usePointPrompts: bool = CONST_DEFAULT_USE_POINT_PROMPTS
    bool_smallParticle: bool = CONST_DEFAULT_SMALL_PARTICLE
    float_scalePixels: float = CONST_SCALE_PIXELS
    float_scaleMicrometers: float = CONST_SCALE_MICROMETERS
    str_device: tp.Optional[str] = None
    bool_retinaMasks: bool = CONST_DEFAULT_RETINA_MASKS
    bool_saveIndividualMasks: bool = CONST_DEFAULT_SAVE_INDIVIDUAL_MASKS


@dataclass
class ObjectMeasurement:
    """개별 마스크 측정 결과."""

    int_index: int
    str_category: str
    int_maskArea: int
    float_confidence: tp.Optional[float]
    int_bboxX: int
    int_bboxY: int
    int_bboxWidth: int
    int_bboxHeight: int
    float_bboxWidthUm: float
    float_bboxHeightUm: float
    float_centroidX: float
    float_centroidY: float
    int_longestHorizontal: int
    int_longestVertical: int
    float_longestHorizontalUm: float
    float_longestVerticalUm: float
    float_aspectRatioWH: tp.Optional[float]


@dataclass
class Sam2AspectRatioResult:
    """전체 실행 결과."""

    list_objects: tp.List[ObjectMeasurement]
    dict_summary: tp.Dict[str, tp.Any]


def normalize_image_to_uint8(arr_img: np.ndarray) -> np.ndarray:
    """이미지 배열을 0~255 범위의 `uint8` 배열로 정규화한다.

    Args:
        arr_img: 임의 dtype을 가질 수 있는 입력 이미지 배열. 일반적으로
            grayscale 또는 single-channel 계산 결과를 받는다.

    Returns:
        입력 배열과 동일한 shape의 `np.uint8` 배열. 입력값의 최소/최대값을 기준으로
        선형 정규화되며, 동적 범위가 0에 가까우면 0으로 채워진 배열을 반환한다.
    """
    arr_f32 = arr_img.astype(np.float32)
    float_mn = float(arr_f32.min())
    float_mx = float(arr_f32.max())
    if float_mx - float_mn < 1e-8:
        return np.zeros_like(arr_img, dtype=np.uint8)
    arr_out = (arr_f32 - float_mn) / (float_mx - float_mn)
    return (arr_out * 255.0).clip(0, 255).astype(np.uint8)


def convert_pixels_to_micrometers(
    float_pixels: float,
    float_scalePixels: float = CONST_SCALE_PIXELS,
    float_scaleMicrometers: float = CONST_SCALE_MICROMETERS,
) -> float:
    """픽셀 길이를 마이크로미터 길이로 환산한다.

    Args:
        float_pixels: 변환할 길이 값. 단위는 pixel이다.
        float_scalePixels: 기준 스케일의 pixel 길이. 예를 들어 `276 px = 50 um`
            조건이면 `276.0`이 들어간다.
        float_scaleMicrometers: 기준 스케일의 micrometer 길이.

    Returns:
        `float_pixels`에 대응하는 마이크로미터 길이. `float_scalePixels`가 0 이하이면
        0.0을 반환한다.
    """
    if float_scalePixels <= 0.0:
        return 0.0
    return float(float_pixels * (float_scaleMicrometers / float_scalePixels))


def create_processing_tiles(
    int_x1: int,
    int_y1: int,
    int_x2: int,
    int_y2: int,
    int_tileSize: int,
    int_stride: int,
) -> tp.List[tp.Tuple[int, int, int, int]]:
    """주어진 ROI 사각형을 겹치는 타일 목록으로 분할한다.

    Args:
        int_x1: ROI 시작 x 좌표.
        int_y1: ROI 시작 y 좌표.
        int_x2: ROI 끝 x 좌표. Python slicing과 동일하게 exclusive 성격으로 사용된다.
        int_y2: ROI 끝 y 좌표. Python slicing과 동일하게 exclusive 성격으로 사용된다.
        int_tileSize: 정사각형 타일의 한 변 길이(pixel).
        int_stride: 인접 타일 시작점 간 이동 간격(pixel).

    Returns:
        각 타일의 `(x1, y1, x2, y2)` 좌표 리스트. ROI가 타일보다 작으면 ROI 전체를
        하나의 타일로 반환한다.
    """
    list_tiles = list()
    if int_x2 - int_x1 <= int_tileSize and int_y2 - int_y1 <= int_tileSize:
        list_tiles.append((int_x1, int_y1, int_x2, int_y2))
        return list_tiles

    list_xs = list(
        range(int_x1, max(int_x1 + 1, int_x2 - int_tileSize + 1), int_stride))
    list_ys = list(
        range(int_y1, max(int_y1 + 1, int_y2 - int_tileSize + 1), int_stride))

    if list_xs and list_xs[-1] != int_x2 - int_tileSize:
        list_xs.append(max(int_x1, int_x2 - int_tileSize))
    if list_ys and list_ys[-1] != int_y2 - int_tileSize:
        list_ys.append(max(int_y1, int_y2 - int_tileSize))

    if not list_xs:
        list_xs = [int_x1]
    if not list_ys:
        list_ys = [int_y1]

    for int_yy in list_ys:
        for int_xx in list_xs:
            int_tx1 = int_xx
            int_ty1 = int_yy
            int_tx2 = min(int_xx + int_tileSize, int_x2)
            int_ty2 = min(int_yy + int_tileSize, int_y2)
            list_tiles.append((int_tx1, int_ty1, int_tx2, int_ty2))
    return list_tiles


def enhance_image_texture(arr_tileGray: np.ndarray) -> np.ndarray:
    """후보점 추출을 위해 grayscale 타일의 texture와 edge를 강화한다.

    Args:
        arr_tileGray: 2차원 grayscale 타일 이미지. dtype은 일반적으로 `uint8`이다.

    Returns:
        CLAHE, blur, morphological gradient, blackhat, Laplacian 결과를 결합한
        `uint8` grayscale 이미지. 이후 후보점 탐색의 입력으로 사용된다.
    """
    obj_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    arr_img = obj_clahe.apply(arr_tileGray)
    arr_blur = cv2.GaussianBlur(arr_img, (3, 3), 0)

    arr_kernelGrad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    arr_grad = cv2.morphologyEx(arr_blur, cv2.MORPH_GRADIENT, arr_kernelGrad)

    arr_kernelBh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    arr_blackhat = cv2.morphologyEx(arr_blur, cv2.MORPH_BLACKHAT, arr_kernelBh)

    arr_lap = cv2.Laplacian(arr_blur, cv2.CV_32F, ksize=3)
    arr_lapAbs = normalize_image_to_uint8(np.abs(arr_lap))

    arr_combined = cv2.addWeighted(arr_grad, 0.45, arr_blackhat, 0.35, 0)
    arr_combined = cv2.addWeighted(arr_combined, 0.8, arr_lapAbs, 0.2, 0)
    return normalize_image_to_uint8(arr_combined)


def sample_interest_points(
    arr_tileGray: np.ndarray,
    int_maxPoints: int,
    int_minDistance: int,
    float_qualityLevel: float,
) -> tp.List[tp.Tuple[int, int]]:
    """타일 내부의 SAM2 point prompt 후보 좌표를 추출한다.

    Args:
        arr_tileGray: 후보점을 찾을 grayscale 타일 이미지.
        int_maxPoints: 최종적으로 유지할 최대 후보점 개수.
        int_minDistance: 후보점 사이의 최소 거리(pixel). 너무 가까운 점은 제거된다.
        float_qualityLevel: `cv2.goodFeaturesToTrack`에 전달되는 quality level.

    Returns:
        `(x, y)` 형식의 정수 좌표 리스트. texture 기반 코너 탐지 결과를 우선 사용하고,
        후보가 부족하면 contour centroid를 fallback으로 보강한 뒤 점수와 거리 제약으로
        상위 점만 남긴다.
    """
    arr_enhanced = enhance_image_texture(arr_tileGray)
    arr_corners = cv2.goodFeaturesToTrack(
        arr_enhanced,
        maxCorners=int_maxPoints * 4,
        qualityLevel=float_qualityLevel,
        minDistance=int_minDistance,
        blockSize=5,
        mask=None,
        useHarrisDetector=False,
    )

    list_scoredPoints: tp.List[tp.Tuple[int, int, float]] = list()
    if arr_corners is not None:
        for arr_c in arr_corners[:, 0, :]:
            int_x = int(round(arr_c[0]))
            int_y = int(round(arr_c[1]))
            int_x = int(np.clip(int_x, 0, arr_enhanced.shape[1] - 1))
            int_y = int(np.clip(int_y, 0, arr_enhanced.shape[0] - 1))
            float_score = float(arr_enhanced[int_y, int_x])
            list_scoredPoints.append((int_x, int_y, float_score))

    if len(list_scoredPoints) < max(8, int_maxPoints // 2):
        _, arr_th = cv2.threshold(
            arr_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        arr_kernelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        arr_th = cv2.morphologyEx(arr_th, cv2.MORPH_OPEN, arr_kernelOpen)
        list_cnts, _ = cv2.findContours(
            arr_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for arr_cnt in sorted(list_cnts, key=cv2.contourArea, reverse=True):
            float_area = float(cv2.contourArea(arr_cnt))
            if float_area < 4.0 or float_area > 400.0:
                continue
            dict_m = cv2.moments(arr_cnt)
            if abs(dict_m["m00"]) < 1e-6:
                continue
            int_x = int(round(dict_m["m10"] / dict_m["m00"]))
            int_y = int(round(dict_m["m01"] / dict_m["m00"]))
            int_x = int(np.clip(int_x, 0, arr_enhanced.shape[1] - 1))
            int_y = int(np.clip(int_y, 0, arr_enhanced.shape[0] - 1))
            float_score = float(arr_enhanced[int_y, int_x]) + float_area
            list_scoredPoints.append((int_x, int_y, float_score))
            if len(list_scoredPoints) >= int_maxPoints * 3:
                break

    list_scoredPoints.sort(key=lambda tpl_item: tpl_item[2], reverse=True)

    list_kept = list()
    for int_px, int_py, _ in list_scoredPoints:
        bool_tooClose = False
        for int_qx, int_qy in list_kept:
            if (int_px - int_qx) ** 2 + (int_py - int_qy) ** 2 < int_minDistance ** 2:
                bool_tooClose = True
                break
        if not bool_tooClose:
            list_kept.append((int_px, int_py))
        if len(list_kept) >= int_maxPoints:
            break
    return list_kept


def calculate_binary_iou(arr_maskA: np.ndarray, arr_maskB: np.ndarray) -> float:
    """두 이진 마스크의 IoU(Intersection over Union)를 계산한다.

    Args:
        arr_maskA: 첫 번째 binary mask. 0은 배경, 0보다 큰 값은 foreground로 간주한다.
        arr_maskB: 두 번째 binary mask. shape은 `arr_maskA`와 같아야 한다.

    Returns:
        두 마스크의 IoU 값. union이 0이면 0.0을 반환한다.
    """
    int_inter = np.logical_and(arr_maskA > 0, arr_maskB > 0).sum()
    int_union = np.logical_or(arr_maskA > 0, arr_maskB > 0).sum()
    if int_union == 0:
        return 0.0
    return float(int_inter / int_union)


def calculate_box_iou(
    tuple_boxA: tp.Tuple[int, int, int, int],
    tuple_boxB: tp.Tuple[int, int, int, int],
) -> float:
    """두 bounding box의 IoU를 계산한다.

    Args:
        tuple_boxA: `(x, y, w, h)` 형식의 첫 번째 bbox.
        tuple_boxB: `(x, y, w, h)` 형식의 두 번째 bbox.

    Returns:
        두 bbox의 IoU 값. union 면적이 0 이하이면 0.0을 반환한다.
    """
    int_ax1, int_ay1, int_aw, int_ah = tuple_boxA
    int_bx1, int_by1, int_bw, int_bh = tuple_boxB
    int_ax2 = int_ax1 + int_aw
    int_ay2 = int_ay1 + int_ah
    int_bx2 = int_bx1 + int_bw
    int_by2 = int_by1 + int_bh

    int_ix1 = max(int_ax1, int_bx1)
    int_iy1 = max(int_ay1, int_by1)
    int_ix2 = min(int_ax2, int_bx2)
    int_iy2 = min(int_ay2, int_by2)

    int_interW = max(0, int_ix2 - int_ix1)
    int_interH = max(0, int_iy2 - int_iy1)
    int_inter = int_interW * int_interH
    int_union = int_aw * int_ah + int_bw * int_bh - int_inter
    if int_union <= 0:
        return 0.0
    return float(int_inter / int_union)


def iter_chunks(
    list_items: tp.Sequence[tp.Tuple[int, int]],
    int_chunkSize: int,
) -> tp.Iterable[tp.Sequence[tp.Tuple[int, int]]]:
    """시퀀스를 고정 크기 chunk 단위로 순회한다.

    Args:
        list_items: `(x, y)` 좌표 시퀀스.
        int_chunkSize: chunk 하나에 포함할 최대 원소 수.

    Yields:
        원본 시퀀스의 연속 구간을 담는 부분 시퀀스. `int_chunkSize`가 1 이하이면
        1로 보정된다.
    """
    int_chunkSize = max(1, int_chunkSize)
    for int_idx in range(0, len(list_items), int_chunkSize):
        yield list_items[int_idx:int_idx + int_chunkSize]


def load_particle_mean_sizes_from_csv(path_particlesCsv: Path) -> np.ndarray:
    """`particles.csv`에서 particle 평균 크기(um)를 읽어온다.

    Args:
        path_particlesCsv: `particles.csv` 파일 경로. UTF-8 BOM(`utf-8-sig`)로 읽는다.

    Returns:
        각 row의 `(float_longestHorizontalUm + float_longestVerticalUm) / 2` 값을 담은
        `np.float32` 1차원 배열. 파일이 없거나 유효한 row가 없으면 빈 배열을 반환한다.
    """
    if not path_particlesCsv.exists():
        return np.array([], dtype=np.float32)

    list_meanSizes: tp.List[float] = []
    with path_particlesCsv.open("r", newline="", encoding="utf-8-sig") as obj_f:
        obj_reader = csv.DictReader(obj_f)
        for dict_row in obj_reader:
            try:
                float_horizontal = float(dict_row["float_longestHorizontalUm"])
                float_vertical = float(dict_row["float_longestVerticalUm"])
            except (KeyError, TypeError, ValueError):
                continue
            list_meanSizes.append((float_horizontal + float_vertical) / 2.0)

    if not list_meanSizes:
        return np.array([], dtype=np.float32)
    return np.array(list_meanSizes, dtype=np.float32)


def get_lot_number_from_input_path(path_inputImage: Path) -> str:
    """입력 이미지 경로에서 lot 번호를 추출한다.

    Args:
        path_inputImage: 원본 입력 이미지 경로.

    Returns:
        절대경로 기준 이미지 파일의 바로 위 directory 이름. 비어 있으면
        `"UnknownLot"`을 반환한다.
    """
    try:
        path_resolved = path_inputImage.resolve()
    except OSError:
        path_resolved = path_inputImage
    str_lotNumber = path_resolved.parent.name.strip()
    return str_lotNumber if str_lotNumber else "UnknownLot"


def save_particle_distribution_histogram(
    path_particlesCsv: Path,
    path_outputImage: Path,
    path_inputImage: Path,
) -> None:
    """particle 크기 분포 histogram을 `png` 파일로 저장한다.

    Args:
        path_particlesCsv: particle 측정 결과 CSV 경로. 평균 크기 계산의 데이터 소스다.
        path_outputImage: 저장할 histogram 이미지 경로. 일반적으로 `particle_dist.png`.
        path_inputImage: lot 번호 추출에 사용할 원본 이미지 경로.

    Returns:
        없음. `matplotlib`의 headless backend(`Agg`)를 사용해 histogram 이미지를
        디스크에 저장한다.

    Notes:
        - x축 값은 micrometer 단위 particle 평균 크기다.
        - 제목은 입력 이미지의 부모 directory 이름(lot 번호)이다.
        - 평균 크기 위치에는 빨간 vertical line과 빨간 텍스트를 함께 표시한다.
        - particle 데이터가 없으면 빈 축에 안내 문구만 그린다.
    """
    arr_meanSizes = load_particle_mean_sizes_from_csv(path_particlesCsv)
    str_lotNumber = get_lot_number_from_input_path(path_inputImage)
    obj_fig, obj_ax = plt.subplots(figsize=(9.6, 6.4), dpi=100)

    try:
        obj_ax.set_title(str_lotNumber, fontsize=28)
        obj_ax.set_ylabel("Count", fontsize=20)
        obj_ax.set_xlabel("Mean of longest horizontal and vertical length (um)", fontsize=20)
        obj_ax.tick_params(axis="both", labelsize=20)

        if arr_meanSizes.size == 0:
            obj_ax.text(
                0.5,
                0.5,
                "No particle data in particles.csv",
                ha="center",
                va="center",
                fontsize=13,
                color="#666666",
                transform=obj_ax.transAxes,
            )
            obj_ax.set_xticks([])
            obj_ax.set_yticks([])
        else:
            int_numBins = int(np.clip(np.sqrt(arr_meanSizes.size), 5, 20))
            float_minValue = float(np.min(arr_meanSizes))
            float_maxValue = float(np.max(arr_meanSizes))
            float_meanValue = float(np.mean(arr_meanSizes))
            if abs(float_maxValue - float_minValue) < 1e-6:
                float_minValue -= 0.5
                float_maxValue += 0.5

            obj_ax.hist(
                arr_meanSizes,
                bins=int_numBins,
                range=(float_minValue, float_maxValue),
                color="#508cf0",
                edgecolor="#323232",
                linewidth=1.0,
            )

            obj_ax.axvline(float_meanValue, color="red", linewidth=2.0)
            float_yMax = obj_ax.get_ylim()[1]
            obj_ax.text(
                float_meanValue,
                float_yMax * 0.96,
                f"Mean: {float_meanValue:.2f} um",
                color="red",
                fontsize=24,
                ha="left",
                va="top",
            )
            obj_ax.grid(axis="y", linestyle="--", alpha=0.25)

        obj_fig.tight_layout()
        obj_fig.savefig(path_outputImage, bbox_inches="tight")
    finally:
        plt.close(obj_fig)


class Sam2AspectRatioService:
    """SAM2 추론, 후처리, 결과 저장을 담당하는 서비스 클래스.

    Attributes:
        obj_config: 전체 파이프라인 설정을 담는 `Sam2AspectRatioConfig`.
        obj_model: 초기화 이후의 Ultralytics `SAM` 모델 인스턴스. 초기에는 `None`.
        dict_modelConfig: YAML 또는 대체 파싱 결과를 담는 메타데이터 dict.
    """

    def __init__(self, obj_config: Sam2AspectRatioConfig) -> None:
        """서비스 객체를 생성한다.

        Args:
            obj_config: 경로, 추론 파라미터, 후처리 파라미터를 포함한 설정 객체.
        """
        self.obj_config = obj_config
        self.obj_model: tp.Optional[SAM] = None
        self.dict_modelConfig: tp.Dict[str, tp.Any] = dict()

    def validate_inputs(self) -> None:
        """필수 입력 경로들의 존재 여부를 검증한다.

        Raises:
            FileNotFoundError: 입력 이미지, 모델 설정 파일, 가중치 파일 중 하나라도
                존재하지 않을 때 발생한다.
        """
        list_requiredPaths = [
            self.obj_config.path_input,
            self.obj_config.path_modelConfig,
            self.obj_config.path_modelWeights,
        ]
        for path_item in list_requiredPaths:
            if not path_item.exists():
                raise FileNotFoundError(f"필수 경로를 찾을 수 없습니다: {path_item}")

    def load_model_config(self) -> None:
        """모델 설정 파일을 읽어 결과 메타데이터용 dict로 정리한다.

        Returns:
            없음. 파싱 결과는 `self.dict_modelConfig`에 저장된다.

        Notes:
            설정 파일이 정상 YAML dict이면 그대로 저장한다. YAML이 아니거나 HTML이
            들어있으면 파싱 상태와 일부 preview만 메타데이터로 남긴다.
        """
        str_rawText = self.obj_config.path_modelConfig.read_text(
            encoding="utf-8", errors="ignore")

        try:
            obj_loaded = yaml.safe_load(str_rawText)
        except yaml.YAMLError:
            obj_loaded = None

        if isinstance(obj_loaded, dict):
            self.dict_modelConfig = obj_loaded
            self.dict_modelConfig.setdefault("config_parse_status", "parsed")
            return

        str_parseStatus = "unparsed"
        if "<!DOCTYPE html>" in str_rawText[:256] or "<html" in str_rawText[:256].lower():
            str_parseStatus = "html_instead_of_yaml"

        self.dict_modelConfig = {
            "config_parse_status": str_parseStatus,
            "config_preview": str_rawText[:200].strip(),
        }

    def resolve_model_weights_path(self) -> Path:
        """
        Ultralytics가 요구하는 SAM2 파일명 alias로 가중치 경로를 정규화.

        일부 체크포인트는 파일 내용은 정상이어도 파일명 규칙이 다르면 Ultralytics가
        지원 모델로 인식하지 못하므로, 필요한 경우 alias 파일을 생성한다.

        Returns:
            Ultralytics가 인식 가능한 파일명으로 정규화된 weight 파일 경로.

        Raises:
            FileNotFoundError: 현재 코드가 지원하지 않는 weight 파일명일 때 발생한다.
        """
        path_weights = self.obj_config.path_modelWeights
        set_supportedNames = {
            "sam_h.pt",
            "sam_l.pt",
            "sam_b.pt",
            "mobile_sam.pt",
            "sam2_t.pt",
            "sam2_s.pt",
            "sam2_b.pt",
            "sam2_l.pt",
            "sam2.1_t.pt",
            "sam2.1_s.pt",
            "sam2.1_b.pt",
            "sam2.1_l.pt",
        }
        dict_aliasNames = {
            "sam2_hiera_tiny.pt": "sam2_t.pt",
            "sam2_hiera_small.pt": "sam2_s.pt",
            "sam2_hiera_base_plus.pt": "sam2_b.pt",
            "sam2_hiera_large.pt": "sam2_l.pt",
            "sam2.1_hiera_tiny.pt": "sam2.1_t.pt",
            "sam2.1_hiera_small.pt": "sam2.1_s.pt",
            "sam2.1_hiera_base_plus.pt": "sam2.1_b.pt",
            "sam2.1_hiera_large.pt": "sam2.1_l.pt",
        }

        if path_weights.name in set_supportedNames:
            return path_weights

        str_aliasName = dict_aliasNames.get(path_weights.name)
        if str_aliasName is None:
            raise FileNotFoundError(
                f"{path_weights} 는 현재 ultralytics가 인식하는 SAM2 체크포인트 이름이 아닙니다."
            )

        path_aliasDir = self.obj_config.path_outputDir / "_model_alias"
        path_aliasDir.mkdir(parents=True, exist_ok=True)
        path_alias = path_aliasDir / str_aliasName

        if path_alias.exists() and path_alias.stat().st_size == path_weights.stat().st_size:
            return path_alias

        if path_alias.exists():
            path_alias.unlink()

        try:
            os.link(path_weights, path_alias)
        except OSError:
            shutil.copy2(path_weights, path_alias)

        return path_alias

    def initialize_model(self) -> None:
        """입력 검증과 설정 로드를 거쳐 SAM2 모델을 초기화한다.

        Returns:
            없음. 초기화된 모델은 `self.obj_model`에 저장된다.
        """
        self.validate_inputs()
        self.load_model_config()
        path_resolvedWeights = self.resolve_model_weights_path()
        self.obj_model = SAM(str(path_resolvedWeights))

    def load_image_bgr(self) -> np.ndarray:
        """입력 이미지를 OpenCV BGR 형식으로 로드한다.

        Returns:
            shape `(H, W, 3)`의 BGR `np.ndarray`.

        Raises:
            FileNotFoundError: 이미지를 읽을 수 없을 때 발생한다.
        """
        arr_image = cv2.imread(
            str(self.obj_config.path_input), cv2.IMREAD_COLOR)
        if arr_image is None:
            raise FileNotFoundError(
                f"이미지를 읽을 수 없습니다: {self.obj_config.path_input}")
        return arr_image

    def extract_inference_roi(
        self,
        arr_imageBgr: np.ndarray,
    ) -> tp.Tuple[np.ndarray, tp.Dict[str, int]]:
        """전체 이미지에서 실제 추론 대상 ROI를 crop한다.

        Args:
            arr_imageBgr: 원본 BGR 이미지 배열.

        Returns:
            `(arr_roiBgr, dict_roi)` 튜플.
            - `arr_roiBgr`: ROI 영역만 잘라낸 BGR 이미지.
            - `dict_roi`: `x_min`, `y_min`, `x_max`, `y_max`, `width`, `height`
              키를 가지는 ROI 메타데이터 dict.

        Raises:
            ValueError: ROI 설정이 이미지 범위와 교차하지 않아 유효한 crop을 만들 수
                없을 때 발생한다.
        """
        int_h, int_w = arr_imageBgr.shape[:2]

        int_x0 = max(0, min(self.obj_config.int_roiXMin, int_w))
        int_y0 = max(0, min(self.obj_config.int_roiYMin, int_h))
        int_x1 = max(int_x0, min(self.obj_config.int_roiXMax, int_w))
        int_y1 = max(int_y0, min(self.obj_config.int_roiYMax, int_h))

        if int_x1 <= int_x0 or int_y1 <= int_y0:
            raise ValueError(
                "유효한 ROI를 만들 수 없습니다. ROI 좌표와 입력 이미지 크기를 확인하세요."
            )

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

    def predict_tiled_point_prompts(
        self,
        arr_inputBgr: np.ndarray,
    ) -> tp.Tuple[np.ndarray, tp.Optional[np.ndarray], tp.Dict[str, tp.Any]]:
        """ROI 이미지를 타일 단위로 SAM2 추론한다.

        Args:
            arr_inputBgr: ROI crop 이후의 BGR 이미지. shape은 일반적으로 `(H, W, 3)`.

        Returns:
            `(arr_masks, arr_scores, dict_debug)` 튜플.
            - `arr_masks`: ROI 좌표계 기준 binary mask 배열. shape은
              `(N, H, W)`이며, mask가 없으면 빈 배열이다.
            - `arr_scores`: 각 mask의 confidence score 배열. score가 없으면 `None`.
            - `dict_debug`: tile 수, 후보점 수, 중복 제거 수, 각 tile/point 정보를 담는
              디버그 dict.

        Notes:
            `bool_usePointPrompts=True`이면 OpenCV 후보점을 추출해 batch point prompt로
            SAM2를 호출하고, `False`이면 tile 전체에 대해 자동 분할을 수행한다.
        """
        if self.obj_model is None:
            self.initialize_model()

        int_roiHeight, int_roiWidth = arr_inputBgr.shape[:2]
        arr_inputGray = cv2.cvtColor(arr_inputBgr, cv2.COLOR_BGR2GRAY)
        list_tiles = create_processing_tiles(
            0,
            0,
            int_roiWidth,
            int_roiHeight,
            int_tileSize=self.obj_config.int_tileSize,
            int_stride=self.obj_config.int_stride,
        )

        dict_predictCommon: tp.Dict[str, tp.Any] = {
            "imgsz": self.obj_config.int_imgSize,
            "retina_masks": self.obj_config.bool_retinaMasks,
            "verbose": False,
        }
        if self.obj_config.str_device:
            dict_predictCommon["device"] = self.obj_config.str_device

        list_keptMasks = list()
        list_keptScores = list()
        list_keptBboxes: tp.List[tp.Tuple[int, int, int, int]] = list()
        list_debugTiles = list()
        list_debugPoints = list()
        int_candidateCount = 0
        int_acceptedCount = 0
        int_bboxDedupRejected = 0

        for int_tileIdx, (int_tx1, int_ty1, int_tx2, int_ty2) in enumerate(list_tiles):
            arr_tileBgr = arr_inputBgr[int_ty1:int_ty2, int_tx1:int_tx2].copy()
            list_promptBatches: tp.List[tp.Optional[tp.Sequence[tp.Tuple[int, int]]]] = [
                None]
            list_points: tp.List[tp.Tuple[int, int]] = []

            if self.obj_config.bool_usePointPrompts:
                arr_tileGray = arr_inputGray[int_ty1:int_ty2,
                                             int_tx1:int_tx2].copy()
                list_points = sample_interest_points(
                    arr_tileGray=arr_tileGray,
                    int_maxPoints=self.obj_config.int_pointsPerTile,
                    int_minDistance=self.obj_config.int_pointMinDistance,
                    float_qualityLevel=self.obj_config.float_pointQualityLevel,
                )
                list_promptBatches = list(iter_chunks(
                    list_points, self.obj_config.int_pointBatchSize))

                for int_px, int_py in list_points:
                    int_candidateCount += 1
                    list_debugPoints.append(
                        {
                            "tile_index": int_tileIdx,
                            "tile_xyxy": [int_tx1, int_ty1, int_tx2, int_ty2],
                            "point_xy_tile": [int_px, int_py],
                            "point_xy_roi": [int_tx1 + int_px, int_ty1 + int_py],
                        }
                    )

            list_debugTiles.append(
                {
                    "tile_index": int_tileIdx,
                    "tile_xyxy": [int_tx1, int_ty1, int_tx2, int_ty2],
                    "num_points": len(list_points),
                    "use_point_prompts": bool(self.obj_config.bool_usePointPrompts),
                }
            )

            for list_pointChunk in list_promptBatches:
                if self.obj_config.bool_usePointPrompts:
                    if not list_pointChunk:
                        continue
                    list_results = self.obj_model(  # type: ignore[misc]
                        source=arr_tileBgr,
                        points=[[int_px, int_py]
                                for int_px, int_py in list_pointChunk],
                        labels=[1] * len(list_pointChunk),
                        **dict_predictCommon,
                    )
                else:
                    list_results = self.obj_model(  # type: ignore[misc]
                        source=arr_tileBgr,
                        **dict_predictCommon,
                    )
                if not list_results:
                    continue

                obj_result = list_results[0]
                if obj_result.masks is None or obj_result.masks.data is None:
                    continue

                arr_tileMasks = obj_result.masks.data.detach().cpu().numpy()
                arr_tileScores = None
                if obj_result.boxes is not None and obj_result.boxes.conf is not None:
                    arr_tileScores = obj_result.boxes.conf.detach().cpu().numpy()

                for int_maskIdx, arr_tm in enumerate(arr_tileMasks):
                    arr_tileMask = (
                        arr_tm > self.obj_config.float_maskBinarizeThreshold).astype(np.uint8)
                    if int(arr_tileMask.sum()) < self.obj_config.int_minValidMaskArea:
                        continue

                    arr_tileContour = self.extract_largest_contour(
                        arr_tileMask)
                    if arr_tileContour is None:
                        continue
                    int_bx, int_by, int_bw, int_bh = cv2.boundingRect(
                        arr_tileContour)
                    int_tileHeight, int_tileWidth = arr_tileMask.shape[:2]
                    if self.is_bbox_near_edge(
                        int_x=int_bx,
                        int_y=int_by,
                        int_w=int_bw,
                        int_h=int_bh,
                        int_width=int_tileWidth,
                        int_height=int_tileHeight,
                        int_margin=self.obj_config.int_tileEdgeMargin,
                    ):
                        continue

                    tuple_globalBox = (
                        int_tx1 + int_bx,
                        int_ty1 + int_by,
                        int_bw,
                        int_bh,
                    )
                    bool_bboxDup = False
                    for tuple_prevBox in list_keptBboxes:
                        if calculate_box_iou(tuple_prevBox, tuple_globalBox) >= self.obj_config.float_bboxDedupIou:
                            bool_bboxDup = True
                            break
                    if bool_bboxDup:
                        int_bboxDedupRejected += 1
                        continue

                    arr_roiMask = np.zeros(
                        (int_roiHeight, int_roiWidth), dtype=np.uint8)
                    arr_roiMask[int_ty1:int_ty2,
                                int_tx1:int_tx2] = arr_tileMask

                    bool_isDup = False
                    for arr_prevMask in list_keptMasks:
                        if calculate_binary_iou(arr_prevMask, arr_roiMask) >= self.obj_config.float_dedupIou:
                            bool_isDup = True
                            break
                    if bool_isDup:
                        continue

                    int_acceptedCount += 1
                    list_keptMasks.append(arr_roiMask)
                    list_keptBboxes.append(tuple_globalBox)
                    if arr_tileScores is not None and int_maskIdx < len(arr_tileScores):
                        list_keptScores.append(
                            float(arr_tileScores[int_maskIdx]))
                    else:
                        list_keptScores.append(None)

        arr_masks = (
            np.stack(list_keptMasks, axis=0).astype(np.uint8)
            if list_keptMasks
            else np.empty((0, int_roiHeight, int_roiWidth), dtype=np.uint8)
        )
        arr_scores = None
        if list_keptScores:
            arr_scores = np.array(
                [np.nan if x is None else float(x) for x in list_keptScores],
                dtype=np.float32,
            )
        dict_debug = {
            "num_tiles": len(list_tiles),
            "num_candidate_points": int_candidateCount,
            "num_accepted_masks": int_acceptedCount,
            "num_bbox_dedup_rejected": int_bboxDedupRejected,
            "tiles": list_debugTiles,
            "candidate_points": list_debugPoints,
        }
        return arr_masks, arr_scores, dict_debug

    def refine_mask_for_area(self, arr_mask: np.ndarray) -> np.ndarray:
        """
        면적 계산 전 binary mask를 후처리한다.

        Args:
            arr_mask: 입력 binary 또는 binary-like mask. 0 초과값을 foreground로 본다.

        Returns:
            morphology가 적용된 `uint8` binary mask.

        Notes:
            area threshold 자체뿐 아니라 이 함수의 morphology 설정도 최종 area 값에
            직접 영향을 준다.
        """
        arr_maskUint8 = (arr_mask > 0).astype(np.uint8)

        int_kernelSize = self.obj_config.int_maskMorphKernelSize
        if int_kernelSize <= 1:
            return arr_maskUint8

        arr_kernel = np.ones((int_kernelSize, int_kernelSize), dtype=np.uint8)
        arr_refined = arr_maskUint8

        if self.obj_config.int_maskMorphOpenIterations > 0:
            arr_refined = cv2.morphologyEx(
                arr_refined,
                cv2.MORPH_OPEN,
                arr_kernel,
                iterations=self.obj_config.int_maskMorphOpenIterations,
            )

        if self.obj_config.int_maskMorphCloseIterations > 0:
            arr_refined = cv2.morphologyEx(
                arr_refined,
                cv2.MORPH_CLOSE,
                arr_kernel,
                iterations=self.obj_config.int_maskMorphCloseIterations,
            )

        return arr_refined

    @staticmethod
    def get_longest_span(arr_mask: np.ndarray, bool_horizontal: bool) -> int:
        """마스크 내부의 가장 긴 가로/세로 span 길이를 계산한다.

        Args:
            arr_mask: 2차원 binary mask.
            bool_horizontal: `True`이면 가로 방향 span, `False`이면 세로 방향 span을
                계산한다.

        Returns:
            foreground 픽셀이 존재하는 한 줄에서의 최대 span 길이(pixel).
        """
        arr_scan = arr_mask if bool_horizontal else arr_mask.T
        int_longest = 0
        for arr_line in arr_scan:
            arr_indices = np.flatnonzero(arr_line)
            if arr_indices.size == 0:
                continue
            int_span = int(arr_indices[-1] - arr_indices[0] + 1)
            if int_span > int_longest:
                int_longest = int_span
        return int_longest

    @staticmethod
    def extract_largest_contour(arr_mask: np.ndarray) -> tp.Optional[np.ndarray]:
        """외곽 contour 중 면적이 가장 큰 contour를 반환한다.

        Args:
            arr_mask: contour를 찾을 binary mask.

        Returns:
            가장 큰 contour의 OpenCV contour 배열. contour가 없으면 `None`.
        """
        list_contours, _ = cv2.findContours(
            arr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not list_contours:
            return None
        return max(list_contours, key=cv2.contourArea)

    @staticmethod
    def is_bbox_near_edge(
        int_x: int,
        int_y: int,
        int_w: int,
        int_h: int,
        int_width: int,
        int_height: int,
        int_margin: int,
    ) -> bool:
        """bbox가 영역 경계에 너무 가까운지 판정한다.

        Args:
            int_x: bbox 좌상단 x.
            int_y: bbox 좌상단 y.
            int_w: bbox width.
            int_h: bbox height.
            int_width: bbox가 놓인 기준 영역의 전체 width.
            int_height: bbox가 놓인 기준 영역의 전체 height.
            int_margin: 경계와의 최소 허용 거리.

        Returns:
            bbox의 어느 한 변이라도 기준 영역 경계에서 `int_margin` 이내면 `True`.
        """
        int_margin = max(0, int_margin)
        int_right = int_x + int_w
        int_bottom = int_y + int_h
        return (
            int_x <= int_margin
            or int_y <= int_margin
            or int_right >= (int_width - int_margin)
            or int_bottom >= (int_height - int_margin)
        )

    def is_bbox_near_roi_edge(
        self,
        int_x: int,
        int_y: int,
        int_w: int,
        int_h: int,
        int_roiWidth: int,
        int_roiHeight: int,
    ) -> bool:
        """bbox가 ROI 경계에 너무 가까운지 판정한다.

        Args:
            int_x: ROI 좌표계 기준 bbox x.
            int_y: ROI 좌표계 기준 bbox y.
            int_w: bbox width.
            int_h: bbox height.
            int_roiWidth: ROI 전체 width.
            int_roiHeight: ROI 전체 height.

        Returns:
            ROI 경계와의 거리가 `obj_config.int_bboxEdgeMargin` 이내이면 `True`.
        """
        return self.is_bbox_near_edge(
            int_x=int_x,
            int_y=int_y,
            int_w=int_w,
            int_h=int_h,
            int_width=int_roiWidth,
            int_height=int_roiHeight,
            int_margin=self.obj_config.int_bboxEdgeMargin,
        )

    def convert_pixels_to_micrometers(self, float_pixels: float) -> float:
        """현재 config의 scale 기준으로 픽셀 길이를 um로 환산한다.

        Args:
            float_pixels: pixel 단위 길이 값.

        Returns:
            현재 설정된 scale(`float_scalePixels`, `float_scaleMicrometers`) 기준의
            micrometer 길이.
        """
        return convert_pixels_to_micrometers(
            float_pixels=float_pixels,
            float_scalePixels=self.obj_config.float_scalePixels,
            float_scaleMicrometers=self.obj_config.float_scaleMicrometers,
        )

    def measure_mask(
        self,
        arr_mask: np.ndarray,
        int_index: int,
        float_confidence: tp.Optional[float],
    ) -> tp.Optional[ObjectMeasurement]:
        """단일 mask의 측정값을 계산해 `ObjectMeasurement`로 변환한다.

        Args:
            arr_mask: ROI 좌표계 기준 binary mask.
            int_index: 현재 mask의 인덱스. 결과 식별용으로 그대로 저장된다.
            float_confidence: SAM2가 제공한 confidence score. 없으면 `None`.

        Returns:
            유효한 객체이면 `ObjectMeasurement`, 너무 작거나 contour가 없거나 ROI 경계에
            너무 가까우면 `None`.
        """
        arr_refinedMask = self.refine_mask_for_area(arr_mask)
        int_maskArea = int(arr_refinedMask.sum())
        if int_maskArea < self.obj_config.int_minValidMaskArea:
            return None

        arr_contour = self.extract_largest_contour(arr_refinedMask)
        if arr_contour is None:
            return None

        int_x, int_y, int_w, int_h = cv2.boundingRect(arr_contour)
        int_roiHeight, int_roiWidth = arr_refinedMask.shape[:2]
        if self.is_bbox_near_roi_edge(int_x, int_y, int_w, int_h, int_roiWidth, int_roiHeight):
            return None

        obj_moments = cv2.moments(arr_contour)
        if obj_moments["m00"] > 0.0:
            float_cx = float(obj_moments["m10"] / obj_moments["m00"])
            float_cy = float(obj_moments["m01"] / obj_moments["m00"])
        else:
            float_cx = float(int_x + int_w / 2.0)
            float_cy = float(int_y + int_h / 2.0)

        # bbox가 실제 mask 외접 사각형이므로, 측정 span은 bbox 크기를 넘지 않도록 제한한다.
        int_horizontal = min(
            self.get_longest_span(arr_refinedMask, bool_horizontal=True),
            int_w,
        )
        int_vertical = min(
            self.get_longest_span(arr_refinedMask, bool_horizontal=False),
            int_h,
        )

        str_category = (
            "particle"
            if int_maskArea >= int(round(self.obj_config.float_particleAreaThreshold))
            else "fragment"
        )

        float_aspectRatio = None
        if str_category == "particle":
            int_longAxis = max(int_horizontal, int_vertical, 1)
            int_shortAxis = max(1, min(int_horizontal, int_vertical))
            float_aspectRatio = float(int_shortAxis / int_longAxis)

        return ObjectMeasurement(
            int_index=int_index,
            str_category=str_category,
            int_maskArea=int_maskArea,
            float_confidence=float_confidence,
            int_bboxX=int(int_x),
            int_bboxY=int(int_y),
            int_bboxWidth=int(int_w),
            int_bboxHeight=int(int_h),
            float_bboxWidthUm=self.convert_pixels_to_micrometers(float(int_w)),
            float_bboxHeightUm=self.convert_pixels_to_micrometers(float(int_h)),
            float_centroidX=float_cx,
            float_centroidY=float_cy,
            int_longestHorizontal=int_horizontal,
            int_longestVertical=int_vertical,
            float_longestHorizontalUm=self.convert_pixels_to_micrometers(float(int_horizontal)),
            float_longestVerticalUm=self.convert_pixels_to_micrometers(float(int_vertical)),
            float_aspectRatioWH=float_aspectRatio,
        )

    def create_overlay(
        self,
        arr_imageBgr: np.ndarray,
        list_objects: tp.List[ObjectMeasurement],
        list_masks: tp.List[np.ndarray],
    ) -> np.ndarray:
        """객체 마스크와 라벨을 원본 ROI 이미지 위에 시각화한다.

        Args:
            arr_imageBgr: ROI 이미지.
            list_objects: 시각화할 객체 측정 결과 리스트.
            list_masks: 각 객체에 대응하는 ROI 좌표계 binary mask 리스트.

        Returns:
            mask overlay, contour, bbox, 텍스트 라벨이 그려진 BGR 이미지.
        """
        arr_overlay = arr_imageBgr.copy()

        for obj_measurement, arr_mask in zip(list_objects, list_masks):
            tpl_color = (60, 220, 60) if obj_measurement.str_category == "particle" else (
                0, 165, 255)
            arr_overlay[arr_mask > 0] = (
                arr_overlay[arr_mask > 0].astype(np.float32) * 0.55
                + np.array(tpl_color, dtype=np.float32) * 0.45
            ).astype(np.uint8)

        for obj_measurement, arr_mask in zip(list_objects, list_masks):
            arr_contour = self.extract_largest_contour(arr_mask)
            if arr_contour is None:
                continue

            tpl_color = (0, 255, 0) if obj_measurement.str_category == "particle" else (
                0, 140, 255)
            cv2.drawContours(arr_overlay, [arr_contour], -1, tpl_color, 1)
            cv2.rectangle(
                arr_overlay,
                (obj_measurement.int_bboxX, obj_measurement.int_bboxY),
                (
                    obj_measurement.int_bboxX + obj_measurement.int_bboxWidth,
                    obj_measurement.int_bboxY + obj_measurement.int_bboxHeight,
                ),
                tpl_color,
                1,
            )

            int_labelX = obj_measurement.int_bboxX
            int_labelY = max(14, obj_measurement.int_bboxY - 4)
            if obj_measurement.str_category == "particle" and obj_measurement.float_aspectRatioWH is not None:
                str_label = (
                    f"P{obj_measurement.int_index} "
                    f"A={obj_measurement.int_maskArea} "
                    f"AR={obj_measurement.float_aspectRatioWH:.2f}"
                )
            else:
                str_label = f"F{obj_measurement.int_index} A={obj_measurement.int_maskArea}"

            cv2.putText(
                arr_overlay,
                str_label,
                (int_labelX, int_labelY),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                tpl_color,
                1,
                cv2.LINE_AA,
            )

        return arr_overlay

    def build_summary(self, list_objects: tp.List[ObjectMeasurement]) -> tp.Dict[str, tp.Any]:
        """단일 이미지 처리 결과에 대한 요약 통계를 생성한다.

        Args:
            list_objects: 유효성 검사를 통과한 객체 측정 결과 리스트.

        Returns:
            JSON 저장에 바로 사용할 수 있는 요약 dict. 설정값, 집계 개수, 비율,
            종횡비 통계가 포함된다.
        """
        list_particles = [
            obj_item for obj_item in list_objects if obj_item.str_category == "particle"]
        list_fragments = [
            obj_item for obj_item in list_objects if obj_item.str_category == "fragment"]
        int_totalObjects = len(list_objects)
        int_particleCount = len(list_particles)
        int_fragmentCount = len(list_fragments)
        list_particleArs = [
            obj_item.float_aspectRatioWH
            for obj_item in list_particles
            if obj_item.float_aspectRatioWH is not None
        ]
        float_micrometersPerPixel = convert_pixels_to_micrometers(
            float_pixels=1.0,
            float_scalePixels=self.obj_config.float_scalePixels,
            float_scaleMicrometers=self.obj_config.float_scaleMicrometers,
        )

        dict_summary: tp.Dict[str, tp.Any] = {
            "input_path": str(self.obj_config.path_input),
            "output_dir": str(self.obj_config.path_outputDir),
            "model_config_path": str(self.obj_config.path_modelConfig),
            "model_config_parse_status": self.dict_modelConfig.get("config_parse_status"),
            "model_weights_path": str(self.obj_config.path_modelWeights),
            "model_weights_resolved_name": self.resolve_model_weights_path().name,
            "model_name": self.dict_modelConfig.get("model", self.obj_config.path_modelWeights.stem),
            "small_particle": bool(self.obj_config.bool_smallParticle),
            "scale_pixels": float(self.obj_config.float_scalePixels),
            "scale_micrometers": float(self.obj_config.float_scaleMicrometers),
            "micrometers_per_pixel": float(float_micrometersPerPixel),
            "bbox_edge_margin": int(self.obj_config.int_bboxEdgeMargin),
            "tile_edge_margin": int(self.obj_config.int_tileEdgeMargin),
            "tile_size": int(self.obj_config.int_tileSize),
            "stride": int(self.obj_config.int_stride),
            "points_per_tile": int(self.obj_config.int_pointsPerTile),
            "point_min_distance": int(self.obj_config.int_pointMinDistance),
            "point_quality_level": float(self.obj_config.float_pointQualityLevel),
            "point_batch_size": int(self.obj_config.int_pointBatchSize),
            "dedup_iou": float(self.obj_config.float_dedupIou),
            "bbox_dedup_iou": float(self.obj_config.float_bboxDedupIou),
            "use_point_prompts": bool(self.obj_config.bool_usePointPrompts),
            "particle_area_threshold": float(self.obj_config.float_particleAreaThreshold),
            "mask_binarize_threshold": float(self.obj_config.float_maskBinarizeThreshold),
            "min_valid_mask_area": int(self.obj_config.int_minValidMaskArea),
            "mask_morph_kernel_size": int(self.obj_config.int_maskMorphKernelSize),
            "mask_morph_open_iterations": int(self.obj_config.int_maskMorphOpenIterations),
            "mask_morph_close_iterations": int(self.obj_config.int_maskMorphCloseIterations),
            "num_total_objects": int_totalObjects,
            "num_particles": int_particleCount,
            "num_fragments": int_fragmentCount,
            "fragment_count": int_fragmentCount,
            "total_object_count": int_totalObjects,
            "normal_particle_count": int_particleCount,
            "fine_particle_count": int_fragmentCount,
            "fine_particle_ratio_percent": calculate_percentage(int_fragmentCount, int_totalObjects),
            "particle_aspect_ratio_mean": None,
            "particle_aspect_ratio_median": None,
            "particle_aspect_ratio_std": None,
            "particle_aspect_ratio_min": None,
            "particle_aspect_ratio_max": None,
        }

        if list_particleArs:
            arr_particleArs = np.array(list_particleArs, dtype=np.float32)
            dict_summary.update(
                {
                    "particle_aspect_ratio_mean": float(np.mean(arr_particleArs)),
                    "particle_aspect_ratio_median": float(np.median(arr_particleArs)),
                    "particle_aspect_ratio_std": float(np.std(arr_particleArs)),
                    "particle_aspect_ratio_min": float(np.min(arr_particleArs)),
                    "particle_aspect_ratio_max": float(np.max(arr_particleArs)),
                }
            )

        return dict_summary

    def save_outputs(
        self,
        arr_inputBgr: np.ndarray,
        arr_inputRoiBgr: np.ndarray,
        arr_overlayRoi: np.ndarray,
        list_objects: tp.List[ObjectMeasurement],
        list_masks: tp.List[np.ndarray],
        dict_summary: tp.Dict[str, tp.Any],
        dict_roi: tp.Dict[str, int],
        dict_debug: tp.Dict[str, tp.Any],
    ) -> None:
        """이미지, CSV, JSON, histogram 등 최종 산출물을 저장한다.

        Args:
            arr_inputBgr: 원본 입력 이미지.
            arr_inputRoiBgr: 추론에 사용된 ROI 이미지.
            arr_overlayRoi: ROI 위에 객체 시각화를 그린 이미지.
            list_objects: 저장할 객체 측정 결과 리스트.
            list_masks: 각 객체에 대응하는 ROI 좌표계 binary mask 리스트.
            dict_summary: summary.json에 저장할 요약 dict.
            dict_roi: ROI 좌표 및 크기 정보 dict.
            dict_debug: 디버그용 tile/point/mask 정보 dict.

        Returns:
            없음. output directory 아래에 png/csv/json 파일들을 기록한다.
        """
        self.obj_config.path_outputDir.mkdir(parents=True, exist_ok=True)

        arr_overlayFull = arr_inputBgr.copy()
        arr_overlayFull[
            dict_roi["y_min"]:dict_roi["y_max"],
            dict_roi["x_min"]:dict_roi["x_max"],
        ] = arr_overlayRoi
        cv2.rectangle(
            arr_overlayFull,
            (dict_roi["x_min"], dict_roi["y_min"]),
            (dict_roi["x_max"], dict_roi["y_max"]),
            (255, 255, 0),
            2,
        )

        cv2.imwrite(str(self.obj_config.path_outputDir /
                    "01_input.png"), arr_inputBgr)
        cv2.imwrite(str(self.obj_config.path_outputDir /
                    "02_input_roi.png"), arr_inputRoiBgr)
        cv2.imwrite(str(self.obj_config.path_outputDir /
                    "03_overlay_roi.png"), arr_overlayRoi)
        cv2.imwrite(str(self.obj_config.path_outputDir /
                    "04_overlay_full.png"), arr_overlayFull)

        path_csvAll = self.obj_config.path_outputDir / "objects.csv"
        with path_csvAll.open("w", newline="", encoding="utf-8-sig") as obj_f:
            if list_objects:
                obj_writer = csv.DictWriter(
                    obj_f, fieldnames=list(asdict(list_objects[0]).keys()))
                obj_writer.writeheader()
                for obj_measurement in list_objects:
                    obj_writer.writerow(asdict(obj_measurement))

        list_particleRows = [asdict(
            obj_item) for obj_item in list_objects if obj_item.str_category == "particle"]
        path_csvParticle = self.obj_config.path_outputDir / "particles.csv"
        with path_csvParticle.open("w", newline="", encoding="utf-8-sig") as obj_f:
            if list_particleRows:
                obj_writer = csv.DictWriter(
                    obj_f, fieldnames=list(list_particleRows[0].keys()))
                obj_writer.writeheader()
                for dict_row in list_particleRows:
                    obj_writer.writerow(dict_row)

        save_particle_distribution_histogram(
            path_particlesCsv=path_csvParticle,
            path_outputImage=self.obj_config.path_outputDir / "particle_dist.png",
            path_inputImage=self.obj_config.path_input,
        )

        with (self.obj_config.path_outputDir / "summary.json").open("w", encoding="utf-8") as obj_f:
            json.dump(dict_summary, obj_f, ensure_ascii=False, indent=2)

        with (self.obj_config.path_outputDir / "objects.json").open("w", encoding="utf-8") as obj_f:
            json.dump([asdict(obj_item) for obj_item in list_objects],
                      obj_f, ensure_ascii=False, indent=2)

        with (self.obj_config.path_outputDir / "debug.json").open("w", encoding="utf-8") as obj_f:
            json.dump(dict_debug, obj_f, ensure_ascii=False, indent=2)

        if not self.obj_config.bool_saveIndividualMasks:
            return

        path_particleMaskDir = self.obj_config.path_outputDir / "particle_masks"
        path_fragmentMaskDir = self.obj_config.path_outputDir / "fragment_masks"
        path_particleMaskDir.mkdir(parents=True, exist_ok=True)
        path_fragmentMaskDir.mkdir(parents=True, exist_ok=True)

        for obj_measurement, arr_mask in zip(list_objects, list_masks):
            path_targetDir = (
                path_particleMaskDir
                if obj_measurement.str_category == "particle"
                else path_fragmentMaskDir
            )
            str_fileName = f"{obj_measurement.str_category}_{obj_measurement.int_index:04d}.png"
            cv2.imwrite(str(path_targetDir / str_fileName),
                        arr_mask.astype(np.uint8) * 255)

    def process(self) -> Sam2AspectRatioResult:
        """단일 이미지에 대한 전체 파이프라인을 실행한다.

        Returns:
            측정 결과 리스트와 summary dict를 포함하는 `Sam2AspectRatioResult`.
        """
        arr_inputBgr = self.load_image_bgr()
        arr_inputRoiBgr, dict_roi = self.extract_inference_roi(arr_inputBgr)
        arr_masks, arr_scores, dict_debug = self.predict_tiled_point_prompts(
            arr_inputRoiBgr)

        list_objects: tp.List[ObjectMeasurement] = []
        list_validMasks: tp.List[np.ndarray] = []

        for int_index, arr_mask in enumerate(arr_masks):
            float_confidence = None
            if arr_scores is not None and int_index < len(arr_scores):
                float_confidence = float(arr_scores[int_index])

            obj_measurement = self.measure_mask(
                arr_mask, int_index=int_index, float_confidence=float_confidence)
            if obj_measurement is None:
                continue

            list_objects.append(obj_measurement)
            list_validMasks.append(
                self.refine_mask_for_area(arr_mask).astype(np.uint8))

        arr_overlay = self.create_overlay(
            arr_inputRoiBgr, list_objects, list_validMasks)
        dict_summary = self.build_summary(list_objects)
        dict_summary["roi"] = dict_roi
        dict_summary["num_tiles"] = dict_debug.get("num_tiles")
        dict_summary["num_candidate_points"] = dict_debug.get(
            "num_candidate_points")
        dict_summary["num_accepted_masks"] = dict_debug.get(
            "num_accepted_masks")
        dict_summary["num_bbox_dedup_rejected"] = dict_debug.get(
            "num_bbox_dedup_rejected")
        self.save_outputs(
            arr_inputBgr,
            arr_inputRoiBgr,
            arr_overlay,
            list_objects,
            list_validMasks,
            dict_summary,
            dict_roi,
            dict_debug,
        )

        return Sam2AspectRatioResult(
            list_objects=list_objects,
            dict_summary=dict_summary,
        )


def collect_input_groups(path_input: Path) -> tp.List[tp.Tuple[str, tp.List[Path]]]:
    """입력 경로에서 처리할 이미지 그룹 목록을 구성한다.

    Args:
        path_input: 단일 이미지 파일 또는 root directory 경로.

    Returns:
        `(group_id, image_paths)` 튜플의 리스트.
        - 단일 파일 입력이면 파일 stem을 group id로 하는 1개 그룹을 반환한다.
        - 디렉터리 입력이면 `IMG_ID` 폴더 단위 또는 flat 이미지 목록을 그룹으로 반환한다.

    Raises:
        FileNotFoundError: 입력 경로가 없거나, 처리할 이미지 파일을 찾지 못할 때 발생한다.
    """
    if not path_input.exists():
        raise FileNotFoundError(f"입력 경로를 찾을 수 없습니다: {path_input}")

    if path_input.is_file():
        return [(path_input.stem, [path_input])]

    list_groupDirs = sorted(
        [path_item for path_item in path_input.iterdir() if path_item.is_dir()])
    list_groupedImages: tp.List[tp.Tuple[str, tp.List[Path]]] = []

    for path_groupDir in list_groupDirs:
        list_imagePaths = sorted(
            [
                path_item
                for path_item in path_groupDir.iterdir()
                if path_item.is_file() and path_item.suffix.lower() in CONST_SUPPORTED_IMAGE_SUFFIXES
            ]
        )
        if not list_imagePaths:
            continue
        list_groupedImages.append((path_groupDir.name, list_imagePaths))

    if list_groupedImages:
        return list_groupedImages

    # 하위 IMG_ID 폴더가 없는 경우에는 기존 단일 디렉터리 flat 구조도 지원한다.
    list_rootImages = sorted(
        [
            path_item
            for path_item in path_input.iterdir()
            if path_item.is_file() and path_item.suffix.lower() in CONST_SUPPORTED_IMAGE_SUFFIXES
        ]
    )
    if list_rootImages:
        return [(path_input.name, list_rootImages)]

    raise FileNotFoundError(f"디렉터리에서 처리할 이미지 파일을 찾지 못했습니다: {path_input}")


def build_image_output_dir(
    path_outputRoot: Path,
    str_groupId: str,
    path_image: Path,
    bool_isBatch: bool,
) -> Path:
    """입력 형태에 맞는 이미지별 출력 폴더 경로를 구성한다.

    Args:
        path_outputRoot: 최상위 output directory.
        str_groupId: 현재 이미지가 속한 group 또는 IMG_ID 이름.
        path_image: 현재 처리 중인 이미지 경로.
        bool_isBatch: 배치 입력 여부.

    Returns:
        단일 입력이면 `path_outputRoot`, 배치 입력이면
        `path_outputRoot / str_groupId / image_name_ext` 경로.
    """
    if not bool_isBatch:
        return path_outputRoot
    str_dirName = f"{path_image.stem}{path_image.suffix.lower().replace('.', '_')}"
    return path_outputRoot / str_groupId / str_dirName


def calculate_mean_from_optional_values(
    list_values: tp.Iterable[tp.Optional[float]],
) -> tp.Optional[float]:
    """`None`을 제외한 평균을 계산한다.

    Args:
        list_values: `float` 또는 `None`으로 이루어진 iterable.

    Returns:
        유효한 값이 하나라도 있으면 평균값, 없으면 `None`.
    """
    list_validValues = [float(x) for x in list_values if x is not None]
    if not list_validValues:
        return None
    return float(np.mean(np.array(list_validValues, dtype=np.float32)))


def calculate_percentage(int_part: int, int_total: int) -> float:
    """부분/전체 비율을 퍼센트로 환산한다.

    Args:
        int_part: 분자에 해당하는 개수.
        int_total: 분모에 해당하는 전체 개수.

    Returns:
        `int_part / int_total * 100.0`. 전체 개수가 0 이하이면 0.0.
    """
    if int_total <= 0:
        return 0.0
    return float((float(int_part) / float(int_total)) * 100.0)


def build_img_id_summary(
    str_imgId: str,
    path_outputRoot: Path,
    list_fileSummaries: tp.List[tp.Dict[str, tp.Any]],
) -> tp.Dict[str, tp.Any]:
    """동일 IMG_ID 그룹의 이미지 요약들을 집계한다.

    Args:
        str_imgId: 그룹 이름 또는 IMG_ID.
        path_outputRoot: 최상위 output directory.
        list_fileSummaries: 같은 IMG_ID에 속한 이미지별 summary dict 리스트.

    Returns:
        IMG_ID 단위 집계 summary dict. 이미지 수, 객체 수, particle/fragment 수,
        비율, 평균 fragment count, 평균 aspect ratio를 포함한다.
    """
    int_totalObjects = int(sum(dict_item.get("num_total_objects", 0) for dict_item in list_fileSummaries))
    int_particleCount = int(sum(dict_item.get("num_particles", 0) for dict_item in list_fileSummaries))
    int_fragmentCount = int(sum(dict_item.get("num_fragments", 0) for dict_item in list_fileSummaries))
    float_meanAspectRatio = calculate_mean_from_optional_values(
        dict_item.get("particle_aspect_ratio_mean") for dict_item in list_fileSummaries
    )
    float_meanFragmentCount = calculate_mean_from_optional_values(
        float(dict_item.get("fragment_count", 0)) for dict_item in list_fileSummaries
    )

    return {
        "img_id": str_imgId,
        "output_dir": str(path_outputRoot / str_imgId),
        "num_images": len(list_fileSummaries),
        "num_total_objects": int_totalObjects,
        "num_particles": int_particleCount,
        "num_fragments": int_fragmentCount,
        "fragment_count_total": int(sum(dict_item.get("fragment_count", 0) for dict_item in list_fileSummaries)),
        "total_object_count": int_totalObjects,
        "normal_particle_count": int_particleCount,
        "fine_particle_count": int_fragmentCount,
        "fine_particle_ratio_percent": calculate_percentage(int_fragmentCount, int_totalObjects),
        "fragment_count_mean_per_image": float_meanFragmentCount,
        "particle_aspect_ratio_mean": float_meanAspectRatio,
        "files": list_fileSummaries,
    }


def build_batch_summary(
    path_input: Path,
    path_outputDir: Path,
    list_groupSummaries: tp.List[tp.Dict[str, tp.Any]],
) -> tp.Dict[str, tp.Any]:
    """배치 처리 전체에 대한 최종 통합 summary를 생성한다.

    Args:
        path_input: 사용자가 넘긴 원본 입력 경로.
        path_outputDir: 배치 결과가 저장된 최상위 output directory.
        list_groupSummaries: IMG_ID 단위 summary dict 리스트.

    Returns:
        전체 배치 요약 dict. 총 이미지 수, 총 객체 수, particle/fragment 수,
        fine particle 비율, IMG_ID 평균 통계를 포함한다.
    """
    int_totalObjects = int(sum(dict_item.get("num_total_objects", 0) for dict_item in list_groupSummaries))
    int_particleCount = int(sum(dict_item.get("num_particles", 0) for dict_item in list_groupSummaries))
    int_fragmentCount = int(sum(dict_item.get("num_fragments", 0) for dict_item in list_groupSummaries))
    float_meanAspectRatioByImgId = calculate_mean_from_optional_values(
        dict_item.get("particle_aspect_ratio_mean") for dict_item in list_groupSummaries
    )
    float_meanFragmentCountByImgId = calculate_mean_from_optional_values(
        dict_item.get("fragment_count_mean_per_image") for dict_item in list_groupSummaries
    )

    return {
        "input_path": str(path_input),
        "output_dir": str(path_outputDir),
        "num_img_ids": len(list_groupSummaries),
        "num_images": int(sum(dict_item.get("num_images", 0) for dict_item in list_groupSummaries)),
        "num_total_objects": int_totalObjects,
        "num_particles": int_particleCount,
        "num_fragments": int_fragmentCount,
        "fragment_count_total": int(sum(dict_item.get("fragment_count_total", 0) for dict_item in list_groupSummaries)),
        "total_object_count": int_totalObjects,
        "normal_particle_count": int_particleCount,
        "fine_particle_count": int_fragmentCount,
        "fine_particle_ratio_percent": calculate_percentage(int_fragmentCount, int_totalObjects),
        "fragment_count": float_meanFragmentCountByImgId,
        "fragment_count_mean_per_img_id": float_meanFragmentCountByImgId,
        "particle_aspect_ratio_mean": float_meanAspectRatioByImgId,
        "particle_aspect_ratio_mean_per_img_id": float_meanAspectRatioByImgId,
        "img_ids": list_groupSummaries,
    }


def build_default_output_dir_name() -> str:
    """기본 output directory 이름을 생성한다.

    Returns:
        `out_sam2_aspect_ratio_YYYYMMDD_HHMMSS` 형식의 문자열.
    """
    str_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"out_sam2_aspect_ratio_{str_timestamp}"


def run_sam2_aspect_ratio(
    str_input: str,
    str_outputDir: str,
    str_modelConfig: str,
    str_modelWeights: str,
    int_roiXMin: int = CONST_ROI_X_MIN,
    int_roiYMin: int = CONST_ROI_Y_MIN,
    int_roiXMax: int = CONST_ROI_X_MAX,
    int_roiYMax: int = CONST_ROI_Y_MAX,
    int_bboxEdgeMargin: int = CONST_BBOX_EDGE_MARGIN,
    int_tileEdgeMargin: int = CONST_TILE_EDGE_MARGIN,
    float_particleAreaThreshold: float = CONST_PARTICLE_AREA_THRESHOLD,
    float_maskBinarizeThreshold: float = CONST_MASK_BINARIZE_THRESHOLD,
    int_minValidMaskArea: int = CONST_MIN_VALID_MASK_AREA,
    int_maskMorphKernelSize: int = CONST_MASK_MORPH_KERNEL_SIZE,
    int_maskMorphOpenIterations: int = CONST_MASK_MORPH_OPEN_ITERATIONS,
    int_maskMorphCloseIterations: int = CONST_MASK_MORPH_CLOSE_ITERATIONS,
    int_imgSize: int = CONST_DEFAULT_IMAGE_SIZE,
    int_tileSize: int = CONST_DEFAULT_TILE_SIZE,
    int_stride: int = CONST_DEFAULT_TILE_STRIDE,
    int_pointsPerTile: int = CONST_DEFAULT_POINTS_PER_TILE,
    int_pointMinDistance: int = CONST_DEFAULT_POINT_MIN_DISTANCE,
    float_pointQualityLevel: float = CONST_DEFAULT_POINT_QUALITY_LEVEL,
    int_pointBatchSize: int = CONST_DEFAULT_POINT_BATCH_SIZE,
    float_dedupIou: float = CONST_DEFAULT_DEDUP_IOU,
    float_bboxDedupIou: float = CONST_DEFAULT_BBOX_DEDUP_IOU,
    bool_usePointPrompts: bool = CONST_DEFAULT_USE_POINT_PROMPTS,
    bool_smallParticle: bool = CONST_DEFAULT_SMALL_PARTICLE,
    str_device: tp.Optional[str] = None,
    bool_retinaMasks: bool = CONST_DEFAULT_RETINA_MASKS,
    bool_saveIndividualMasks: bool = CONST_DEFAULT_SAVE_INDIVIDUAL_MASKS,
) -> tp.Dict[str, tp.Any]:
    """스크립트 외부에서 사용할 수 있는 최상위 실행 헬퍼 함수.

    Args:
        str_input: 단일 이미지 경로 또는 batch root directory 경로.
        str_outputDir: 결과 저장 root directory 경로.
        str_modelConfig: SAM2 설정 파일 경로.
        str_modelWeights: SAM2 weight 파일 경로.
        int_roiXMin: ROI 시작 x 좌표.
        int_roiYMin: ROI 시작 y 좌표.
        int_roiXMax: ROI 끝 x 좌표.
        int_roiYMax: ROI 끝 y 좌표.
        int_bboxEdgeMargin: ROI 경계 제외 margin.
        int_tileEdgeMargin: tile 경계 제외 margin.
        float_particleAreaThreshold: particle / fragment 분류 기준 면적.
        float_maskBinarizeThreshold: raw SAM mask를 binary mask로 바꾸는 threshold.
        int_minValidMaskArea: 이 값보다 작은 mask는 무시한다.
        int_maskMorphKernelSize: morphology kernel 크기.
        int_maskMorphOpenIterations: open 연산 반복 횟수.
        int_maskMorphCloseIterations: close 연산 반복 횟수.
        int_imgSize: SAM2 추론 input size.
        int_tileSize: ROI 분할 타일 크기.
        int_stride: 타일 stride.
        int_pointsPerTile: tile당 후보 point 수.
        int_pointMinDistance: 후보 point 최소 거리.
        float_pointQualityLevel: point 추출용 quality level.
        int_pointBatchSize: 한 번의 SAM2 호출에 묶을 point 개수.
        float_dedupIou: mask 기준 중복 제거 IoU threshold.
        float_bboxDedupIou: bbox 기준 중복 제거 IoU threshold.
        bool_usePointPrompts: OpenCV 후보점 기반 point prompt 사용 여부.
        bool_smallParticle: small particle scale(`184 px = 10 um`) 사용 여부.
        str_device: 추론 device 문자열. 예: `cpu`, `cuda:0`.
        bool_retinaMasks: retina mask 사용 여부.
        bool_saveIndividualMasks: 개별 mask png 저장 여부.

    Returns:
        단일 입력이면 단일 이미지 summary dict, directory 입력이면 batch summary dict.
    """
    path_input = Path(str_input)
    path_outputRoot = Path(str_outputDir)
    list_inputGroups = collect_input_groups(path_input)
    bool_isBatch = path_input.is_dir()
    float_scalePixels = CONST_SMALL_PARTICLE_SCALE_PIXELS if bool_smallParticle else CONST_SCALE_PIXELS
    float_scaleMicrometers = CONST_SMALL_PARTICLE_SCALE_MICROMETERS if bool_smallParticle else CONST_SCALE_MICROMETERS

    def create_config(str_groupId: str, path_image: Path) -> Sam2AspectRatioConfig:
        """현재 실행 파라미터로 이미지별 config 객체를 만든다."""
        return Sam2AspectRatioConfig(
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
            bool_smallParticle=bool_smallParticle,
            float_scalePixels=float_scalePixels,
            float_scaleMicrometers=float_scaleMicrometers,
            str_device=str_device,
            bool_retinaMasks=bool_retinaMasks,
            bool_saveIndividualMasks=bool_saveIndividualMasks,
        )

    if not bool_isBatch:
        str_groupId, list_imagePaths = list_inputGroups[0]
        print(
            f"[single] processing image: {list_imagePaths[0].name}",
            flush=True,
        )
        obj_service = Sam2AspectRatioService(
            create_config(str_groupId, list_imagePaths[0]))
        obj_result = obj_service.process()
        print(
            f"[single] done: {list_imagePaths[0].name}",
            flush=True,
        )
        return obj_result.dict_summary

    path_outputRoot.mkdir(parents=True, exist_ok=True)

    str_firstGroupId, list_firstGroupImages = list_inputGroups[0]
    print(
        f"[batch] initialize model with first image: {list_firstGroupImages[0].name}",
        flush=True,
    )
    obj_sharedService = Sam2AspectRatioService(
        create_config(str_firstGroupId, list_firstGroupImages[0]))
    obj_sharedService.initialize_model()

    list_groupSummaries: tp.List[tp.Dict[str, tp.Any]] = []
    int_numGroups = len(list_inputGroups)
    for int_groupIndex, (str_groupId, list_imagePaths) in enumerate(list_inputGroups, start=1):
        print(
            f"[batch][group {int_groupIndex}/{int_numGroups}] IMG_ID={str_groupId} "
            f"({len(list_imagePaths)} images)",
            flush=True,
        )
        list_fileSummaries: tp.List[tp.Dict[str, tp.Any]] = []

        int_numImages = len(list_imagePaths)
        for int_imageIndex, path_image in enumerate(list_imagePaths, start=1):
            print(
                f"  [image {int_imageIndex}/{int_numImages}] {path_image.name}",
                flush=True,
            )
            obj_service = Sam2AspectRatioService(
                create_config(str_groupId, path_image))
            obj_service.obj_model = obj_sharedService.obj_model
            obj_service.dict_modelConfig = dict(
                obj_sharedService.dict_modelConfig)
            obj_result = obj_service.process()

            dict_fileSummary = dict(obj_result.dict_summary)
            dict_fileSummary["img_id"] = str_groupId
            dict_fileSummary["image_name"] = path_image.name
            dict_fileSummary["image_path"] = str(path_image)
            list_fileSummaries.append(dict_fileSummary)

        dict_groupSummary = build_img_id_summary(
            str_imgId=str_groupId,
            path_outputRoot=path_outputRoot,
            list_fileSummaries=list_fileSummaries,
        )
        with (path_outputRoot / str_groupId / "img_id_summary.json").open("w", encoding="utf-8") as obj_f:
            json.dump(dict_groupSummary, obj_f, ensure_ascii=False, indent=2)
        list_groupSummaries.append(dict_groupSummary)
        print(
            f"[batch][group done] IMG_ID={str_groupId} "
            f"mean_ar={dict_groupSummary.get('particle_aspect_ratio_mean')} "
            f"mean_fragment_count={dict_groupSummary.get('fragment_count_mean_per_image')}",
            flush=True,
        )

    dict_batchSummary = build_batch_summary(
        path_input=path_input,
        path_outputDir=path_outputRoot,
        list_groupSummaries=list_groupSummaries,
    )
    with (path_outputRoot / "batch_summary.json").open("w", encoding="utf-8") as obj_f:
        json.dump(dict_batchSummary, obj_f, ensure_ascii=False, indent=2)
    print(
        f"[batch] done: num_img_ids={dict_batchSummary['num_img_ids']} "
        f"num_images={dict_batchSummary['num_images']}",
        flush=True,
    )
    return dict_batchSummary


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 생성한다.

    Returns:
        본 스크립트가 지원하는 모든 command-line option이 등록된
        `argparse.ArgumentParser` 객체.
    """
    obj_parser = argparse.ArgumentParser(
        description="SAM2로 객체를 분할하고 particle / fragment 분류 및 particle 종횡비를 계산합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    obj_parser.add_argument(
        "--input",
        default="img/5000_test_1.jpg",
        help="입력 이미지 또는 이미지 디렉터리 경로",
    )
    obj_parser.add_argument(
        "--output_dir",
        default=build_default_output_dir_name(),
        help="결과 저장 폴더",
    )
    obj_parser.add_argument(
        "--model_cfg",
        default="model/sam2.1_hiera_t.yaml",
        help="SAM2 YAML 설정 파일 경로",
    )
    obj_parser.add_argument(
        "--model",
        default="model/sam2.1_hiera_base_plus.pt",
        help="SAM2 가중치 파일 경로",
    )
    obj_parser.add_argument(
        "--roi_x_min",
        type=int,
        default=CONST_ROI_X_MIN,
        help="SAM2 추론 ROI의 시작 x 좌표",
    )
    obj_parser.add_argument(
        "--roi_y_min",
        type=int,
        default=CONST_ROI_Y_MIN,
        help="SAM2 추론 ROI의 시작 y 좌표",
    )
    obj_parser.add_argument(
        "--roi_x_max",
        type=int,
        default=CONST_ROI_X_MAX,
        help="SAM2 추론 ROI의 끝 x 좌표",
    )
    obj_parser.add_argument(
        "--roi_y_max",
        type=int,
        default=CONST_ROI_Y_MAX,
        help="SAM2 추론 ROI의 끝 y 좌표",
    )
    obj_parser.add_argument(
        "--bbox_edge_margin",
        type=int,
        default=CONST_BBOX_EDGE_MARGIN,
        help="ROI 경계와 이 margin 이내인 bbox는 제외",
    )
    obj_parser.add_argument(
        "--tile_edge_margin",
        type=int,
        default=CONST_TILE_EDGE_MARGIN,
        help="tile 경계와 이 margin 이내인 bbox는 제외",
    )
    obj_parser.add_argument(
        "--area_threshold",
        type=float,
        default=CONST_PARTICLE_AREA_THRESHOLD,
        help="particle / fragment 분류 면적 threshold",
    )
    obj_parser.add_argument(
        "--mask_binarize_threshold",
        type=float,
        default=CONST_MASK_BINARIZE_THRESHOLD,
        help="SAM2 raw mask를 binary mask로 바꾸는 threshold",
    )
    obj_parser.add_argument(
        "--min_valid_mask_area",
        type=int,
        default=CONST_MIN_VALID_MASK_AREA,
        help="이 값보다 작은 마스크는 무시",
    )
    obj_parser.add_argument(
        "--mask_morph_kernel_size",
        type=int,
        default=CONST_MASK_MORPH_KERNEL_SIZE,
        help="면적 계산 전 morphology kernel size. 0/1이면 비활성화",
    )
    obj_parser.add_argument(
        "--mask_morph_open_iterations",
        type=int,
        default=CONST_MASK_MORPH_OPEN_ITERATIONS,
        help="면적 계산 전 open iteration 수",
    )
    obj_parser.add_argument(
        "--mask_morph_close_iterations",
        type=int,
        default=CONST_MASK_MORPH_CLOSE_ITERATIONS,
        help="면적 계산 전 close iteration 수",
    )
    obj_parser.add_argument(
        "--imgsz",
        type=int,
        default=CONST_DEFAULT_IMAGE_SIZE,
        help="SAM2 추론 이미지 크기",
    )
    obj_parser.add_argument(
        "--tile_size",
        type=int,
        default=CONST_DEFAULT_TILE_SIZE,
        help="ROI 내부 타일 크기",
    )
    obj_parser.add_argument(
        "--stride",
        type=int,
        default=CONST_DEFAULT_TILE_STRIDE,
        help="타일 stride",
    )
    obj_parser.add_argument(
        "--points_per_tile",
        type=int,
        default=CONST_DEFAULT_POINTS_PER_TILE,
        help="각 타일에서 추출할 후보점 수",
    )
    obj_parser.add_argument(
        "--point_min_distance",
        type=int,
        default=CONST_DEFAULT_POINT_MIN_DISTANCE,
        help="후보점 최소 거리",
    )
    obj_parser.add_argument(
        "--point_quality_level",
        type=float,
        default=CONST_DEFAULT_POINT_QUALITY_LEVEL,
        help="goodFeaturesToTrack qualityLevel",
    )
    obj_parser.add_argument(
        "--point_batch_size",
        type=int,
        default=CONST_DEFAULT_POINT_BATCH_SIZE,
        help="한 번의 SAM2 호출에 묶어 넣을 point 수",
    )
    obj_parser.add_argument(
        "--dedup_iou",
        type=float,
        default=CONST_DEFAULT_DEDUP_IOU,
        help="타일/포인트 간 중복 마스크 제거 IoU threshold",
    )
    obj_parser.add_argument(
        "--bbox_dedup_iou",
        type=float,
        default=CONST_DEFAULT_BBOX_DEDUP_IOU,
        help="IoU dedup 전에 적용할 bbox 중복 제거 IoU threshold",
    )
    obj_parser.add_argument(
        "--device",
        default=None,
        help="예: cpu, cuda:0",
    )
    obj_parser.add_argument(
        "--retina_masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retina mask 사용",
    )
    obj_parser.add_argument(
        "--use_point_prompts",
        action=argparse.BooleanOptionalAction,
        default=CONST_DEFAULT_USE_POINT_PROMPTS,
        help="opencv 후보점 기반 point prompt 추론 사용 여부",
    )
    obj_parser.add_argument(
        "--small_particle",
        action="store_true",
        help="길이 환산 scale을 184 pixel = 10 um 기준으로 변경",
    )
    obj_parser.add_argument(
        "--save_mask_imgs",
        "--save_individual_masks",
        dest="save_mask_imgs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="개별 particle / fragment mask 이미지 저장",
    )
    return obj_parser


def main() -> None:
    """CLI 진입점.

    Returns:
        없음. command-line argument를 파싱한 뒤 파이프라인을 실행하고 summary를
        표준 출력에 JSON 형태로 출력한다.
    """
    obj_args = build_arg_parser().parse_args()

    dict_summary = run_sam2_aspect_ratio(
        str_input=obj_args.input,
        str_outputDir=obj_args.output_dir,
        str_modelConfig=obj_args.model_cfg,
        str_modelWeights=obj_args.model,
        int_roiXMin=obj_args.roi_x_min,
        int_roiYMin=obj_args.roi_y_min,
        int_roiXMax=obj_args.roi_x_max,
        int_roiYMax=obj_args.roi_y_max,
        int_bboxEdgeMargin=obj_args.bbox_edge_margin,
        int_tileEdgeMargin=obj_args.tile_edge_margin,
        float_particleAreaThreshold=obj_args.area_threshold,
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
        bool_smallParticle=obj_args.small_particle,
        str_device=obj_args.device,
        bool_retinaMasks=obj_args.retina_masks,
        bool_saveIndividualMasks=obj_args.save_mask_imgs,
    )

    print("===== SAM2 Aspect Ratio 결과 요약 =====")
    print(json.dumps(dict_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print(f"Elapsed time: {time.time() - start_time:.4f} seconds")
