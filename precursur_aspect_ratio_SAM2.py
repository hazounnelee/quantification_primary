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
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import SAM


# =========================================================
# Const / Tunable Hyperparameters
# =========================================================
# `particle` / `fragment` 분류 기준 면적
CONST_PARTICLE_AREA_THRESHOLD: float = 100.0

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
CONST_DEFAULT_IMAGE_SIZE: int = 1024
CONST_DEFAULT_RETINA_MASKS: bool = True
CONST_DEFAULT_SAVE_INDIVIDUAL_MASKS: bool = True


@dataclass
class Sam2AspectRatioConfig:
    """SAM2 기반 객체 분류 및 종횡비 측정 설정."""

    path_input: Path
    path_outputDir: Path
    path_modelConfig: Path
    path_modelWeights: Path
    float_particleAreaThreshold: float = CONST_PARTICLE_AREA_THRESHOLD
    float_maskBinarizeThreshold: float = CONST_MASK_BINARIZE_THRESHOLD
    int_minValidMaskArea: int = CONST_MIN_VALID_MASK_AREA
    int_maskMorphKernelSize: int = CONST_MASK_MORPH_KERNEL_SIZE
    int_maskMorphOpenIterations: int = CONST_MASK_MORPH_OPEN_ITERATIONS
    int_maskMorphCloseIterations: int = CONST_MASK_MORPH_CLOSE_ITERATIONS
    int_imgSize: int = CONST_DEFAULT_IMAGE_SIZE
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
    float_centroidX: float
    float_centroidY: float
    int_longestHorizontal: int
    int_longestVertical: int
    float_aspectRatioWH: tp.Optional[float]


@dataclass
class Sam2AspectRatioResult:
    """전체 실행 결과."""

    list_objects: tp.List[ObjectMeasurement]
    dict_summary: tp.Dict[str, tp.Any]


class Sam2AspectRatioService:
    """SAM2 추론 및 후처리를 담당하는 서비스 클래스."""

    def __init__(self, obj_config: Sam2AspectRatioConfig) -> None:
        self.obj_config = obj_config
        self.obj_model: tp.Optional[SAM] = None
        self.dict_modelConfig: tp.Dict[str, tp.Any] = dict()

    def validate_inputs(self) -> None:
        """입력 경로 유효성 검증."""
        list_requiredPaths = [
            self.obj_config.path_input,
            self.obj_config.path_modelConfig,
            self.obj_config.path_modelWeights,
        ]
        for path_item in list_requiredPaths:
            if not path_item.exists():
                raise FileNotFoundError(f"필수 경로를 찾을 수 없습니다: {path_item}")

    def load_model_config(self) -> None:
        """SAM2 YAML 설정을 로드하여 결과 메타데이터에 포함."""
        str_rawText = self.obj_config.path_modelConfig.read_text(encoding="utf-8", errors="ignore")

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
        """SAM2 모델 초기화."""
        self.validate_inputs()
        self.load_model_config()
        path_resolvedWeights = self.resolve_model_weights_path()
        self.obj_model = SAM(str(path_resolvedWeights))

    def load_image_bgr(self) -> np.ndarray:
        """입력 이미지를 BGR 형식으로 로드."""
        arr_image = cv2.imread(str(self.obj_config.path_input), cv2.IMREAD_COLOR)
        if arr_image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {self.obj_config.path_input}")
        return arr_image

    def predict(self) -> tp.Tuple[np.ndarray, tp.Optional[np.ndarray], tp.Optional[np.ndarray]]:
        """SAM2 자동 세그멘테이션 수행."""
        if self.obj_model is None:
            self.initialize_model()

        dict_predictKwargs: tp.Dict[str, tp.Any] = {
            "source": str(self.obj_config.path_input),
            "imgsz": self.obj_config.int_imgSize,
            "retina_masks": self.obj_config.bool_retinaMasks,
            "verbose": False,
        }
        if self.obj_config.str_device:
            dict_predictKwargs["device"] = self.obj_config.str_device

        list_results = self.obj_model(**dict_predictKwargs)  # type: ignore[misc]
        if not list_results:
            raise RuntimeError("SAM2 결과가 비어 있습니다.")

        obj_result = list_results[0]

        arr_masks = np.empty((0, 0, 0), dtype=np.uint8)
        if obj_result.masks is not None and obj_result.masks.data is not None:
            arr_maskData = obj_result.masks.data.detach().cpu().numpy()
            arr_masks = (arr_maskData > self.obj_config.float_maskBinarizeThreshold).astype(np.uint8)

        arr_boxes = None
        if obj_result.boxes is not None and obj_result.boxes.xyxy is not None:
            arr_boxes = obj_result.boxes.xyxy.detach().cpu().numpy()

        arr_scores = None
        if obj_result.boxes is not None and obj_result.boxes.conf is not None:
            arr_scores = obj_result.boxes.conf.detach().cpu().numpy()

        return arr_masks, arr_boxes, arr_scores

    def refine_mask_for_area(self, arr_mask: np.ndarray) -> np.ndarray:
        """
        면적 계산 전 binary mask 후처리.

        area_threshold 자체뿐 아니라 이 함수의 threshold / morphology 설정도
        최종 area 값에 직접 영향을 준다.
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
        """마스크 내부의 가장 긴 가로/세로 span 길이 계산."""
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
        """외곽 contour 중 가장 큰 contour 반환."""
        list_contours, _ = cv2.findContours(arr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not list_contours:
            return None
        return max(list_contours, key=cv2.contourArea)

    def measure_mask(
        self,
        arr_mask: np.ndarray,
        int_index: int,
        float_confidence: tp.Optional[float],
    ) -> tp.Optional[ObjectMeasurement]:
        """개별 마스크의 면적/위치/종횡비 측정."""
        arr_refinedMask = self.refine_mask_for_area(arr_mask)
        int_maskArea = int(arr_refinedMask.sum())
        if int_maskArea < self.obj_config.int_minValidMaskArea:
            return None

        arr_contour = self.extract_largest_contour(arr_refinedMask)
        if arr_contour is None:
            return None

        int_x, int_y, int_w, int_h = cv2.boundingRect(arr_contour)

        obj_moments = cv2.moments(arr_contour)
        if obj_moments["m00"] > 0.0:
            float_cx = float(obj_moments["m10"] / obj_moments["m00"])
            float_cy = float(obj_moments["m01"] / obj_moments["m00"])
        else:
            float_cx = float(int_x + int_w / 2.0)
            float_cy = float(int_y + int_h / 2.0)

        int_horizontal = self.get_longest_span(arr_refinedMask, bool_horizontal=True)
        int_vertical = self.get_longest_span(arr_refinedMask, bool_horizontal=False)

        str_category = (
            "particle"
            if int_maskArea >= int(round(self.obj_config.float_particleAreaThreshold))
            else "fragment"
        )

        float_aspectRatio = None
        if str_category == "particle":
            float_aspectRatio = float(int_horizontal / max(1, int_vertical))

        return ObjectMeasurement(
            int_index=int_index,
            str_category=str_category,
            int_maskArea=int_maskArea,
            float_confidence=float_confidence,
            int_bboxX=int(int_x),
            int_bboxY=int(int_y),
            int_bboxWidth=int(int_w),
            int_bboxHeight=int(int_h),
            float_centroidX=float_cx,
            float_centroidY=float_cy,
            int_longestHorizontal=int_horizontal,
            int_longestVertical=int_vertical,
            float_aspectRatioWH=float_aspectRatio,
        )

    def create_overlay(
        self,
        arr_imageBgr: np.ndarray,
        list_objects: tp.List[ObjectMeasurement],
        list_masks: tp.List[np.ndarray],
    ) -> np.ndarray:
        """마스크와 라벨을 시각화한 오버레이 생성."""
        arr_overlay = arr_imageBgr.copy()

        for obj_measurement, arr_mask in zip(list_objects, list_masks):
            tpl_color = (60, 220, 60) if obj_measurement.str_category == "particle" else (0, 165, 255)
            arr_overlay[arr_mask > 0] = (
                arr_overlay[arr_mask > 0].astype(np.float32) * 0.55
                + np.array(tpl_color, dtype=np.float32) * 0.45
            ).astype(np.uint8)

        for obj_measurement, arr_mask in zip(list_objects, list_masks):
            arr_contour = self.extract_largest_contour(arr_mask)
            if arr_contour is None:
                continue

            tpl_color = (0, 255, 0) if obj_measurement.str_category == "particle" else (0, 140, 255)
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
        """요약 통계 생성."""
        list_particles = [obj_item for obj_item in list_objects if obj_item.str_category == "particle"]
        list_fragments = [obj_item for obj_item in list_objects if obj_item.str_category == "fragment"]
        list_particleArs = [
            obj_item.float_aspectRatioWH
            for obj_item in list_particles
            if obj_item.float_aspectRatioWH is not None
        ]

        dict_summary: tp.Dict[str, tp.Any] = {
            "input_path": str(self.obj_config.path_input),
            "output_dir": str(self.obj_config.path_outputDir),
            "model_config_path": str(self.obj_config.path_modelConfig),
            "model_config_parse_status": self.dict_modelConfig.get("config_parse_status"),
            "model_weights_path": str(self.obj_config.path_modelWeights),
            "model_weights_resolved_name": self.resolve_model_weights_path().name,
            "model_name": self.dict_modelConfig.get("model", self.obj_config.path_modelWeights.stem),
            "particle_area_threshold": float(self.obj_config.float_particleAreaThreshold),
            "mask_binarize_threshold": float(self.obj_config.float_maskBinarizeThreshold),
            "min_valid_mask_area": int(self.obj_config.int_minValidMaskArea),
            "mask_morph_kernel_size": int(self.obj_config.int_maskMorphKernelSize),
            "mask_morph_open_iterations": int(self.obj_config.int_maskMorphOpenIterations),
            "mask_morph_close_iterations": int(self.obj_config.int_maskMorphCloseIterations),
            "num_total_objects": len(list_objects),
            "num_particles": len(list_particles),
            "num_fragments": len(list_fragments),
            "fragment_count": len(list_fragments),
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
        arr_overlay: np.ndarray,
        list_objects: tp.List[ObjectMeasurement],
        list_masks: tp.List[np.ndarray],
        dict_summary: tp.Dict[str, tp.Any],
    ) -> None:
        """이미지/CSV/JSON 결과 저장."""
        self.obj_config.path_outputDir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(self.obj_config.path_outputDir / "01_input.png"), arr_inputBgr)
        cv2.imwrite(str(self.obj_config.path_outputDir / "02_overlay.png"), arr_overlay)

        path_csvAll = self.obj_config.path_outputDir / "objects.csv"
        with path_csvAll.open("w", newline="", encoding="utf-8-sig") as obj_f:
            if list_objects:
                obj_writer = csv.DictWriter(obj_f, fieldnames=list(asdict(list_objects[0]).keys()))
                obj_writer.writeheader()
                for obj_measurement in list_objects:
                    obj_writer.writerow(asdict(obj_measurement))

        list_particleRows = [asdict(obj_item) for obj_item in list_objects if obj_item.str_category == "particle"]
        path_csvParticle = self.obj_config.path_outputDir / "particles.csv"
        with path_csvParticle.open("w", newline="", encoding="utf-8-sig") as obj_f:
            if list_particleRows:
                obj_writer = csv.DictWriter(obj_f, fieldnames=list(list_particleRows[0].keys()))
                obj_writer.writeheader()
                for dict_row in list_particleRows:
                    obj_writer.writerow(dict_row)

        with (self.obj_config.path_outputDir / "summary.json").open("w", encoding="utf-8") as obj_f:
            json.dump(dict_summary, obj_f, ensure_ascii=False, indent=2)

        with (self.obj_config.path_outputDir / "objects.json").open("w", encoding="utf-8") as obj_f:
            json.dump([asdict(obj_item) for obj_item in list_objects], obj_f, ensure_ascii=False, indent=2)

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
            cv2.imwrite(str(path_targetDir / str_fileName), arr_mask.astype(np.uint8) * 255)

    def process(self) -> Sam2AspectRatioResult:
        """전체 파이프라인 실행."""
        arr_inputBgr = self.load_image_bgr()
        arr_masks, _, arr_scores = self.predict()

        list_objects: tp.List[ObjectMeasurement] = []
        list_validMasks: tp.List[np.ndarray] = []

        for int_index, arr_mask in enumerate(arr_masks):
            float_confidence = None
            if arr_scores is not None and int_index < len(arr_scores):
                float_confidence = float(arr_scores[int_index])

            obj_measurement = self.measure_mask(arr_mask, int_index=int_index, float_confidence=float_confidence)
            if obj_measurement is None:
                continue

            list_objects.append(obj_measurement)
            list_validMasks.append(self.refine_mask_for_area(arr_mask).astype(np.uint8))

        arr_overlay = self.create_overlay(arr_inputBgr, list_objects, list_validMasks)
        dict_summary = self.build_summary(list_objects)
        self.save_outputs(arr_inputBgr, arr_overlay, list_objects, list_validMasks, dict_summary)

        return Sam2AspectRatioResult(
            list_objects=list_objects,
            dict_summary=dict_summary,
        )


def run_sam2_aspect_ratio(
    str_input: str,
    str_outputDir: str,
    str_modelConfig: str,
    str_modelWeights: str,
    float_particleAreaThreshold: float = CONST_PARTICLE_AREA_THRESHOLD,
    float_maskBinarizeThreshold: float = CONST_MASK_BINARIZE_THRESHOLD,
    int_minValidMaskArea: int = CONST_MIN_VALID_MASK_AREA,
    int_maskMorphKernelSize: int = CONST_MASK_MORPH_KERNEL_SIZE,
    int_maskMorphOpenIterations: int = CONST_MASK_MORPH_OPEN_ITERATIONS,
    int_maskMorphCloseIterations: int = CONST_MASK_MORPH_CLOSE_ITERATIONS,
    int_imgSize: int = CONST_DEFAULT_IMAGE_SIZE,
    str_device: tp.Optional[str] = None,
    bool_retinaMasks: bool = CONST_DEFAULT_RETINA_MASKS,
    bool_saveIndividualMasks: bool = CONST_DEFAULT_SAVE_INDIVIDUAL_MASKS,
) -> tp.Dict[str, tp.Any]:
    """외부 호출용 헬퍼 함수."""
    obj_config = Sam2AspectRatioConfig(
        path_input=Path(str_input),
        path_outputDir=Path(str_outputDir),
        path_modelConfig=Path(str_modelConfig),
        path_modelWeights=Path(str_modelWeights),
        float_particleAreaThreshold=float_particleAreaThreshold,
        float_maskBinarizeThreshold=float_maskBinarizeThreshold,
        int_minValidMaskArea=int_minValidMaskArea,
        int_maskMorphKernelSize=int_maskMorphKernelSize,
        int_maskMorphOpenIterations=int_maskMorphOpenIterations,
        int_maskMorphCloseIterations=int_maskMorphCloseIterations,
        int_imgSize=int_imgSize,
        str_device=str_device,
        bool_retinaMasks=bool_retinaMasks,
        bool_saveIndividualMasks=bool_saveIndividualMasks,
    )
    obj_service = Sam2AspectRatioService(obj_config)
    obj_result = obj_service.process()
    return obj_result.dict_summary


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서 생성."""
    obj_parser = argparse.ArgumentParser(
        description="SAM2로 객체를 분할하고 particle / fragment 분류 및 particle 종횡비를 계산합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    obj_parser.add_argument(
        "--input",
        default="img/5000_test_1.jpg",
        help="입력 이미지 경로",
    )
    obj_parser.add_argument(
        "--output_dir",
        default="out_sam2_aspect_ratio",
        help="결과 저장 폴더",
    )
    obj_parser.add_argument(
        "--model_cfg",
        default="model/sam2.1_hiera_t.yaml",
        help="SAM2 YAML 설정 파일 경로",
    )
    obj_parser.add_argument(
        "--model",
        default="model/sam2.1_hiera_tiny.pt",
        help="SAM2 가중치 파일 경로",
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
        "--save_individual_masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="개별 particle / fragment 마스크 저장",
    )
    return obj_parser


def main() -> None:
    """CLI 진입점."""
    obj_args = build_arg_parser().parse_args()

    dict_summary = run_sam2_aspect_ratio(
        str_input=obj_args.input,
        str_outputDir=obj_args.output_dir,
        str_modelConfig=obj_args.model_cfg,
        str_modelWeights=obj_args.model,
        float_particleAreaThreshold=obj_args.area_threshold,
        float_maskBinarizeThreshold=obj_args.mask_binarize_threshold,
        int_minValidMaskArea=obj_args.min_valid_mask_area,
        int_maskMorphKernelSize=obj_args.mask_morph_kernel_size,
        int_maskMorphOpenIterations=obj_args.mask_morph_open_iterations,
        int_maskMorphCloseIterations=obj_args.mask_morph_close_iterations,
        int_imgSize=obj_args.imgsz,
        str_device=obj_args.device,
        bool_retinaMasks=obj_args.retina_masks,
        bool_saveIndividualMasks=obj_args.save_individual_masks,
    )

    print("===== SAM2 Aspect Ratio 결과 요약 =====")
    print(json.dumps(dict_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
