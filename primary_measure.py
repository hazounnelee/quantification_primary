#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1차 입자 두께 측정 CLI 진입점."""
import json
import sys
import time

from services.primary_particle import run_primary_particle_analysis, build_primary_arg_parser
from utils.metrics import json_default


def main() -> None:
    # Windows 한글 출력 처리
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    obj_parser = build_primary_arg_parser()
    obj_args = obj_parser.parse_args()
    float_start = time.perf_counter()

    dict_summary = run_primary_particle_analysis(
        str_input=obj_args.input,
        str_outputDir=obj_args.output_dir,
        str_modelWeights=obj_args.model,
        str_modelConfig=obj_args.model_cfg,
        str_device=obj_args.device,
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
        bool_autoCenterCrop=obj_args.auto_center_crop,
        float_centerCropRatio=obj_args.center_crop_ratio,
        bool_saveIndividualMasks=obj_args.save_mask_imgs,
        bool_retinaMasks=obj_args.retina_masks,
        str_particleType=obj_args.particle_type or "unknown",
        str_magnification=obj_args.magnification or "unknown",
        str_particleMode=obj_args.particle_mode,
        bool_autoDetectSphere=obj_args.auto_detect_sphere,
        float_sphereCapFraction=obj_args.sphere_cap_fraction,
        str_measureMode=obj_args.measure_mode,
    )

    print("===== 1차 입자 분석 결과 요약 =====")
    print(json.dumps(dict_summary, ensure_ascii=False, indent=2, default=json_default))
    print(f"Elapsed time: {time.perf_counter() - float_start:.4f} seconds")


if __name__ == "__main__":
    main()
