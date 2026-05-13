#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_summary.json을 읽어 2차 입자 히스토그램을 재생성하는 독립 스크립트."""
import argparse
import json
import sys
from pathlib import Path

from utils.histograms import save_secondary_batch_histograms, save_lot_particle_scatter_histogram


def main() -> None:
    obj_parser = argparse.ArgumentParser(
        description="batch_summary.json에서 2차 입자 히스토그램을 생성합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    obj_parser.add_argument(
        "summary",
        metavar="BATCH_SUMMARY_JSON",
        help="batch_summary.json 경로",
    )
    obj_parser.add_argument(
        "--output_dir", "-o", default=None,
        help="히스토그램 저장 디렉토리. 미지정 시 summary 파일과 같은 디렉토리.",
    )
    obj_parser.add_argument(
        "--lot", default=None,
        help=(
            "LOT 하위 폴더명. 지정하면 해당 LOT의 모든 입자(전구체+미분)에 대해 "
            "히스토그램+1D 산점도를 추가로 생성한다."
        ),
    )
    obj_args = obj_parser.parse_args()

    path_summary = Path(obj_args.summary)
    if not path_summary.exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {path_summary}", file=sys.stderr)
        sys.exit(1)

    with path_summary.open(encoding="utf-8") as obj_f:
        dict_batchSummary = json.load(obj_f)

    path_outputDir = Path(obj_args.output_dir) if obj_args.output_dir else path_summary.parent
    path_outputDir.mkdir(parents=True, exist_ok=True)

    str_lot_title = obj_args.lot or path_summary.parent.name

    print(f"[info] summary: {path_summary}")
    print(f"[info] output : {path_outputDir}")
    print(f"[info] lot    : {str_lot_title}")

    save_secondary_batch_histograms(dict_batchSummary, path_outputDir, str_lot=str_lot_title)
    print("[done] batch_hist_size.png, batch_hist_size_std.png, batch_hist_sphericity.png, batch_hist_fine_ratio.png 저장 완료")

    if obj_args.lot:
        path_lot_dir = path_summary.parent / obj_args.lot
        if not path_lot_dir.is_dir():
            print(f"[WARN] LOT 폴더를 찾을 수 없습니다: {path_lot_dir}", file=sys.stderr)
        else:
            path_scatter = path_outputDir / f"lot_{obj_args.lot}_scatter.png"
            save_lot_particle_scatter_histogram(path_lot_dir, path_scatter, str_lot=obj_args.lot)
            print(f"[done] {path_scatter.name} 저장 완료")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
