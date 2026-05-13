#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""특정 LOT 폴더의 모든 입자(전구체+미분)에 대해 히스토그램+1D 산점도를 생성하는 스크립트."""
import argparse
import sys
from pathlib import Path

from utils.histograms import save_lot_particle_scatter_histogram


def main() -> None:
    obj_parser = argparse.ArgumentParser(
        description=(
            "LOT 출력 폴더의 모든 objects.csv를 읽어 "
            "입도 히스토그램 + 1D 산점도를 생성합니다."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    obj_parser.add_argument(
        "lot_dir",
        metavar="LOT_DIR",
        help="LOT 출력 폴더 경로 (하위에 이미지별 폴더와 objects.csv가 있어야 함)",
    )
    obj_parser.add_argument(
        "--output", "-o", default=None,
        help="저장할 PNG 경로. 미지정 시 LOT_DIR/lot_scatter.png.",
    )
    obj_parser.add_argument(
        "--lot", default=None,
        help="차트 제목에 표시할 LOT 이름. 미지정 시 폴더명 사용.",
    )
    obj_args = obj_parser.parse_args()

    path_lot_dir = Path(obj_args.lot_dir)
    if not path_lot_dir.is_dir():
        print(f"[ERROR] 폴더를 찾을 수 없습니다: {path_lot_dir}", file=sys.stderr)
        sys.exit(1)

    str_lot = obj_args.lot or path_lot_dir.name
    path_output = Path(obj_args.output) if obj_args.output else path_lot_dir / "lot_scatter.png"
    path_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] lot_dir: {path_lot_dir}")
    print(f"[info] output : {path_output}")
    print(f"[info] lot    : {str_lot}")

    save_lot_particle_scatter_histogram(path_lot_dir, path_output, str_lot=str_lot)
    print(f"[done] {path_output.name} 저장 완료")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
