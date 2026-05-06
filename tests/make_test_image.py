#!/usr/bin/env python3
"""침상 입자 합성 테스트 이미지 생성.

목적: fusion 후 min_length 적용 순서를 검증

입자 설계 (--area_threshold 500, --min_length 40):
  A)  긴 침상   120px × 20px at 45° → LSD 검출 후 longAxisPx ~120  → 통과
  B1) 짧은 침상  38px × 20px at 45°, center(480,360)
  B2) 짧은 침상  38px × 20px at 45°, center(500,360) [B1과 20px 평행 이동]
      → 이미지에서 분리, 마스크는 겹침
      → 개별 longAxisPx ~38 < 40 → no-fuse 시 탈락
      → fuse 후 longAxisPx ~58 ≥ 40 → fuse 시 통과
  C)  짧은 침상   20px × 15px at 45° → longAxisPx ~20 → 항상 탈락

기대:
  case1 (min_length=0,  no fuse): A + B1 + B2 + C = ~4개
  case2 (min_length=40, no fuse): A 만 = 1개  (B1,B2 탈락)
  case3 (min_length=40, --fuse):  A + fused-B = 2개
"""
import cv2
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

H, W = 768, 1024
arr = np.zeros((H, W), dtype=np.uint8)


def draw_rod(img, cx, cy, length, thickness, angle_deg, brightness=240):
    rad = np.radians(angle_deg)
    dx, dy = np.cos(rad) * length / 2, np.sin(rad) * length / 2
    pt1 = (int(cx - dx), int(cy - dy))
    pt2 = (int(cx + dx), int(cy + dy))
    cv2.line(img, pt1, pt2, brightness, thickness)


# A) 긴 침상
draw_rod(arr, cx=200, cy=200, length=120, thickness=20, angle_deg=45)

# B1, B2: x축으로 20px 떨어진 평행 침상 (이미지에서 분리, 마스크 겹침)
draw_rod(arr, cx=480, cy=360, length=38, thickness=20, angle_deg=45)
draw_rod(arr, cx=500, cy=360, length=38, thickness=20, angle_deg=45)

# C) 짧은 고립 침상
draw_rod(arr, cx=820, cy=600, length=20, thickness=15, angle_deg=45)

path_out = OUT / "test_acicular.png"
cv2.imwrite(str(path_out), arr)
print(f"saved: {path_out}")

# LSD 검출 미리보기
obj_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
arr_eq = obj_clahe.apply(arr)
obj_lsd = cv2.createLineSegmentDetector(0)
arr_lines, _, _, _ = obj_lsd.detect(arr_eq)
print(f"LSD raw: {len(arr_lines)}개 (길이>20px만 출력)")
for line in sorted(arr_lines, key=lambda l: -np.sqrt((l[0][2]-l[0][0])**2+(l[0][3]-l[0][1])**2)):
    x1,y1,x2,y2 = line[0]
    lp = np.sqrt((x2-x1)**2+(y2-y1)**2)
    if lp > 20:
        ang = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
        print(f"  len={lp:.1f}  angle={ang:.1f}°  ({x1:.0f},{y1:.0f})")
