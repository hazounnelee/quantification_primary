#!/usr/bin/env python3
"""Generate pipeline PDF diagrams for primary_measure.py and secondary_measure.py."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
import numpy as np

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────────
_NANUM_PATH = (
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8"
    "/7a0b5c0f3c1d41c4c52a33343496c9c65ad52c50.asset/AssetData/NanumGothic.ttc"
)
try:
    font_manager.fontManager.addfont(_NANUM_PATH)
    matplotlib.rcParams["font.family"] = "Nanum Gothic"
except Exception:
    pass  # 폰트 없으면 경고만 뜨고 계속 진행

# ── Colour palette ─────────────────────────────────────────────────────────────
C_IO      = "#1D4ED8"   # blue     – input / output nodes
C_PREP    = "#15803D"   # green    – preprocessing
C_BRANCH  = "#B45309"   # amber    – decision / branch
C_LSD     = "#6D28D9"   # violet   – LSD-specific
C_SAM2    = "#0E7490"   # cyan     – SAM2-specific
C_MEAS    = "#B91C1C"   # red      – measurement / classification
C_OUT     = "#1E3A8A"   # indigo   – outputs
C_COMMON  = "#374151"   # slate    – shared steps (density, summary)
BG        = "#F8FAFC"   # page background


# ── Drawing helpers ────────────────────────────────────────────────────────────

def box(ax, cx, cy, w, h, text, color=C_COMMON, fs=7.8, tc="white",
        bold=False, alpha=1.0, style="round,pad=0.15"):
    """Rounded rectangle with centred multiline text."""
    fb = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle=style, linewidth=1.0,
                        edgecolor="#FFFFFF", facecolor=color, alpha=alpha, zorder=2)
    ax.add_patch(fb)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            multialignment="center", zorder=3, linespacing=1.35)


def diamond(ax, cx, cy, w, h, text, color=C_BRANCH, fs=7.8):
    """Diamond shape for decision nodes."""
    pts = np.array([[cx, cy + h / 2], [cx + w / 2, cy],
                    [cx, cy - h / 2], [cx - w / 2, cy]])
    poly = plt.Polygon(pts, closed=True, facecolor=color,
                       edgecolor="#FFFFFF", linewidth=1.0, zorder=2)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold", multialignment="center", zorder=3)


def arr(ax, x1, y1, x2, y2, label=None, lc="#444444", lw=1.4, ls="-"):
    """Arrow from (x1,y1) → (x2,y2) with optional side label."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=lc, lw=lw,
                                connectionstyle="arc3,rad=0.0"),
                zorder=1)
    if label:
        mx, my = (x1 + x2) / 2 + 0.08, (y1 + y2) / 2
        ax.text(mx, my, label, fontsize=6.5, color="#555555", style="italic",
                va="center", zorder=3)


def hconn(ax, x1, y, x2, yend, lc="#444444", lw=1.4):
    """Horizontal then vertical connector (L-shape) ending with arrowhead."""
    ax.plot([x1, x2], [y, y], color=lc, lw=lw, zorder=1)
    ax.annotate("", xy=(x2, yend), xytext=(x2, y),
                arrowprops=dict(arrowstyle="->", color=lc, lw=lw), zorder=1)


def leg_patch(color, label):
    return mpatches.Patch(facecolor=color, edgecolor="white", label=label)


# ── PRIMARY PIPELINE ───────────────────────────────────────────────────────────

def draw_primary():
    fig = plt.figure(figsize=(20, 30))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 30)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)

    W  = 20      # page width
    BW = 14      # common box width
    BH = 1.05   # standard box height
    XC = 10      # horizontal centre

    # thin separator lines between sections
    def sep(y): ax.plot([0.4, W - 0.4], [y, y], color="#CBD5E1", lw=0.6, ls="--", zorder=0)

    # ── Title ──────────────────────────────────────────────────────────────────
    ax.text(XC, 29.2, "primary_measure.py — Full Pipeline",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#1E293B")
    ax.text(XC, 28.6, "particle_type ∈ {acicular, plate}  ×  magnification ∈ {20k, 50k}",
            ha="center", va="center", fontsize=9, color="#475569")

    # ── Input ──────────────────────────────────────────────────────────────────
    box(ax, XC, 27.8, BW, BH,
        "INPUT  —  SEM image (.jpg / .png / .tiff / …)\n"
        "single file  or  batch directory",
        color=C_IO, fs=8.5, bold=True)
    arr(ax, XC, 27.27, XC, 26.7)
    sep(27.0)

    # ── Preprocessing ──────────────────────────────────────────────────────────
    box(ax, XC, 26.2, BW, 0.95,
        "IMAGE NORMALISATION  (--preprocess_width W,  default 1024)\n"
        "resize → W × round(W×1636/2048)  (bilinear)    ·    crop bottom round(W×100/2048) px\n"
        "default:  1024 × 818  →  crop 50 px  →  1024 × 768",
        color=C_PREP, fs=7.5)
    arr(ax, XC, 25.72, XC, 25.15)
    sep(24.9)

    # ── ROI extraction ─────────────────────────────────────────────────────────
    box(ax, XC, 24.4, BW, 1.05,
        "ROI EXTRACTION  (auto_detect_sphere=False, default)\n"
        "center crop  (crop_ratio = 0.60)   —   applies to both 20k and 50k\n"
        "manual override:  --roi_x_min/y_min/x_max/y_max",
        color=C_PREP, fs=7.5)
    arr(ax, XC, 23.75, XC, 23.2)
    sep(22.95)

    # ── Branch decision ────────────────────────────────────────────────────────
    diamond(ax, XC, 22.55, 5.5, 0.9, "measure_mode?", fs=9)
    # LSD label
    ax.text(5.0, 22.55, "lsd", fontsize=8, color="#6D28D9",
            fontweight="bold", va="center", ha="right")
    # SAM2 label
    ax.text(15.0, 22.55, "sam2", fontsize=8, color="#0E7490",
            fontweight="bold", va="center", ha="left")

    # left branch line
    ax.plot([7.25, 4.5], [22.55, 22.55], color="#444", lw=1.4, zorder=1)
    ax.annotate("", xy=(4.5, 21.9), xytext=(4.5, 22.55),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.4), zorder=1)
    # right branch line
    ax.plot([12.75, 15.5], [22.55, 22.55], color="#444", lw=1.4, zorder=1)
    ax.annotate("", xy=(15.5, 21.9), xytext=(15.5, 22.55),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.4), zorder=1)

    sep(22.0)

    # ── Column headers ─────────────────────────────────────────────────────────
    ax.text(4.5, 21.7, "LSD  (acicular)", ha="center", fontsize=9.5,
            fontweight="bold", color=C_LSD)
    ax.text(15.5, 21.7, "SAM2  (plate / acicular hybrid)", ha="center",
            fontsize=9.5, fontweight="bold", color=C_SAM2)
    ax.plot([10, 10], [22.0, 3.4], color="#CBD5E1", lw=1.0, ls=":", zorder=0)

    LX, RX = 4.5, 15.5   # column centres
    CW = 8.4              # column box width
    CH = 0.95             # column box height

    def larr(y1, y2):  arr(ax, LX, y1, LX, y2)
    def rarr(y1, y2):  arr(ax, RX, y1, RX, y2)

    # ── LSD column ─────────────────────────────────────────────────────────────
    y = 21.2
    box(ax, LX, y, CW, CH,
        "CONTRAST ENHANCEMENT\n"
        "CLAHE  (clipLimit=2.0, 8×8 tile)   +   GaussianBlur (3×3)",
        color=C_LSD, fs=7.5); larr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, LX, y, CW, 1.1,
        "THRESHOLDING   (--lsd_adaptive_thresh)\n"
        "Otsu  →  single global threshold  |  Adaptive  →  local Gaussian\n"
        "auto-invert if >55% white (ensures particles = foreground)\n"
        "→  density  =  foreground(particle)_px / total_px  (computed here)",
        color=C_LSD, fs=7.2); larr(y - 0.55, y - 1.15)

    y -= 1.45
    box(ax, LX, y, CW, CH,
        "LSD SEGMENT DETECTION\n"
        "cv2.createLineSegmentDetector()  →  line segments from gradient\n"
        "→  (x1,y1)-(x2,y2), length, angle per segment",
        color=C_LSD, fs=7.5); larr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, LX, y, CW, CH,
        "DEDUP  (sort longest-first)\n"
        "sort candidates by length descending  →  greedy deduplication\n"
        "reject if centre_dist < 12 px  AND  |Δangle| < 25°",
        color=C_LSD, fs=7.5); larr(y - 0.48, y - 1.0)

    y -= 1.48
    box(ax, LX, y, CW, 1.15,
        "PERPENDICULAR THICKNESS\n"
        "7 sample positions along segment  (t = 0.2 … 0.8)\n"
        "scan ⊥ axis  ±  max(40, 2×px/µm)  pixels\n"
        "binary lookup (polarity-normalised)  →  nearest foreground run\n"
        "thickness  =  median of 7 estimates   (discard < 2 px)",
        color=C_LSD, fs=7.2); larr(y - 0.58, y - 1.22)

    y -= 1.52
    box(ax, LX, y, CW, 1.05,
        "CLASSIFICATION  (LSD)\n"
        "AR  =  thickness / length\n"
        "AR < acicular_threshold (0.40)  →  acicular\n"
        "AR ≥ threshold  →  plate\n"
        "mask area < area_threshold  →  discard",
        color=C_MEAS, fs=7.5); larr(y - 0.53, y - 1.05)

    y -= 1.35
    box(ax, LX, y, CW, 1.0,
        "POST-PROCESSING  (optional)\n"
        "--fuse: Union-Find merge of overlapping parallel segments\n"
        "        Δangle < 15°,  overlap ≥ 70%  →  fuse into one\n"
        "--min_length (default 10 px × scale): discard if long_axis < threshold",
        color=C_LSD, fs=7.2)

    LSD_END_Y = y - 0.50

    # ── SAM2 column ────────────────────────────────────────────────────────────
    y = 21.2
    box(ax, RX, y, CW, CH,
        "TILE GRID\n"
        "ROI → overlapping tiles  (tile_size × stride)\n"
        "edge tiles extended to ROI boundary  ·  duplicates removed",
        color=C_SAM2, fs=7.5); rarr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, RX, y, CW, CH,
        "TEXTURE ENHANCEMENT  (per tile)\n"
        "CLAHE  +  Sobel gradient  +  Laplacian  +  morphological blackhat\n"
        "→  single enhanced grayscale channel",
        color=C_SAM2, fs=7.5); rarr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, RX, y, CW, CH,
        "INTEREST-POINT SAMPLING  (Shi-Tomasi)\n"
        "cv2.goodFeaturesToTrack on enhanced tile\n"
        "fallback: Otsu → contour centroids  (if sparse)\n"
        "up to points_per_tile,  spacing ≥ point_min_distance",
        color=C_SAM2, fs=7.5); rarr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, RX, y, CW, CH,
        "SAM2 BATCH INFERENCE  (Ultralytics)\n"
        "foreground point prompts  (label = 1),  batch_size points per call\n"
        "raw logit masks  →  binarise at mask_binarize_threshold",
        color=C_SAM2, fs=7.5); rarr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, RX, y, CW, CH,
        "BBOX IoU DEDUP  (tile-level fast filter)\n"
        "reject mask if bbox IoU ≥ 0.85 with any accepted mask\n"
        "→  promote tile-coord mask → ROI-coord mask",
        color=C_SAM2, fs=7.5); rarr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, RX, y, CW, CH,
        "PIXEL IoU DEDUP  (ROI-level)\n"
        "binary mask IoU ≥ 0.60  →  reject as duplicate",
        color=C_SAM2, fs=7.5); rarr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, RX, y, CW, CH,
        "MASK REFINEMENT  (morphology)\n"
        "open  (int_maskMorphOpenIterations)\n"
        "close  (int_maskMorphCloseIterations)",
        color=C_SAM2, fs=7.5); rarr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, RX, y, CW, 1.05,
        "minAreaRect  +  CLASSIFICATION  (SAM2)\n"
        "thickness = min(rect_W, rect_H)   ·   long_axis = max\n"
        "AR  =  thickness / long_axis\n"
        "area < area_threshold → fragment\n"
        "AR < 0.40 → acicular    ·    AR ≥ 0.40 → plate",
        color=C_MEAS, fs=7.5)

    SAM2_END_Y = y - 0.53

    sep(3.5)

    # ── Merge + density ────────────────────────────────────────────────────────
    MERGE_Y = 3.15
    # lines from branches down to merge
    ax.plot([LX, LX], [LSD_END_Y, MERGE_Y + 0.4], color="#444", lw=1.4, zorder=1)
    ax.plot([RX, RX], [SAM2_END_Y, MERGE_Y + 0.4], color="#444", lw=1.4, zorder=1)
    ax.plot([LX, RX], [MERGE_Y + 0.4, MERGE_Y + 0.4], color="#444", lw=1.4, zorder=1)
    ax.annotate("", xy=(XC, MERGE_Y + 0.05), xytext=(XC, MERGE_Y + 0.4),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.4), zorder=1)

    box(ax, XC, MERGE_Y - 0.32, BW, 0.95,
        "ROI DENSITY\n"
        "LSD mode: density = white_px / total_px  (from binary already computed)\n"
        "SAM2 mode: Otsu-binarise ROI grey  →  density = white_px / total_px",
        color=C_COMMON, fs=7.8)
    arr(ax, XC, MERGE_Y - 0.80, XC, MERGE_Y - 1.35)
    sep(1.55)

    # ── Outputs ────────────────────────────────────────────────────────────────
    box(ax, XC, 1.0, BW, 0.9,
        "OUTPUTS  (per image)\n"
        "01_input.png  02_input_roi.png  03_overlay_roi.png  04_overlay_full.png\n"
        "lsd_01…06_*.png  |  05_opencv_candidates.png  |  06_sphere_detection.png\n"
        "objects.csv  acicular.csv / plate.csv  |  thickness_dist.png\n"
        "summary.json  (roi_density)   objects.json   debug.json\n"
        "Batch:  img_id_summary.json  ·  batch_summary.json  (roi_density_mean)",
        color=C_OUT, fs=7.5, bold=True)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_patches = [
        leg_patch(C_IO,     "Input / Output"),
        leg_patch(C_PREP,   "Preprocessing / ROI"),
        leg_patch(C_BRANCH, "Decision / Branch"),
        leg_patch(C_LSD,    "LSD-specific"),
        leg_patch(C_SAM2,   "SAM2-specific"),
        leg_patch(C_MEAS,   "Measurement / Classification"),
        leg_patch(C_COMMON, "Shared (both modes)"),
        leg_patch(C_OUT,    "Output"),
    ]
    ax.legend(handles=legend_patches, loc="lower left",
              bbox_to_anchor=(0.01, 0.00), fontsize=7.5,
              framealpha=0.9, edgecolor="#CBD5E1", ncol=4,
              handlelength=1.2, handleheight=0.8, columnspacing=1.0)

    return fig


# ── SECONDARY PIPELINE ─────────────────────────────────────────────────────────

def draw_secondary():
    fig = plt.figure(figsize=(14, 28))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 28)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)

    XC = 7
    BW = 12.8

    def sep(y): ax.plot([0.3, 13.7], [y, y], color="#CBD5E1", lw=0.6, ls="--", zorder=0)
    def dn(y1, y2): arr(ax, XC, y1, XC, y2)

    # Title
    ax.text(XC, 27.55, "secondary_measure.py — 파이프라인",
            ha="center", fontsize=14, fontweight="bold", color="#1E293B")
    ax.text(XC, 27.05, "HCT 포인트 프롬프트  →  SAM2  →  Kasa 복원  →  hull 병합  →  S / S' / eq_d 측정",
            ha="center", fontsize=8.5, color="#475569")
    sep(26.75)

    # ── INPUT ─────────────────────────────────────────────────────────────────
    box(ax, XC, 26.3, BW, 0.75,
        "INPUT  —  SEM image  (single file  or  batch directory)",
        color=C_IO, fs=8.5, bold=True)
    dn(25.92, 25.42); sep(25.22)

    # ── PREPROCESSING ─────────────────────────────────────────────────────────
    box(ax, XC, 24.77, BW, 1.0,
        "전처리  (IMAGE NORMALISATION)\n"
        "resize → preprocess_width W (default 1024) × round(W×1636/2048)  [bilinear]\n"
        "bottom crop: round(W×100/2048) px  스케일바 제거  →  default 1024 × 768",
        color=C_PREP, fs=7.8)
    dn(24.27, 23.77); sep(23.57)

    # ── ROI ───────────────────────────────────────────────────────────────────
    box(ax, XC, 23.22, BW, 0.85,
        "ROI 추출\n"
        "--roi_x_min / y_min / x_max / y_max  직접 지정  (default: 전체 이미지)\n"
        "또는  auto_center_crop 활성 시 center crop",
        color=C_PREP, fs=7.8)
    dn(22.79, 22.29); sep(22.09)

    # ── TILE GRID ─────────────────────────────────────────────────────────────
    box(ax, XC, 21.69, BW, 0.8,
        "타일 분할  (SLIDING WINDOW TILE GRID)\n"
        "ROI → 겹치는 tile grid  (tile_size=512, stride=256)  ·  경계 타일 확장  ·  중복 제거",
        color=C_SAM2, fs=7.8)
    dn(21.29, 20.79)

    # ── HCT PROMPTS ───────────────────────────────────────────────────────────
    box(ax, XC, 19.59, BW, 2.2,
        "포인트 프롬프트 추출  detect_hct_prompts()  [per tile]\n\n"
        "① HCT  (primary)\n"
        "     GaussBlur(9×9) → HoughCircles → 각 원 중심 좌표\n"
        "     밝기 검증: 중심 주변 patch 평균 < Otsu×0.85 → 배경 갭 오탐 제거\n"
        "② fallback  (HCT 검출 0개 시)\n"
        "     Canny → MORPH_CLOSE(7×7,×3) → RETR_CCOMP hole fill → 거리변환 peak\n"
        "③ 미커버 전경 보완\n"
        "     fg AND NOT circles_mask → connectedComponents → 블롭별 거리변환 peak\n"
        "④ 네거티브  ~fg → erode(×4) → linspace N개 균등 샘플링",
        color=C_SAM2, fs=7.5)
    dn(18.49, 17.99)

    # ── SAM2 ──────────────────────────────────────────────────────────────────
    box(ax, XC, 17.54, BW, 0.85,
        "SAM2 배치 추론  (Ultralytics)\n"
        "포지티브 (label=1) + 네거티브 (label=0)  ·  batch_size=32\n"
        "raw logit mask → binarize  (mask_binarize_threshold=0.0)",
        color=C_SAM2, fs=7.8)
    dn(17.11, 16.61); sep(16.41)

    # ── DEDUP ─────────────────────────────────────────────────────────────────
    box(ax, XC, 15.96, BW, 1.0,
        "마스크 중복 제거  (DEDUP)\n"
        "① bbox IoU ≥ 0.85 → reject  [tile-level fast filter]\n"
        "② binary mask IoU ≥ 0.60 → reject  [ROI-level]\n"
        "③ 포함 관계: inter / small_area ≥ 0.75 → reject  [ROI-level]",
        color=C_SAM2, fs=7.8)
    dn(15.46, 14.96)

    # ── POSTPROCESS ───────────────────────────────────────────────────────────
    box(ax, XC, 14.51, BW, 0.9,
        "마스크 후처리  (POSTPROCESSING)\n"
        "① Smooth: MORPH_CLOSE(5×5,×2) + OPEN(5×5,×1) → 최대 연결 컴포넌트만 유지\n"
        "② Peanut split: minAreaRect AR < 0.6 → 거리변환 2-peak → Watershed 분리",
        color=C_SAM2, fs=7.8)
    dn(14.06, 13.56); sep(13.36)

    # ── PUNCH-OUT ─────────────────────────────────────────────────────────────
    box(ax, XC, 12.96, BW, 0.8,
        "포함 마스크 펀치아웃  (PUNCH-OUT)\n"
        "작은 마스크 J가 큰 마스크 I에 97%+ 포함 → I에서 J 픽셀 제거\n"
        "→ 입자 테두리만 남겨 밝기 필터가 선·배경을 정확히 판별하게 함",
        color=C_SAM2, fs=7.8)
    dn(12.56, 12.06)

    # ── BRIGHTNESS FILTER ─────────────────────────────────────────────────────
    box(ax, XC, 11.66, BW, 0.8,
        "밝기 필터  (BRIGHTNESS FILTER)\n"
        "전체 ROI Otsu → brightness_thresh = Otsu × 0.5\n"
        "마스크 영역 평균 밝기 < threshold → 배경 / SEM 선 아티팩트 → 제거",
        color=C_PREP, fs=7.8)
    dn(11.26, 10.76); sep(10.56)

    # ── KASA RESTORATION ──────────────────────────────────────────────────────
    box(ax, XC, 9.96, BW, 1.4,
        "부분 절단 입자 복원  (KASA CIRCLE FITTING)\n"
        "조건: solidity < 0.97  OR  수평/수직 직선 구간 ≥15px (approxPolyDP)  OR  ROI 경계 접촉\n"
        "→ convex hull 포인트에 Kasa 최소제곱 원 피팅  (cx, cy, r)\n"
        "     검증: solidity ≥ 0.5  ·  CV(hull→원 거리) ≤ 10%  ·  면적비 이상 없음\n"
        "→ 피팅 원 내부 중 밝은 픽셀 (≥ Otsu×0.75) 을 마스크에 추가 복원",
        color=C_MEAS, fs=7.8)
    dn(9.26, 8.76)

    # ── HULL MASK ─────────────────────────────────────────────────────────────
    box(ax, XC, 8.36, BW, 0.8,
        "Hull 마스크 적용\n"
        "모든 마스크에 convex hull fill 적용\n"
        "→ 이후 면적·크기·S'·분류 모두 hull 기준    (S=원형도는 원본 컨투어 값 유지)",
        color=C_MEAS, fs=7.8)
    dn(7.96, 7.46)

    # ── HULL MERGE ────────────────────────────────────────────────────────────
    box(ax, XC, 7.01, BW, 0.85,
        "Hull 마스크 97%+ 겹침 병합  (UNION-FIND)\n"
        "inter / min_area ≥ 0.97 → Union-Find로 그룹화\n"
        "같은 그룹 → 합집합(OR) 마스크로 병합 후 재측정",
        color=C_MEAS, fs=7.8)
    dn(6.58, 6.08); sep(5.88)

    # ── MEASUREMENT ───────────────────────────────────────────────────────────
    box(ax, XC, 5.43, BW, 1.1,
        "측정  (MEASUREMENT)\n"
        "S   = 4π × hull_area / hull_perimeter²           [원형도 Wadell 2D]  ←  원본 컨투어 hull\n"
        "S'  = b / a  (cv2.fitEllipse 단축 / 장축비)       [타원도]  ←  hull 마스크 컨투어\n"
        "eq_diameter = 2 × √(hull_area / π)  →  × (µm/px)  [등가원 지름]",
        color=C_MEAS, fs=7.8)
    dn(4.88, 4.38)

    # ── CLASSIFICATION ────────────────────────────────────────────────────────
    box(ax, XC, 3.98, BW, 0.8,
        "분류  (CLASSIFICATION)\n"
        "hull_area ≥ area_threshold (1500 px²) → particle\n"
        "hull_area <  area_threshold             → fragment",
        color=C_MEAS, fs=7.8)
    dn(3.58, 3.08); sep(2.88)

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    box(ax, XC, 2.43, BW, 0.9,
        "출력  (OUTPUT)\n"
        "input_roi.png  ·  classified.png  ·  overlay_roi.png  ·  summary.json\n"
        "[--debug]  overlay_S/Sp.png  tiles.png  prompts.png  objects.csv  size/sph_dist.png\n"
        "Batch:  img_id_summary.json  ·  batch_summary.json  ·  히스토그램 PNG",
        color=C_OUT, fs=7.8, bold=True)

    # Legend
    legend_patches = [
        leg_patch(C_IO,     "Input / Output"),
        leg_patch(C_PREP,   "전처리 / ROI / 밝기 필터"),
        leg_patch(C_SAM2,   "HCT / SAM2 / Dedup / Postprocess"),
        leg_patch(C_MEAS,   "Kasa 복원 / Hull / 측정 / 분류"),
        leg_patch(C_OUT,    "Output"),
    ]
    ax.legend(handles=legend_patches, loc="lower left",
              bbox_to_anchor=(0.01, 0.005), fontsize=8,
              framealpha=0.9, edgecolor="#CBD5E1", ncol=3,
              handlelength=1.2, handleheight=0.8)

    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating pipeline_primary.pdf …")
    with PdfPages("pipeline_primary.pdf") as pdf:
        fig = draw_primary()
        pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    print("  → pipeline_primary.pdf")

    print("Generating pipeline_secondary.pdf …")
    with PdfPages("pipeline_secondary.pdf") as pdf:
        fig = draw_secondary()
        pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    print("  → pipeline_secondary.pdf")

    print("Done.")
