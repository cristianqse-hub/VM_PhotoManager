#!/usr/bin/env python3
"""
Standalone test script for 1xN row stitching from Group/.

Default mode (recommended):
- Uses all images in sorted order.
- Estimates pairwise offsets with template matching in expected overlap range.
- Blends with feather weights to reduce seams.

Optional:
- OpenCV Stitcher comparison mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import List, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError as exc:
    print(
        "[ERR] OpenCV no instalado. Ejecuta:\n"
        "      pip install opencv-contrib-python\n"
        f"Detalle: {exc}"
    )
    sys.exit(1)


def natural_key(path: Path) -> list[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def list_images(input_dir: Path) -> list[Path]:
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=natural_key)
    return files


def load_images(paths: list[Path], max_dim: int) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WRN] No se pudo leer: {p}")
            continue
        img = np.ascontiguousarray(img.astype(np.uint8))

        if max_dim > 0:
            h, w = img.shape[:2]
            largest = max(h, w)
            if largest > max_dim:
                scale = max_dim / float(largest)
                img = cv2.resize(
                    img,
                    (max(64, int(w * scale)), max(64, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
        images.append(img)
    return images


def pair_offset(
    left_img: np.ndarray,
    right_img: np.ndarray,
    overlap: float,
    overlap_min: float,
    overlap_max: float,
) -> Tuple[int, int, float]:
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    h = min(left_gray.shape[0], right_gray.shape[0])
    w_left = left_gray.shape[1]
    w_right = right_gray.shape[1]
    w_ref = min(w_left, w_right)

    min_ov = max(8, int(w_ref * overlap_min))
    max_ov = max(min_ov + 8, int(w_ref * overlap_max))
    exp_ov = int(w_ref * overlap)

    # Template from left side of right image (where overlap should be).
    tpl_w = max(32, min(int(w_ref * 0.12), min_ov))
    tpl_h = max(32, int(h * 0.7))
    y0_tpl = max(0, (right_gray.shape[0] - tpl_h) // 2)
    template = right_gray[y0_tpl : y0_tpl + tpl_h, 0:tpl_w]

    # Search only in right side of left image.
    x0_search = max(0, w_left - max_ov)
    x1_search = min(w_left, w_left - min_ov + tpl_w)
    if x1_search <= x0_search + tpl_w:
        x0_search = max(0, w_left - max_ov - tpl_w)
        x1_search = w_left
    search = left_gray[:, x0_search:x1_search]

    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    match_x = x0_search + max_loc[0]
    match_y = max_loc[1]

    dx = match_x
    dy = match_y - y0_tpl

    # Guardrails to avoid obvious jumps.
    dx_min = w_left - max_ov
    dx_max = w_left - min_ov
    if dx < dx_min or dx > dx_max or max_val < 0.15:
        dx = int(w_left - exp_ov)
        dy = 0

    dy_limit = int(h * 0.1)
    dy = int(np.clip(dy, -dy_limit, dy_limit))
    return int(dx), int(dy), float(max_val)


def compute_positions(
    images: List[np.ndarray],
    overlap: float,
    overlap_min: float,
    overlap_max: float,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    positions: List[Tuple[int, int]] = [(0, 0)]
    scores: List[float] = []

    for i in range(1, len(images)):
        prev = images[i - 1]
        curr = images[i]
        dx, dy, score = pair_offset(prev, curr, overlap, overlap_min, overlap_max)
        x_prev, y_prev = positions[-1]
        positions.append((x_prev + dx, y_prev + dy))
        scores.append(score)
    return positions, scores


def feather_mask(h: int, w: int, feather_x_ratio: float, feather_y_ratio: float) -> np.ndarray:
    fx = max(2, int(w * feather_x_ratio))
    fy = max(2, int(h * feather_y_ratio))

    wx = np.ones((w,), dtype=np.float32)
    wy = np.ones((h,), dtype=np.float32)
    ramp_x = np.linspace(0.05, 1.0, fx, dtype=np.float32)
    ramp_y = np.linspace(0.1, 1.0, fy, dtype=np.float32)

    wx[:fx] = np.minimum(wx[:fx], ramp_x)
    wx[-fx:] = np.minimum(wx[-fx:], ramp_x[::-1])
    wy[:fy] = np.minimum(wy[:fy], ramp_y)
    wy[-fy:] = np.minimum(wy[-fy:], ramp_y[::-1])

    return wy[:, None] * wx[None, :]


def linear_compose(
    images: List[np.ndarray],
    positions: List[Tuple[int, int]],
    feather_x_ratio: float,
    feather_y_ratio: float,
) -> np.ndarray:
    min_x = min(x for x, _ in positions)
    min_y = min(y for _, y in positions)
    shifted = [(x - min_x, y - min_y) for x, y in positions]

    max_x = max(x + img.shape[1] for (x, y), img in zip(shifted, images))
    max_y = max(y + img.shape[0] for (x, y), img in zip(shifted, images))

    acc = np.zeros((max_y, max_x, 3), dtype=np.float32)
    wgt = np.zeros((max_y, max_x), dtype=np.float32)

    for (x, y), img in zip(shifted, images):
        h, w = img.shape[:2]
        mask = feather_mask(h, w, feather_x_ratio, feather_y_ratio)
        acc[y : y + h, x : x + w] += img.astype(np.float32) * mask[:, :, None]
        wgt[y : y + h, x : x + w] += mask

    out = np.zeros_like(acc, dtype=np.uint8)
    valid = wgt > 1e-6
    out[valid] = (acc[valid] / wgt[valid, None]).clip(0, 255).astype(np.uint8)
    return out


def crop_black_borders(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y : y + h, x : x + w]


def stitcher_mode(images: List[np.ndarray]) -> np.ndarray:
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    if hasattr(stitcher, "setPanoConfidenceThresh"):
        stitcher.setPanoConfidenceThresh(0.5)
    if hasattr(stitcher, "setWaveCorrection"):
        stitcher.setWaveCorrection(False)
    status, pano = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"Stitcher fallo con status={status}")
    return pano


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose 1xN overlapped images from Group/")
    parser.add_argument("--input", default="Group", help="Carpeta de entrada")
    parser.add_argument("--output", default="Group/composite_result.png", help="Imagen de salida")
    parser.add_argument("--max-dim", type=int, default=0, help="Limite lado mayor por imagen (0=sin resize)")
    parser.add_argument("--overlap", type=float, default=0.20, help="Solape esperado (0..1)")
    parser.add_argument("--overlap-min", type=float, default=0.08, help="Solape minimo permitido (0..1)")
    parser.add_argument("--overlap-max", type=float, default=0.45, help="Solape maximo permitido (0..1)")
    parser.add_argument("--feather-x", type=float, default=0.08, help="Feather horizontal (0..1)")
    parser.add_argument("--feather-y", type=float, default=0.03, help="Feather vertical (0..1)")
    parser.add_argument("--mode", choices=["linear", "stitcher"], default="linear")
    parser.add_argument("--no-crop", action="store_true", help="No recortar bordes negros")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.exists():
        print(f"[ERR] No existe carpeta de entrada: {input_dir}")
        return 1

    image_paths = list_images(input_dir)
    if len(image_paths) < 2:
        print("[ERR] Se necesitan al menos 2 imagenes.")
        return 1

    print(f"[INF] Imagenes detectadas: {len(image_paths)}")
    images = load_images(image_paths, max_dim=args.max_dim)
    if len(images) != len(image_paths):
        print(f"[WRN] Se cargaron {len(images)}/{len(image_paths)} imagenes.")
    if len(images) < 2:
        print("[ERR] No hay suficientes imagenes validas.")
        return 1

    if args.mode == "stitcher":
        pano = stitcher_mode(images)
    else:
        positions, scores = compute_positions(
            images=images,
            overlap=args.overlap,
            overlap_min=args.overlap_min,
            overlap_max=args.overlap_max,
        )
        print(
            f"[INF] Offsets calculados para {len(scores)} pares. "
            f"score medio={float(np.mean(scores)):.3f}"
        )
        pano = linear_compose(
            images=images,
            positions=positions,
            feather_x_ratio=args.feather_x,
            feather_y_ratio=args.feather_y,
        )

    if not args.no_crop:
        pano = crop_black_borders(pano)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), pano)
    if not ok:
        print(f"[ERR] No se pudo guardar: {output_path}")
        return 1

    print(f"[INF] Composicion completada. Archivo: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
