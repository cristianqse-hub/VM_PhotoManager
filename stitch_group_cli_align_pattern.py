#!/usr/bin/env python3
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
        "      python -m pip install opencv-python\n"
        f"Detalle: {exc}"
    )
    sys.exit(1)


VALID_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_OVERLAP = 0.20
DEFAULT_OVERLAP_MIN = 0.08
DEFAULT_OVERLAP_MAX = 0.45
DEFAULT_FEATHER_X = 0.08
DEFAULT_FEATHER_Y = 0.03
ALIGN_PATTERN_THRESHOLD = 0.75
ALIGN_PATTERN_COUNT = 2


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "si", "sí", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Valor booleano no válido. Usa true/false, yes/no, 1/0."
    )


def natural_key(path: Path) -> list[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def list_images(input_dir: Path, image_id: str | None = None) -> list[Path]:
    files = [
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]
    if image_id:
        files = [p for p in files if p.name.startswith(image_id)]
    files.sort(key=natural_key)
    return files


def load_images(paths: list[Path]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WRN] No se pudo leer: {p}")
            continue
        images.append(np.ascontiguousarray(img.astype(np.uint8)))
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

    tpl_w = max(32, min(int(w_ref * 0.12), min_ov))
    tpl_h = max(32, int(h * 0.7))
    y0_tpl = max(0, (right_gray.shape[0] - tpl_h) // 2)
    template = right_gray[y0_tpl : y0_tpl + tpl_h, 0:tpl_w]

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

    dx_min = w_left - max_ov
    dx_max = w_left - min_ov
    if dx < dx_min or dx > dx_max or max_val < 0.15:
        dx = int(w_left - exp_ov)
        dy = 0

    dy_limit = int(h * 0.1)
    dy = int(np.clip(dy, -dy_limit, dy_limit))
    return int(dx), int(dy), float(max_val)


def compute_positions(images: List[np.ndarray]) -> Tuple[List[Tuple[int, int]], List[float]]:
    positions: List[Tuple[int, int]] = [(0, 0)]
    scores: List[float] = []

    for i in range(1, len(images)):
        dx, dy, score = pair_offset(
            images[i - 1],
            images[i],
            DEFAULT_OVERLAP,
            DEFAULT_OVERLAP_MIN,
            DEFAULT_OVERLAP_MAX,
        )
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


def linear_compose(images: List[np.ndarray], positions: List[Tuple[int, int]]) -> np.ndarray:
    min_x = min(x for x, _ in positions)
    min_y = min(y for _, y in positions)
    shifted = [(x - min_x, y - min_y) for x, y in positions]

    max_x = max(x + img.shape[1] for (x, y), img in zip(shifted, images))
    max_y = max(y + img.shape[0] for (x, y), img in zip(shifted, images))

    acc = np.zeros((max_y, max_x, 3), dtype=np.float32)
    wgt = np.zeros((max_y, max_x), dtype=np.float32)

    for (x, y), img in zip(shifted, images):
        h, w = img.shape[:2]
        mask = feather_mask(h, w, DEFAULT_FEATHER_X, DEFAULT_FEATHER_Y)
        acc[y : y + h, x : x + w] += img.astype(np.float32) * mask[:, :, None]
        wgt[y : y + h, x : x + w] += mask

    out = np.zeros_like(acc, dtype=np.uint8)
    valid = wgt > 1e-6
    out[valid] = (acc[valid] / wgt[valid, None]).clip(0, 255).astype(np.uint8)
    return out


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-12:
        return image

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos_v = abs(matrix[0, 0])
    sin_v = abs(matrix[0, 1])
    new_w = int((h * sin_v) + (w * cos_v))
    new_h = int((h * cos_v) + (w * sin_v))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image, matrix, (new_w, new_h))


def crop_black_borders(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y : y + h, x : x + w]


def _non_max_suppress(result: np.ndarray, x: int, y: int, tpl_w: int, tpl_h: int) -> None:
    x0 = max(0, x - tpl_w // 2)
    y0 = max(0, y - tpl_h // 2)
    x1 = min(result.shape[1], x + tpl_w // 2)
    y1 = min(result.shape[0], y + tpl_h // 2)
    result[y0:y1, x0:x1] = -1.0


def find_pattern_matches(
    image: np.ndarray,
    pattern: np.ndarray,
    threshold: float = ALIGN_PATTERN_THRESHOLD,
    expected_count: int = ALIGN_PATTERN_COUNT,
) -> list[tuple[float, tuple[float, float], tuple[int, int]]]:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern_gray = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

    img_h, img_w = image_gray.shape[:2]
    pat_h, pat_w = pattern_gray.shape[:2]
    if pat_h > img_h or pat_w > img_w:
        raise ValueError(
            "El patrón es más grande que la imagen compuesta; no se puede alinear por patrón."
        )

    result = cv2.matchTemplate(image_gray, pattern_gray, cv2.TM_CCOEFF_NORMED)
    work = result.copy()
    matches: list[tuple[float, tuple[float, float], tuple[int, int]]] = []

    for _ in range(expected_count):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val < threshold:
            break
        top_left = (int(max_loc[0]), int(max_loc[1]))
        center = (top_left[0] + pat_w / 2.0, top_left[1] + pat_h / 2.0)
        matches.append((float(max_val), center, top_left))
        _non_max_suppress(work, top_left[0], top_left[1], pat_w, pat_h)

    return matches


def compute_pattern_alignment_angle(
    image: np.ndarray,
    pattern: np.ndarray,
    threshold: float = ALIGN_PATTERN_THRESHOLD,
    expected_count: int = ALIGN_PATTERN_COUNT,
) -> tuple[float, list[tuple[float, float]], list[float]]:
    matches = find_pattern_matches(
        image=image,
        pattern=pattern,
        threshold=threshold,
        expected_count=expected_count,
    )
    if len(matches) != expected_count:
        raise ValueError(
            "No se encontraron exactamente 2 patrones válidos con "
            f"threshold={threshold:.2f}. Encontrados: {len(matches)}"
        )

    matches.sort(key=lambda item: item[1][0])
    centers = [item[1] for item in matches]
    scores = [item[0] for item in matches]

    (x1, y1), (x2, y2) = centers
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        raise ValueError("Los 2 patrones detectados tienen el mismo centro; no se puede calcular ángulo.")

    angle_deg = -float(np.degrees(np.arctan2(dy, dx)))
    return angle_deg, centers, scores


def delete_used_images(paths: list[Path]) -> None:
    for p in paths:
        try:
            p.unlink()
            print(f"[INF] Eliminada imagen parcial: {p}")
        except FileNotFoundError:
            print(f"[WRN] No existía al intentar borrar: {p}")
        except PermissionError:
            print(f"[WRN] Sin permisos para borrar: {p}")
        except OSError as exc:
            print(f"[WRN] No se pudo borrar {p}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compone imágenes solapadas 1xN desde una carpeta. "
            "Opcionalmente corrige la inclinación global con una pendiente "
            "o alineando 2 patrones iguales dentro de la composición."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Carpeta donde están las imágenes")
    parser.add_argument(
        "--id",
        dest="image_id",
        default=None,
        help=(
            "Prefijo/ID del nombre de archivo. Si se indica, solo se usan "
            "las imágenes cuyo nombre empiece por ese ID."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta completa de salida, incluyendo nombre del fichero final",
    )
    parser.add_argument(
        "--delete-partials",
        type=str2bool,
        default=True,
        help="Borrar imágenes parciales usadas tras generar la final (default: true)",
    )
    parser.add_argument(
        "--slope",
        type=float,
        default=0.0,
        help=(
            "Pendiente global dy/dx para compensar inclinación. "
            "Ejemplo: 0.02 o -0.015. Default: 0.0"
        ),
    )
    parser.add_argument(
        "--align-pattern",
        dest="align_pattern",
        default=None,
        help=(
            "Ruta de una imagen patrón que debe aparecer exactamente 2 veces en la "
            "composición final. Se usa matchTemplate con threshold fijo 0.75 para "
            "detectar ambos patrones y calcular el ángulo a partir de sus centros."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    align_pattern_path = (
        Path(args.align_pattern).expanduser().resolve() if args.align_pattern else None
    )

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERR] No existe la carpeta de entrada: {input_dir}")
        return 1

    if align_pattern_path is not None and not align_pattern_path.is_file():
        print(f"[ERR] No existe la imagen de patrón: {align_pattern_path}")
        return 1

    image_paths = list_images(input_dir, image_id=args.image_id)
    if not image_paths:
        if args.image_id:
            print(
                f"[ERR] No se encontraron imágenes en {input_dir} "
                f"cuyo nombre empiece por '{args.image_id}'."
            )
        else:
            print(f"[ERR] No se encontraron imágenes válidas en {input_dir}.")
        return 1

    if len(image_paths) < 2:
        print("[ERR] Se necesitan al menos 2 imágenes para componer la imagen final.")
        return 1

    print(f"[INF] Carpeta de entrada: {input_dir}")
    print(f"[INF] Imágenes seleccionadas: {len(image_paths)}")
    if args.image_id:
        print(f"[INF] Filtro por ID/prefijo: {args.image_id}")
    if align_pattern_path is not None:
        print(
            f"[INF] Corrigiendo inclinación con patrón={align_pattern_path} "
            f"(n={ALIGN_PATTERN_COUNT}, threshold={ALIGN_PATTERN_THRESHOLD:.2f})"
        )
    elif abs(args.slope) > 1e-12:
        print(f"[INF] Corrigiendo inclinación con slope={args.slope}")
    for p in image_paths:
        print(f"       - {p.name}")

    images = load_images(image_paths)
    if len(images) != len(image_paths):
        print(f"[WRN] Se cargaron {len(images)}/{len(image_paths)} imágenes.")
    if len(images) < 2:
        print("[ERR] No hay suficientes imágenes válidas tras la carga.")
        return 1

    try:
        positions, scores = compute_positions(images)
        print(
            f"[INF] Offsets calculados para {len(scores)} pares. "
            f"score medio={float(np.mean(scores)):.3f}"
        )
        pano = linear_compose(images=images, positions=positions)

        if align_pattern_path is not None:
            pattern = cv2.imread(str(align_pattern_path), cv2.IMREAD_COLOR)
            if pattern is None:
                print(f"[ERR] No se pudo leer la imagen de patrón: {align_pattern_path}")
                return 1
            angle_deg, centers, match_scores = compute_pattern_alignment_angle(
                image=pano,
                pattern=pattern,
                threshold=ALIGN_PATTERN_THRESHOLD,
                expected_count=ALIGN_PATTERN_COUNT,
            )
            print(
                "[INF] Patrones detectados en centros: "
                f"({centers[0][0]:.1f}, {centers[0][1]:.1f}) y "
                f"({centers[1][0]:.1f}, {centers[1][1]:.1f})"
            )
            print(
                "[INF] Scores de match del patrón: "
                f"{match_scores[0]:.3f}, {match_scores[1]:.3f}"
            )
            print(f"[INF] Ángulo de corrección aplicado por patrón: {angle_deg:.4f} grados")
            pano = rotate_image(pano, angle_deg)
        elif abs(args.slope) > 1e-12:
            angle_deg = -np.degrees(np.arctan(args.slope))
            print(f"[INF] Ángulo de corrección aplicado por slope: {angle_deg:.4f} grados")
            pano = rotate_image(pano, angle_deg)

        pano = crop_black_borders(pano)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), pano)
        if not ok:
            print(f"[ERR] No se pudo guardar: {output_path}")
            return 1

        print(f"[INF] Composición completada. Archivo final: {output_path}")

        if args.delete_partials:
            delete_used_images(image_paths)
        else:
            print("[INF] Conservando imágenes parciales usadas.")
        return 0

    except Exception as exc:
        print(f"[ERR] Falló la composición: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
