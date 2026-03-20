from __future__ import annotations

from dataclasses import dataclass, field
import json
import multiprocessing as mp
from pathlib import Path
import re
from typing import Any

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

from stitch_group_cli_align_pattern import (
    ALIGN_PATTERN_COUNT,
    ALIGN_PATTERN_THRESHOLD,
    compute_pattern_alignment_angle,
    compute_positions,
    crop_black_borders,
    linear_compose,
    load_images,
    rotate_image,
)


TEMPS_DIR = Path("Temps")
COMMANDS_DIR = TEMPS_DIR / "Commands"
PHOTOS_DIR = TEMPS_DIR / "Photos"
RUNTIME_STATE_PATH = TEMPS_DIR / "runtime_state.json"
VALID_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass
class ComposeObject:
    object_id: str
    output_name: str = ""
    align_pattern: str = ""
    slope: float = 0.0
    delete_partials: bool = False
    enabled: bool = True
    last_status: str = "idle"
    last_error: str = ""
    last_built_signature: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def resolved_output(self) -> Path | None:
        return Path(self.output_name.strip()) if self.output_name.strip() else None

    def has_minimum_attrs(self) -> bool:
        return self.resolved_output() is not None


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "si", "sí", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Booleano no valido: {value}")


def parse_command_file(command_path: Path) -> dict[str, str]:
    params: dict[str, str] = {}
    text = command_path.read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            params[key] = value
    return params


def apply_params(obj: ComposeObject, params: dict[str, str]) -> None:
    for key, value in params.items():
        key_upper = key.strip().upper()
        if key_upper == "ID":
            continue
        if key_upper == "OUTPUT_NAME":
            obj.output_name = value
        elif key_upper == "ALIGN_PATTERN":
            obj.align_pattern = value
        elif key_upper == "SLOPE":
            try:
                obj.slope = float(value)
            except ValueError:
                print(f"[WRN] SLOPE invalido para ID={obj.object_id}: {value}")
        elif key_upper == "DELETE_PARTIALS":
            try:
                obj.delete_partials = parse_bool(value)
            except ValueError as exc:
                print(f"[WRN] {exc}")
        elif key_upper == "ENABLED":
            try:
                obj.enabled = parse_bool(value)
            except ValueError as exc:
                print(f"[WRN] {exc}")
        else:
            obj.extra[key] = value


def parse_photo_filename(file_name: str) -> dict[str, Any] | None:
    pattern = (
        r"^(?P<id>\d{8}_\d{6,9})Img_?"
        r"(?P<cols>\d+)x(?P<rows>\d+)_"
        r"(?P<i>\d+)-(?P<j>\d+)_.*\.(?P<ext>bmp|png|jpg|jpeg|tif|tiff)$"
    )
    match = re.match(pattern, file_name, flags=re.IGNORECASE)
    if not match:
        return None
    data = match.groupdict()
    return {
        "id": data["id"],
        "cols": int(data["cols"]),
        "rows": int(data["rows"]),
        "i": int(data["i"]),
        "j": int(data["j"]),
        "name": file_name,
    }


def collect_photo_sets() -> dict[str, dict[str, Any]]:
    sets: dict[str, dict[str, Any]] = {}
    if not PHOTOS_DIR.exists():
        return sets

    files = sorted([p for p in PHOTOS_DIR.iterdir() if p.is_file()], key=lambda p: p.name.lower())
    for file_path in files:
        if file_path.suffix.lower() not in VALID_EXTS:
            continue
        parsed = parse_photo_filename(file_path.name)
        if parsed is None:
            continue

        set_id = parsed["id"]
        item = sets.setdefault(
            set_id,
            {
                "id": set_id,
                "cols": parsed["cols"],
                "rows": parsed["rows"],
                "coords": set(),
                "files_by_coord": {},
                "invalid_dimensions": False,
            },
        )

        if item["cols"] != parsed["cols"] or item["rows"] != parsed["rows"]:
            item["invalid_dimensions"] = True
            item["cols"] = max(item["cols"], parsed["cols"])
            item["rows"] = max(item["rows"], parsed["rows"])

        coord = (parsed["i"], parsed["j"])
        item["coords"].add(coord)
        if coord not in item["files_by_coord"]:
            item["files_by_coord"][coord] = []
        item["files_by_coord"][coord].append(file_path)

    for item in sets.values():
        for coord_files in item["files_by_coord"].values():
            coord_files.sort(key=lambda p: p.name.lower())

    return sets


def list_ordered_set_images(set_data: dict[str, Any]) -> list[Path] | None:
    cols = set_data["cols"]
    rows = set_data["rows"]
    files_by_coord = set_data["files_by_coord"]
    ordered: list[Path] = []
    for j in range(1, rows + 1):
        for i in range(1, cols + 1):
            coord = (i, j)
            if coord not in files_by_coord or not files_by_coord[coord]:
                return None
            ordered.append(files_by_coord[coord][0])
    return ordered


def maybe_generate_composite(obj: ComposeObject, photo_sets: dict[str, dict[str, Any]]) -> bool:
    if not obj.enabled:
        obj.last_status = "disabled"
        return False

    if cv2 is None:
        obj.last_status = "error"
        obj.last_error = "OpenCV no disponible"
        return False

    set_data = photo_sets.get(obj.object_id)
    if set_data is None:
        obj.last_status = "waiting_photos"
        return False

    cols = set_data["cols"]
    rows = set_data["rows"]
    expected = cols * rows
    found = len(set_data["coords"])

    if set_data.get("invalid_dimensions"):
        obj.last_status = "invalid_set"
        obj.last_error = "Dimensiones cols/rows inconsistentes en nombres de imagen."
        return False

    if found < expected:
        obj.last_status = f"incomplete_set:{found}/{expected}"
        return False

    if not obj.has_minimum_attrs():
        obj.last_status = "missing_attrs"
        return False

    ordered_paths = list_ordered_set_images(set_data)
    if ordered_paths is None:
        obj.last_status = "incomplete_set"
        return False

    signature = "|".join([p.name for p in ordered_paths])
    if signature == obj.last_built_signature and obj.resolved_output() and obj.resolved_output().exists():
        obj.last_status = "up_to_date"
        return False

    images = load_images(ordered_paths)
    if len(images) != len(ordered_paths):
        obj.last_status = "error"
        obj.last_error = "No se han podido cargar todas las imagenes del set."
        return False

    try:
        positions, scores = compute_positions(images)
        pano = linear_compose(images=images, positions=positions)

        if obj.align_pattern.strip():
            pattern_path = Path(obj.align_pattern.strip())
            if pattern_path.is_file():
                pattern = cv2.imread(str(pattern_path), cv2.IMREAD_COLOR)
                if pattern is None:
                    raise ValueError(f"No se pudo leer align_pattern: {pattern_path}")
                angle_deg, centers, match_scores = compute_pattern_alignment_angle(
                    image=pano,
                    pattern=pattern,
                    threshold=ALIGN_PATTERN_THRESHOLD,
                    expected_count=ALIGN_PATTERN_COUNT,
                )
                print(
                    "[INF] Alineacion por patron "
                    f"ID={obj.object_id}, angle={angle_deg:.4f}, centers={centers}, "
                    f"scores={[round(x, 3) for x in match_scores]}"
                )
                if abs(angle_deg) > 1e-12:
                    base_crop = crop_black_borders(pano)
                    cand_pos = crop_black_borders(rotate_image(pano, angle_deg))
                    cand_neg = crop_black_borders(rotate_image(pano, -angle_deg))

                    candidates = [
                        ("0.0000", base_crop),
                        (f"{angle_deg:.4f}", cand_pos),
                        (f"{-angle_deg:.4f}", cand_neg),
                    ]
                    best_angle_txt, best_img = min(candidates, key=lambda item: item[1].shape[0])
                    pano = best_img

                    print(
                        "[INF] Evaluacion candidatos de rotacion "
                        f"ID={obj.object_id}: h0={base_crop.shape[0]}, "
                        f"h+={cand_pos.shape[0]}, h-={cand_neg.shape[0]}, "
                        f"mejor_candidato={best_angle_txt}"
                    )
                    print(
                        "[INF] Rotacion aplicada por patron "
                        f"ID={obj.object_id}: angle_final={best_angle_txt} grados"
                    )
                else:
                    print(f"[INF] Rotacion por patron omitida ID={obj.object_id}: angulo ~ 0")
            else:
                print(f"[WRN] ALIGN_PATTERN no existe para ID={obj.object_id}: {pattern_path}")
        elif abs(obj.slope) > 1e-12:
            angle_deg = -np.degrees(np.arctan(obj.slope))
            print(
                "[INF] Rotacion por slope "
                f"ID={obj.object_id}: slope={obj.slope}, angle={float(angle_deg):.4f}"
            )
            pano = rotate_image(pano, float(angle_deg))

        pano = crop_black_borders(pano)

        output_path = obj.resolved_output()
        if output_path is None:
            obj.last_status = "missing_attrs"
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ok = cv2.imwrite(str(output_path), pano)
        if not ok:
            raise RuntimeError(f"No se pudo guardar salida: {output_path}")

        if obj.delete_partials:
            for p in ordered_paths:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            obj.last_status = f"generated_and_deleted:{output_path}"
            obj.last_error = ""
            obj.last_built_signature = signature
            mean_score = float(np.mean(scores)) if scores else 1.0
            print(
                f"[INF] Generada imagen compuesta ID={obj.object_id} "
                f"({len(ordered_paths)} imgs, score={mean_score:.3f}) -> {output_path}"
            )
            print(f"[INF] Objeto ID={obj.object_id} marcado para eliminar (DELETE_PARTIALS=true).")
            return True

        obj.last_status = f"generated:{output_path}"
        obj.last_error = ""
        obj.last_built_signature = signature
        obj.enabled = False
        mean_score = float(np.mean(scores)) if scores else 1.0
        print(
            f"[INF] Generada imagen compuesta ID={obj.object_id} "
            f"({len(ordered_paths)} imgs, score={mean_score:.3f}) -> {output_path}"
        )
        print(f"[INF] Objeto ID={obj.object_id} marcado como inactivo tras generar.")
        return False
    except Exception as exc:
        obj.last_status = "error"
        obj.last_error = str(exc)
        print(f"[ERR] Fallo generando ID={obj.object_id}: {exc}")
        return False


def write_runtime_state(objects: dict[str, ComposeObject], photo_sets: dict[str, dict[str, Any]]) -> None:
    RUNTIME_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    object_ids = sorted(objects.keys())
    sets_payload = []
    for set_id in sorted(photo_sets.keys()):
        set_data = photo_sets[set_id]
        cols = set_data["cols"]
        rows = set_data["rows"]
        expected = cols * rows
        found = len(set_data["coords"])
        sets_payload.append(
            {
                "id": set_id,
                "cols": cols,
                "rows": rows,
                "expected": expected,
                "found": found,
                "complete": found == expected and not set_data.get("invalid_dimensions", False),
                "invalid_dimensions": bool(set_data.get("invalid_dimensions", False)),
                "object_exists": set_id in objects,
            }
        )

    payload = {
        "object_ids": object_ids,
        "objects": [
            {
                "id": obj.object_id,
                "status": obj.last_status,
                "error": obj.last_error,
                "output": str(obj.resolved_output()) if obj.resolved_output() else "",
                "enabled": obj.enabled,
            }
            for obj in sorted(objects.values(), key=lambda x: x.object_id)
        ],
        "sets": sets_payload,
    }
    RUNTIME_STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_command_files(objects: dict[str, ComposeObject]) -> None:
    COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in COMMANDS_DIR.iterdir() if p.is_file()], key=lambda p: p.name.lower())
    for file_path in files:
        try:
            params = parse_command_file(file_path)
            print(f"[INF] Leyendo comando: {file_path.name}")
            for key, value in params.items():
                print(f"[INF]   {key}:{value}")
            object_id = params.get("ID", "").strip()
            if not object_id:
                print(f"[WRN] Comando sin ID, ignorado: {file_path.name}")
                file_path.unlink(missing_ok=True)
                continue
            if "OUTPUT_NAME" not in {k.strip().upper() for k in params.keys()}:
                print(f"[WRN] Comando sin OUTPUT_NAME, ignorado: {file_path.name}")
                file_path.unlink(missing_ok=True)
                continue

            if object_id not in objects:
                objects[object_id] = ComposeObject(object_id=object_id)
                print(f"[INF] Creado nuevo objeto ID={object_id}")

            apply_params(objects[object_id], params)
            print(f"[INF] Comando aplicado a ID={object_id} desde {file_path.name}")
            file_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[ERR] Error procesando comando {file_path.name}: {exc}")


def execution_loop(stop_event: mp.Event) -> None:
    objects: dict[str, ComposeObject] = {}

    COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    while not stop_event.is_set():
        process_command_files(objects)
        photo_sets = collect_photo_sets()
        delete_ids: list[str] = []
        for object_id, obj in list(objects.items()):
            should_delete_object = maybe_generate_composite(obj, photo_sets)
            if should_delete_object:
                delete_ids.append(object_id)
        for object_id in delete_ids:
            objects.pop(object_id, None)
            print(f"[INF] Objeto eliminado: ID={object_id}")
        write_runtime_state(objects, photo_sets)
        stop_event.wait(timeout=0.5)
