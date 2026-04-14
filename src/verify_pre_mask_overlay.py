import argparse
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


PRE_SUFFIX = "_pre_disaster"
POST_SUFFIX = "_post_disaster"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a binary building mask from the pre-disaster label and overlay "
            "that same mask on both the pre- and post-disaster images."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("3/train/train"),
        help=(
            "Dataset root containing images/ and labels/ folders. "
            "Default: 3/train/train"
        ),
    )
    parser.add_argument(
        "--sample-id",
        help=(
            "Sample stem without the disaster suffix, e.g. "
            "'guatemala-volcano_00000000' or "
            "'guatemala-volcano_00000000_q0'."
        ),
    )
    parser.add_argument(
        "--pre-image",
        type=Path,
        help="Optional path to the pre-disaster image. Used instead of --sample-id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("verification_outputs"),
        help="Directory where the mask and overlay images will be written.",
    )
    parser.add_argument(
        "--color",
        default="255,0,0",
        help="Overlay RGB color as 'R,G,B'. Default: 255,0,0",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=110,
        help="Overlay alpha in [0, 255]. Default: 110",
    )
    parser.add_argument(
        "--outline-width",
        type=int,
        default=2,
        help="Outline width in pixels for building borders. Default: 2",
    )
    return parser.parse_args()


def infer_sample_id(pre_image: Path) -> str:
    name = pre_image.stem
    if PRE_SUFFIX not in name:
        raise ValueError(
            f"Expected a pre-disaster image name containing '{PRE_SUFFIX}', got '{pre_image.name}'."
        )
    return name.replace(PRE_SUFFIX, "")


def resolve_paths(root: Path, sample_id: str) -> tuple[Path, Path, Path]:
    images_dir = root / "images"
    labels_dir = root / "labels"
    pre_image = images_dir / f"{sample_id}{PRE_SUFFIX}.png"
    post_image = images_dir / f"{sample_id}{POST_SUFFIX}.png"
    pre_label = labels_dir / f"{sample_id}{PRE_SUFFIX}.json"

    missing = [path for path in (pre_image, post_image, pre_label) if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required file(s):\n{missing_text}")

    return pre_image, post_image, pre_label


def strip_outer_parentheses(text: str) -> str:
    text = text.strip()
    if len(text) < 2 or text[0] != "(" or text[-1] != ")":
        raise ValueError(f"Expected parenthesized text, got: {text[:50]}")
    return text[1:-1].strip()


def split_top_level_groups(text: str) -> list[str]:
    groups: list[str] = []
    depth = 0
    start = None
    for index, char in enumerate(text):
        if char == "(":
            if depth == 0:
                start = index + 1
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and start is not None:
                groups.append(text[start:index].strip())
                start = None
    return groups


def parse_ring(coord_text: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for pair in coord_text.split(","):
        x_str, y_str = pair.strip().split()
        points.append((float(x_str), float(y_str)))
    return points


def parse_wkt_polygons(wkt_text: str) -> list[list[tuple[float, float]]]:
    wkt_text = wkt_text.strip()
    geometry_type = wkt_text.split("(", 1)[0].strip().upper()
    body = wkt_text[wkt_text.index("(") :]

    if geometry_type == "POLYGON":
        polygon_body = strip_outer_parentheses(body)
        rings = split_top_level_groups(polygon_body)
        return [parse_ring(rings[0])] if rings else []

    if geometry_type == "MULTIPOLYGON":
        polygons: list[list[tuple[float, float]]] = []
        multi_body = strip_outer_parentheses(body)
        for polygon_body in split_top_level_groups(multi_body):
            rings = split_top_level_groups(polygon_body)
            if rings:
                polygons.append(parse_ring(rings[0]))
        return polygons

    raise ValueError(f"Unsupported geometry type: {geometry_type}")


def load_building_polygons(label_path: Path) -> list[list[tuple[float, float]]]:
    with label_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    polygons: list[list[tuple[float, float]]] = []
    for feature in data.get("features", {}).get("xy", []):
        if feature.get("properties", {}).get("feature_type") != "building":
            continue
        polygons.extend(parse_wkt_polygons(feature["wkt"]))
    return polygons


def build_mask(size: tuple[int, int], polygons: Iterable[list[tuple[float, float]]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=255)
    return mask


def draw_outlines(size: tuple[int, int], polygons: Iterable[list[tuple[float, float]]], color: tuple[int, int, int], width: int) -> Image.Image:
    outline = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(outline)
    for polygon in polygons:
        if len(polygon) >= 2:
            closed = polygon + [polygon[0]]
            draw.line(closed, fill=(*color, 255), width=width)
    return outline


def apply_overlay(
    base_image: Image.Image,
    mask: Image.Image,
    polygons: Iterable[list[tuple[float, float]]],
    color: tuple[int, int, int],
    alpha: int,
    outline_width: int,
) -> Image.Image:
    base_rgba = base_image.convert("RGBA")
    overlay = Image.new("RGBA", base_rgba.size, (*color, 0))
    overlay.putalpha(mask.point(lambda value: alpha if value else 0))
    blended = Image.alpha_composite(base_rgba, overlay)
    outline = draw_outlines(base_rgba.size, polygons, color, outline_width)
    return Image.alpha_composite(blended, outline)


def parse_color(color_text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in color_text.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise ValueError("Color must be three comma-separated integers in [0, 255].")
    return tuple(parts)  # type: ignore[return-value]


def build_contact_sheet(images: list[tuple[str, Image.Image]]) -> Image.Image:
    font = ImageFont.load_default()
    label_height = 24
    width, height = images[0][1].size
    sheet = Image.new("RGB", (width * len(images), height + label_height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for index, (label, image) in enumerate(images):
        x_offset = index * width
        draw.text((x_offset + 8, 5), label, fill=(0, 0, 0), font=font)
        sheet.paste(image.convert("RGB"), (x_offset, label_height))

    return sheet


def main() -> None:
    args = parse_args()
    color = parse_color(args.color)
    alpha = max(0, min(args.alpha, 255))

    if args.pre_image:
        sample_id = infer_sample_id(args.pre_image)
        root = args.pre_image.parent.parent
    elif args.sample_id:
        sample_id = args.sample_id
        root = args.root
    else:
        raise ValueError("Provide either --sample-id or --pre-image.")

    pre_image_path, post_image_path, pre_label_path = resolve_paths(root, sample_id)
    output_dir = args.output_dir / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    polygons = load_building_polygons(pre_label_path)
    pre_image = Image.open(pre_image_path).convert("RGB")
    post_image = Image.open(post_image_path).convert("RGB")

    if pre_image.size != post_image.size:
        raise ValueError(
            f"Image size mismatch: pre={pre_image.size}, post={post_image.size}"
        )

    mask = build_mask(pre_image.size, polygons)
    pre_overlay = apply_overlay(pre_image, mask, polygons, color, alpha, args.outline_width)
    post_overlay = apply_overlay(post_image, mask, polygons, color, alpha, args.outline_width)

    mask_path = output_dir / f"{sample_id}_pre_building_mask.png"
    pre_overlay_path = output_dir / f"{sample_id}_pre_overlay.png"
    post_overlay_path = output_dir / f"{sample_id}_post_overlay_using_pre_mask.png"
    contact_sheet_path = output_dir / f"{sample_id}_verification_sheet.png"

    mask.save(mask_path)
    pre_overlay.save(pre_overlay_path)
    post_overlay.save(post_overlay_path)
    build_contact_sheet(
        [
            ("Pre Image", pre_image),
            ("Pre + Pre Mask", pre_overlay),
            ("Post + Pre Mask", post_overlay),
        ]
    ).save(contact_sheet_path)

    print(f"Sample: {sample_id}")
    print(f"Pre image: {pre_image_path}")
    print(f"Post image: {post_image_path}")
    print(f"Pre label: {pre_label_path}")
    print(f"Building polygons: {len(polygons)}")
    print(f"Mask: {mask_path}")
    print(f"Pre overlay: {pre_overlay_path}")
    print(f"Post overlay: {post_overlay_path}")
    print(f"Verification sheet: {contact_sheet_path}")


if __name__ == "__main__":
    main()
