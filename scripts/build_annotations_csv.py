from pathlib import Path
import csv
import re
import argparse

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

PART_PATTERNS = [
    (r"(?:^|[_\-])lateral[_\-]?body(?:$|[_\-])", "whole_body", "lateral"),
    (r"(?:^|[_\-])dorsal[_\-]?body(?:$|[_\-])", "whole_body", "dorsal"),
    (r"(?:^|[_\-])whole[_\-]?body(?:$|[_\-])", "whole_body", "whole"),
    (r"(?:^|[_\-])body(?:$|[_\-])", "whole_body", "whole"),

    (r"(?:^|[_\-])carapace(?:$|[_\-])", "carapace", "isolated"),
    (r"(?:^|[_\-])pseudorostrum(?:$|[_\-])", "pseudorostrum", "isolated"),
    (r"(?:^|[_\-])eyelobe(?:$|[_\-])", "eyelobe", "isolated"),
    (r"(?:^|[_\-])abdomen(?:$|[_\-])", "abdomen", "isolated"),
    (r"(?:^|[_\-])thorax(?:$|[_\-])", "thorax", "isolated"),
    (r"(?:^|[_\-])pereon(?:$|[_\-])", "pereon", "isolated"),
    (r"(?:^|[_\-])pleon(?:$|[_\-])", "pleon", "isolated"),
    (r"(?:^|[_\-])uropod(?:$|[_\-])", "uropod", "isolated"),
    (r"(?:^|[_\-])endopod(?:$|[_\-])", "endopod", "isolated"),
    (r"(?:^|[_\-])exopod(?:$|[_\-])", "exopod", "isolated"),
    (r"(?:^|[_\-])maxilliped[_\-]?3(?:$|[_\-])", "maxilliped_3", "isolated"),
    (r"(?:^|[_\-])maxilipede[_\-]?3(?:$|[_\-])", "maxilliped_3", "isolated"),
    (r"(?:^|[_\-])antennule(?:$|[_\-])", "antennule", "isolated"),
    (r"(?:^|[_\-])antenna(?:$|[_\-])", "antenna", "isolated"),

    (r"(?:^|[_\-])pereopod[_\-]?1(?:$|[_\-])", "pereopod_1", "isolated"),
    (r"(?:^|[_\-])pereopod[_\-]?2(?:$|[_\-])", "pereopod_2", "isolated"),
    (r"(?:^|[_\-])pereopod[_\-]?3(?:$|[_\-])", "pereopod_3", "isolated"),
    (r"(?:^|[_\-])pereopod[_\-]?4(?:$|[_\-])", "pereopod_4", "isolated"),
    (r"(?:^|[_\-])pereopod[_\-]?5(?:$|[_\-])", "pereopod_5", "isolated"),

    (r"(?:^|[_\-])p1(?:$|[_\-])", "pereopod_1", "isolated"),
    (r"(?:^|[_\-])p2(?:$|[_\-])", "pereopod_2", "isolated"),
    (r"(?:^|[_\-])p3(?:$|[_\-])", "pereopod_3", "isolated"),
    (r"(?:^|[_\-])p4(?:$|[_\-])", "pereopod_4", "isolated"),
    (r"(?:^|[_\-])p5(?:$|[_\-])", "pereopod_5", "isolated"),
]

FIELDNAMES = [
    "image_path",
    "image_name",
    "species",
    "body_part",
    "view",
    "source",
]


def detect_part_and_view(filename: str) -> tuple[str, str]:
    name = filename.lower()
    for pattern, body_part, view in PART_PATTERNS:
        if re.search(pattern, name):
            return body_part, view
    return "unknown", "unknown"


def collect_rows(processed_root: Path) -> list[dict]:
    rows = []
    species_dirs = [p for p in processed_root.iterdir() if p.is_dir()]

    for species_dir in sorted(species_dirs, key=lambda p: p.name.lower()):
        species = species_dir.name
        images_dir = species_dir / "images"
        if not images_dir.exists():
            continue

        for img_path in sorted(images_dir.iterdir(), key=lambda p: p.name.lower()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in VALID_EXTS:
                continue

            body_part, view = detect_part_and_view(img_path.stem)
            rows.append({
                "image_path": str(img_path.resolve()),
                "image_name": img_path.name,
                "species": species,
                "body_part": body_part,
                "view": view,
                "source": "illustration",
            })
    return rows


def read_existing(csv_path: Path) -> dict[str, dict]:
    existing = {}
    if not csv_path.exists():
        return existing

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing[row["image_path"]] = row
    return existing


def write_csv(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build or update annotations.csv from processed Campylaspis dataset.")
    parser.add_argument(
        "--processed_root",
        required=True,
        help="Path to processed dataset root, e.g. /home/.../datasets/campylaspis/processed",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to output CSV, e.g. /home/.../datasets/campylaspis_parts/annotations.csv",
    )
    parser.add_argument(
        "--mode",
        choices=["rewrite", "update"],
        default="update",
        help="rewrite = regenerate all rows; update = preserve existing rows and append only new images",
    )
    args = parser.parse_args()

    processed_root = Path(args.processed_root).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not processed_root.exists():
        raise FileNotFoundError(f"processed_root not found: {processed_root}")

    new_rows = collect_rows(processed_root)

    if args.mode == "rewrite":
        final_rows = new_rows
    else:
        existing = read_existing(output_csv)
        merged = dict(existing)
        for row in new_rows:
            merged[row["image_path"]] = row
        final_rows = sorted(merged.values(), key=lambda r: (r["species"].lower(), r["image_name"].lower()))

    write_csv(output_csv, final_rows)

    total = len(final_rows)
    unknown = sum(1 for r in final_rows if r["body_part"] == "unknown")
    print(f"CSV saved to: {output_csv}")
    print(f"Total rows: {total}")
    print(f"Unknown body_part rows: {unknown}")

    if unknown > 0:
        print("Examples of unknown rows:")
        shown = 0
        for row in final_rows:
            if row["body_part"] == "unknown":
                print(f"  - {row['image_name']}")
                shown += 1
                if shown >= 10:
                    break


if __name__ == "__main__":
    main()
