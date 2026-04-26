from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PdfToImage import convert_pdf_to_images
from PictureToData import convert_picture_to_data


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = BASE_DIR / "flyer.pdf"
DEFAULT_IMAGE_DIR = BASE_DIR / "flyerImages"
DEFAULT_OUTPUT_PATH = BASE_DIR / "items.json"


def convert_flyer_pdf_to_items(
    pdf_path: str | Path = DEFAULT_PDF_PATH,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, Any]]:
    generated_images = convert_pdf_to_images(pdf_path, image_dir)
    all_items: list[dict[str, Any]] = []

    for page_number, image_path in enumerate(generated_images, start=1):
        page_items = convert_picture_to_data(image_path)
        if not page_items:
            continue

        for item in page_items:
            item["page"] = page_number
            item["image"] = str(image_path)

        all_items.extend(page_items)
    print(json.dumps(all_items, indent=2))
    if all_items:
        Path(output_path).write_text(json.dumps(all_items, indent=2), encoding="utf-8")

    return all_items


def main() -> None:
    items = convert_flyer_pdf_to_items()
    if not items:
        print("No grocery items found.")
        return

    print(json.dumps(items, indent=2))


if __name__ == "__main__":
    main()
