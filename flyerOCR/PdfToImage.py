from pathlib import Path

import pypdfium2 as pdfium


def convert_pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 300,
) -> list[Path]:
    pdf_file = Path(pdf_path)
    image_dir = Path(output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(str(pdf_file))
    generated_images: list[Path] = []

    try:
        for index, page in enumerate(pdf, start=1):
            bitmap = page.render(scale=dpi / 72)
            image = bitmap.to_pil()
            image_path = image_dir / f"{pdf_file.stem}_{index}.png"
            image.save(image_path, "PNG")
            generated_images.append(image_path)
    finally:
        pdf.close()

    return generated_images


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    pages = convert_pdf_to_images(base_dir / "flyer.pdf", base_dir / "flyerImages")
    for page in pages:
        print(f"Saved {page}")
