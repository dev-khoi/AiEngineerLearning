import pypdfium2 as pdfium
import os

os.makedirs("flyerImages", exist_ok=True)

base_dir = os.path.dirname(__file__)
pdf_path = os.path.join(base_dir, "flyer.pdf")

pdf = pdfium.PdfDocument(pdf_path)

for i, page in enumerate(pdf):
    bitmap = page.render(scale=300 / 72)  # 300 DPI equivalent
    image = bitmap.to_pil()
    image.save(os.path.join(base_dir, "flyerImages", f"flyer_{i+1}.png"), "PNG")
    print("Saved flyer page", i + 1)

pdf.close()
