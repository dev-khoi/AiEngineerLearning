from pathlib import Path
import json
import platform

import torch


BASE_DIR = Path(__file__).resolve().parent
IMAGE_FILE = BASE_DIR / "flyerImages" / "flyer_1.png"
OUTPUT_DIR = BASE_DIR / "flyerOutput"
RESULTS_FILE = OUTPUT_DIR / "nemotron_ocr_results.json"
TEXT_FILE = OUTPUT_DIR / "nemotron_ocr_text.txt"


def ensure_supported_environment() -> None:
    if platform.system() != "Linux":
        raise RuntimeError(
            "Nemotron OCR v2 officially supports Linux with an NVIDIA GPU. "
            f"Current platform: {platform.system()}."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Nemotron OCR v2 needs a CUDA-enabled PyTorch install. "
            "Your environment does not have CUDA available."
        )


def load_pipeline():
    try:
        from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
    except ImportError as exc:
        raise RuntimeError(
            "Nemotron OCR v2 is not installed. Clone `nvidia/nemotron-ocr-v2` and install "
            "it with `pip install --no-build-isolation -v .` inside a Linux Python 3.12 + CUDA environment."
        ) from exc

    return NemotronOCRV2(lang="multi")


def save_outputs(predictions: list[dict]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    extracted_text = "\n".join(pred.get("text", "") for pred in predictions if pred.get("text"))
    TEXT_FILE.write_text(extracted_text, encoding="utf-8")


def main() -> None:
    ensure_supported_environment()

    if not IMAGE_FILE.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_FILE}")

    ocr = load_pipeline()
    predictions = ocr(str(IMAGE_FILE), merge_level="paragraph")
    save_outputs(predictions)

    print(f"Saved structured OCR results to: {RESULTS_FILE}")
    print(f"Saved extracted text to: {TEXT_FILE}")
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
