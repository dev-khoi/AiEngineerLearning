from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_PATH = "zai-org/GLM-OCR"
PROMPT = (
    "Extract grocery flyer items from this image. "
    "Return plain text only. One item per line. "
    "Use this exact format: name=<name> ; price=<price> ; bbx=<x1,y1,x2,y2>. "
    "If bbx is unknown, leave it empty as bbx=. "
    "If a price appears like 599, normalize it to 5.99. "
    "Ignore headers, footers, legal text, and non-product content. "
    "If there are no grocery items, return an empty response."
)

_processor = None
_model = None


def load_ocr_model(model_path: str = MODEL_PATH):
    global _processor, _model
    if _processor is None or _model is None:
        _processor = AutoProcessor.from_pretrained(model_path)
        _model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype="auto",
            device_map="auto",
        )
    return _processor, _model


def _parse_line_format(raw_text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in raw_text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue

        if not cleaned.startswith("name="):
            continue

        parts = [part.strip() for part in cleaned.split(";")]
        data: dict[str, Any] = {}
        for part in parts:
            if not part:
                continue
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            data[key.strip()] = value.strip()

        if data:
            items.append(data)

    return items


def _normalize_price_value(price: Any) -> str:
    if price is None:
        return ""

    value = str(price).strip()
    if not value:
        return ""

    digits = re.sub(r"\D", "", value)

    if digits and len(digits) >= 3 and "/" in value:
        normalized_number = f"{int(digits[:-2])}.{digits[-2:]}"
        suffix = value[value.find("/") :].strip()
        return f"{normalized_number} {suffix}".strip()

    if re.fullmatch(r"\$?\d{3,4}", value):
        normalized = f"{int(digits[:-2])}.{digits[-2:]}"
        return normalized

    if re.fullmatch(r"\d+", value) and len(value) >= 3:
        return f"{int(value[:-2])}.{value[-2:]}"

    return value.replace("$", "")


def _normalize_bbx_value(bbx: Any) -> list[float]:
    if isinstance(bbx, str):
        return []

    if not isinstance(bbx, list) or len(bbx) != 4:
        return []

    normalized: list[float] = []
    for value in bbx:
        if isinstance(value, (int, float)):
            normalized.append(float(value))
        else:
            try:
                normalized.append(float(str(value).strip()))
            except ValueError:
                return []
    return normalized


def _flatten_item_candidates(items: list[Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []

    for item in items:
        if isinstance(item, dict):
            flattened.append(item)
        elif isinstance(item, list):
            flattened.extend(_flatten_item_candidates(item))

    return flattened


def normalize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_items: list[dict[str, Any]] = []
    for item in _flatten_item_candidates(items):
        name = str(item.get("name", "")).strip()
        price = _normalize_price_value(item.get("price", ""))
        bbx = _normalize_bbx_value(item.get("bbx", []))

        if not name or not price:
            continue

        normalized_items.append({"name": name, "price": price, "bbx": bbx})

    return normalized_items


def convert_picture_to_data(
    image_path: str | Path,
    model_path: str = MODEL_PATH,
) -> list[dict[str, Any]]:
    processor, model = load_ocr_model(model_path)
    image_file = Path(image_path).resolve()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_file)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    extracted_items = _parse_line_format(output_text)
    return normalize_items(extracted_items)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    image_path = base_dir / "flyerImages" / "flyer_1.png"
    result = json.dumps(convert_picture_to_data(image_path), indent=2)
