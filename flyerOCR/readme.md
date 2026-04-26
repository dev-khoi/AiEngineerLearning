# important (errorReport), state of the project:
output: 

Loading weights: 100%|████████████| 510/510 [00:02<00:00, 183.96it/s]
[]
No grocery items found.



reason: the condition for normalizing the data is too strict, 

way to solve: test ocr output -> test on small amount of picture first 



# Flyer OCR Pipeline

This folder now has an end-to-end pipeline for grocery flyer extraction:

1. Read `flyer.pdf`
2. Convert each PDF page into a PNG image
3. Run each page image through the OCR model
4. Normalize the extracted items into this shape:

```json
[
  {
    "name": "Strawberries 907 g",
    "price": "5.99",
    "bbx": [120.0, 340.0, 480.0, 510.0],
    "page": 1,
    "image": ".../flyerImages/flyer_1.png"
  }
]
```

## Files

- `convertPdfToImage.py`: converts a PDF into page images
- `convertPictureToData.py`: loads the OCR model once, sends each image to the model, extracts JSON, and normalizes the item list
- `main.py`: orchestrates the full PDF -> images -> item extraction flow

## Output rules

- Each item must contain `name`, `price`, and `bbx`
- `bbx` is normalized to `[x1, y1, x2, y2]`
- If a price looks like `599`, it is normalized to `5.99`
- Invalid items are dropped when `name`, `price`, or `bbx` is missing
- If a page has no grocery items, that page is skipped
- If the full PDF has no items, no `items.json` file is written

## How it runs

Run from the repo root:

```bash
.venv\Scripts\python.exe flyerOCR\main.py
```

If items are found, the pipeline writes:

- `flyerOCR/items.json`

It also prints the extracted items to the console.

## Main function behavior

`convert_flyer_pdf_to_items()` in `main.py`:

- converts `flyer.pdf` into images in `flyerImages/`
- runs OCR on every generated page image
- skips pages that return an empty item list
- adds `page` and `image` metadata to each item
- writes all items into `items.json`

## OCR prompt behavior

The OCR prompt asks the model to:

- return JSON only
- return an array of flyer items
- use exactly `name`, `price`, and `bbx`
- ignore headers, legal text, and non-product content
- return `[]` when no grocery items are present

## Notes

- The current OCR model is `zai-org/GLM-OCR`
- The code expects the model to return a JSON array in its response
- Grocery flyers can still produce imperfect OCR, so the normalization step cleans the common `599` -> `5.99` style error automatically
