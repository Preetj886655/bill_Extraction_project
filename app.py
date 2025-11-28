# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests, io
from pdf2image import convert_from_bytes
import pytesseract
import cv2
import numpy as np
import re
from typing import List, Dict, Any
from PIL import Image
from fuzzywuzzy import fuzz

app = FastAPI(title="Bill Extraction API")

money_re = re.compile(r'[\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?]')

class Req(BaseModel):
    document: str

def download_file(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def pdf_to_images(pdf_bytes: bytes):
    try:
        return convert_from_bytes(pdf_bytes, dpi=300)
    except Exception:
        # if not a PDF or conversion fails, try to open as image
        return [Image.open(io.BytesIO(pdf_bytes)).convert("RGB")]

def preprocess_pil_image(pil_img):
    arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    # adaptive threshold - helps for OCR
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY,11,2)
    return th

def ocr_with_boxes(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    results = []
    n = len(data['text'])
    for i in range(n):
        text = str(data['text'][i]).strip()
        conf = data['conf'][i]
        try:
            conf = int(conf)
        except:
            try:
                conf = float(conf)
            except:
                conf = -1
        if text:
            results.append({
                "text": text,
                "left": int(data['left'][i]),
                "top": int(data['top'][i]),
                "width": int(data['width'][i]),
                "height": int(data['height'][i]),
                "conf": conf
            })
    return results

def group_rows(words, y_tol=12):
    words_sorted = sorted(words, key=lambda w: w['top'])
    rows = []
    for w in words_sorted:
        placed = False
        for r in rows:
            if abs(w['top'] - r['y_mean']) <= y_tol:
                r['words'].append(w)
                r['y_mean'] = int(sum([x['top'] for x in r['words']]) / len(r['words']))
                placed = True
                break
        if not placed:
            rows.append({'y_mean': w['top'], 'words': [w]})
    for r in rows:
        r['words'] = sorted(r['words'], key=lambda x: x['left'])
    return rows

def parse_row_to_item(row):
    texts = [w['text'] for w in row['words']]
    line_text = " ".join(texts)
    amount = None
    qty = None
    rate = None
    numeric_tokens = [t for t in texts if re.search(r'\d', t)]
    if numeric_tokens:
        last = numeric_tokens[-1].replace(',', '')
        try:
            amount = float(re.sub(r'[^\d.]','', last))
        except:
            amount = None
    for t in reversed(texts[:-1]):
        if re.fullmatch(r'\d+', t):
            qty = int(t); break
    first_num_idx = None
    for i,t in enumerate(texts):
        if re.search(r'\d', t):
            first_num_idx = i; break
    item_name = " ".join(texts[:first_num_idx]) if first_num_idx else line_text
    return {"item_name": item_name.strip(), "item_amount": amount, "item_rate": rate, "item_quantity": qty, "raw": line_text}

def dedupe_items(items):
    merged = []
    for it in items:
        found = False
        for m in merged:
            if it['item_amount'] and m['item_amount'] and abs(it['item_amount'] - m['item_amount']) < 0.01:
                score = fuzz.token_set_ratio(it['item_name'], m['item_name'])
                if score > 85:
                    if len(it['item_name']) > len(m['item_name']):
                        m['item_name'] = it['item_name']
                    if not m.get('item_quantity') and it.get('item_quantity'):
                        m['item_quantity'] = it['item_quantity']
                    found = True
                    break
        if not found:
            merged.append(it)
    return merged

@app.post("/extract-bill-data")
def extract(req: Req):
    try:
        b = download_file(req.document)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download error: {e}")
    # Determine image pages
    try:
        imgs = pdf_to_images(b)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image conversion error: {e}")

    pagewise = []
    all_items = []

    for i, pil in enumerate(imgs):
        proc = preprocess_pil_image(pil)
        words = ocr_with_boxes(proc)
        rows = group_rows(words, y_tol=12)
        page_items = []
        for r in rows:
            parsed = parse_row_to_item(r)
            # filter non-items
            name_lower = parsed['item_name'].lower()
            if parsed['item_amount'] is not None and parsed['item_name'] and not any(k in name_lower for k in ['total', 'subtotal', 'tax', 'gst', 'discount']):
                page_items.append(parsed)
                all_items.append(parsed)
        pagewise.append({"page_no": str(i+1), "bill_items": page_items})

    uniq_items = dedupe_items(all_items)
    reconciled_amount = sum([it['item_amount'] for it in uniq_items if it['item_amount']])
    return {"is_success": True, "data": {"pagewise_line_items": pagewise, "total_item_count": len(uniq_items), "reconciled_amount": round(reconciled_amount, 2)}}
