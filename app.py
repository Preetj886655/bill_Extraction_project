# file: app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests, io, os, tempfile
from pdf2image import convert_from_bytes
import pytesseract
import cv2
import numpy as np
import re
from typing import List, Dict, Any
from fuzzywuzzy import fuzz   # pip install fuzzywuzzy[speedup]

app = FastAPI()

money_re = re.compile(r'[\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?]+')  # simplified

class Req(BaseModel):
    document: str

def download_file(url: str) -> bytes:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.content

def pdf_to_images(pdf_bytes: bytes):
    return convert_from_bytes(pdf_bytes, dpi=300)

def preprocess_pil_image(pil_img):
    arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    # adaptive threshold
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY,11,2)
    return th

def ocr_with_boxes(image):
    # returns list of dicts: {text, left, top, width, height, conf}
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    results = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        conf = int(data['conf'][i]) if data['conf'][i].isdigit() else -1
        if text:
            results.append({
                "text": text,
                "left": data['left'][i],
                "top": data['top'][i],
                "width": data['width'][i],
                "height": data['height'][i],
                "conf": conf
            })
    return results

def group_rows(words, y_tol=10):
    # group words into rows by y-coordinate
    words_sorted = sorted(words, key=lambda w: w['top'])
    rows = []
    for w in words_sorted:
        placed = False
        for r in rows:
            if abs(w['top'] - r['y_mean']) <= y_tol:
                r['words'].append(w);
                # update mean y
                r['y_mean'] = int(sum([x['top'] for x in r['words']]) / len(r['words']))
                placed = True; break
        if not placed:
            rows.append({'y_mean': w['top'], 'words': [w]})
    # sort words in every row by left
    for r in rows:
        r['words'] = sorted(r['words'], key=lambda x: x['left'])
    return rows

def parse_row_to_item(row):
    texts = [w['text'] for w in row['words']]
    line_text = " ".join(texts)
    # find money tokens - rightmost money likely amount
    money_tokens = [t for t in texts if re.search(r'\d', t)]
    # naive: last numeric token as amount
    amount = None
    qty = None
    rate = None
    # find tokens that resemble amounts by regex for floats
    numeric_tokens = [t for t in texts if re.search(r'\d', t)]
    if numeric_tokens:
        # last numeric token -> amount
        last = numeric_tokens[-1].replace(',', '')
        try:
            amount = float(re.sub(r'[^\d.]','', last))
        except:
            amount = None
    # attempt to find qty (an integer)
    for t in reversed(texts[:-1]):
        if re.fullmatch(r'\d+', t):
            qty = int(t); break
    # item name everything to left of first numeric-looking token
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
            # compare amounts
            if it['item_amount'] and m['item_amount'] and abs(it['item_amount'] - m['item_amount']) < 0.01:
                score = fuzz.token_set_ratio(it['item_name'], m['item_name'])
                if score > 85:
                    # merge — keep longer name and non-null qty
                    if len(it['item_name']) > len(m['item_name']): m['item_name'] = it['item_name']
                    if not m.get('item_quantity'): m['item_quantity'] = it.get('item_quantity')
                    found = True; break
        if not found:
            merged.append(it)
    return merged

@app.post("/extract-bill-data")
def extract(req: Req):
    try:
        b = download_file(req.document)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    imgs = pdf_to_images(b) if req.document.lower().endswith(".pdf") or b[:4]==b'%PDF' else [Image.open(io.BytesIO(b))]
    pagewise = []
    total_items = 0
    all_items = []
    import PIL.Image as Image
    for i, pil in enumerate(imgs):
        proc = preprocess_pil_image(pil)
        words = ocr_with_boxes(proc)
        rows = group_rows(words, y_tol=12)
        page_items = []
        for r in rows:
            parsed = parse_row_to_item(r)
            # filter obviously non-line rows (like headers)
            if parsed['item_amount'] is not None and parsed['item_name'] and parsed['item_name'].lower() not in ['total','subtotal','tax']:
                page_items.append(parsed)
                all_items.append(parsed)
        pagewise.append({"page_no": str(i+1), "bill_items": page_items})
        total_items += len(page_items)
    # dedupe across pages
    uniq_items = dedupe_items(all_items)
    reconciled_amount = sum([it['item_amount'] for it in uniq_items if it['item_amount']])
    return {"is_success": True, "data": {"pagewise_line_items": pagewise, "total_item_count": len(uniq_items), "reconciled_amount": round(reconciled_amount, 2)}}
