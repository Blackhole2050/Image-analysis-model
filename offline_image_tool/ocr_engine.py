"""Low-memory OCR and geometry extraction backend.

Uses RapidOCR's small ONNX models and explicit geometric heuristics. It does
not infer hidden table values or fabricate graph samples. Ambiguous structure
is reported for human review.
"""
from __future__ import annotations

import re
from pathlib import Path


def initialize():
    import rapidocr
    from rapidocr import EngineType, LangCls, LangDet, LangRec, ModelType, RapidOCR

    # RapidOCR initializes ONNX Runtime sessions lazily. Keep one instance for
    # repeated MCP requests so model weights are not repeatedly loaded.
    global _ENGINE
    if _ENGINE is None:
        model_root = str(Path(rapidocr.__file__).resolve().parent / "models")
        _ENGINE = RapidOCR(params={
            "Global.model_root_dir": model_root,
            "Det.engine_type": EngineType.ONNXRUNTIME, "Det.model_type": ModelType.SMALL,
            "Det.lang_type": LangDet.EN,
            "Cls.engine_type": EngineType.ONNXRUNTIME, "Cls.model_type": ModelType.MOBILE,
            "Cls.lang_type": LangCls.CH,
            "Rec.engine_type": EngineType.ONNXRUNTIME, "Rec.model_type": ModelType.SMALL,
            "Rec.lang_type": LangRec.EN,
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.enable_cpu_mem_arena": False,
        })
    return _ENGINE


def _ocr(image):
    import numpy as np
    engine = initialize()
    result = engine(np.asarray(image.convert("RGB")))
    items = []
    if result is None:
        return []
    for box, text, score in zip(result.boxes, result.txts, result.scores):
        points = [(float(p[0]), float(p[1])) for p in box]
        xs, ys = zip(*points)
        items.append({"text": str(text).strip(), "score": float(score),
                      "left": min(xs), "top": min(ys), "right": max(xs), "bottom": max(ys),
                      "cx": (min(xs) + max(xs)) / 2, "cy": (min(ys) + max(ys)) / 2})
    return sorted(items, key=lambda x: (x["cy"], x["left"]))


_ENGINE = None


def _clusters(values, tolerance):
    groups = []
    for value in sorted(values):
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [sum(group) / len(group) for group in groups]


def _grid(image):
    """Return strong ruling lines, combining luminance and individual color channels."""
    import cv2
    import numpy as np
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 31, -2)
    # A colored border can have weak grayscale contrast. A border that stands
    # out in any RGB channel still contributes to the line mask. Morphological
    # opening below removes most text-sized fragments before projection.
    sample = rgb[::max(1, rgb.shape[0] // 256), ::max(1, rgb.shape[1] // 256)]
    if np.any(np.ptp(sample, axis=2) > 12):
        for channel in cv2.split(rgb):
            channel_mask = cv2.adaptiveThreshold(255 - channel, 255,
                                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                                 cv2.THRESH_BINARY, 31, -2)
            binary = cv2.bitwise_or(binary, channel_mask)
    h, w = gray.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 35), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 35)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)

    def coordinates(mask, axis, minimum):
        projection = (mask > 0).sum(axis=axis)
        indices = [i for i, amount in enumerate(projection) if amount >= minimum]
        return _clusters(indices, 3.0)

    rows = coordinates(horizontal, 1, max(8, int(w * 0.12)))
    cols = coordinates(vertical, 0, max(8, int(h * 0.12)))
    return rows, cols


def _cell_fill(image, left, top, right, bottom, rgb_pixels=None):
    """Estimate a cell's dominant fill color; return None for white/unfilled cells."""
    import numpy as np
    x0, x1 = max(0, int(left) + 2), min(image.width, int(right) - 2)
    y0, y1 = max(0, int(top) + 2), min(image.height, int(bottom) - 2)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    if rgb_pixels is None:
        rgb_pixels = np.asarray(image.convert("RGB"))
    pixels = rgb_pixels[y0:y1, x0:x1]
    # Use a center crop so borders and most text do not dominate the estimate.
    h, w = pixels.shape[:2]
    pixels = pixels[h // 5:max(h // 5 + 1, h - h // 5),
                    w // 5:max(w // 5 + 1, w - w // 5)].reshape(-1, 3)
    quantized = (pixels.astype(np.uint16) // 16).astype(np.uint8)
    bins, inverse, counts = np.unique(quantized, axis=0, return_inverse=True,
                                      return_counts=True)
    dominant_index = int(counts.argmax())
    color_pixels = pixels[inverse == dominant_index]
    rgb = np.median(color_pixels, axis=0).astype(int)
    # Ignore white or near-white paper so ordinary cells retain Excel defaults.
    if int(rgb.min()) >= 238 and int(rgb.max()) - int(rgb.min()) <= 18:
        return None
    return "#" + "".join(f"{value:02X}" for value in rgb)


def _grid_table(items, image):
    import numpy as np
    horizontal, vertical = _grid(image)
    if len(horizontal) < 3 or len(vertical) < 3:
        return None
    if len(horizontal) > 300 or len(vertical) > 100:
        return None
    row_edges = [0.0] + horizontal + [float(image.height)]
    col_edges = [0.0] + vertical + [float(image.width)]
    matrix = [[[] for _ in range(len(col_edges) - 1)] for _ in range(len(row_edges) - 1)]
    for item in items:
        ri = next((i for i in range(len(row_edges) - 1) if row_edges[i] <= item["cy"] < row_edges[i + 1]), None)
        ci = next((i for i in range(len(col_edges) - 1) if col_edges[i] <= item["cx"] < col_edges[i + 1]), None)
        if ri is not None and ci is not None:
            matrix[ri][ci].append(item)
    rows, colors = [], []
    rgb_pixels = np.asarray(image.convert("RGB"))
    sample = rgb_pixels[::max(1, rgb_pixels.shape[0] // 256),
                        ::max(1, rgb_pixels.shape[1] // 256)]
    has_color = bool(np.any(np.ptp(sample, axis=2) > 12))
    for ri, cells in enumerate(matrix):
        values = []
        row_colors = []
        for ci, cell in enumerate(cells):
            cell.sort(key=lambda x: (x["cy"], x["left"]))
            values.append(" ".join(x["text"] for x in cell).strip() or None)
            row_colors.append(_cell_fill(image, col_edges[ci], row_edges[ri],
                                         col_edges[ci + 1], row_edges[ri + 1], rgb_pixels)
                              if has_color else None)
        if any(values):
            rows.append(values)
            colors.append(row_colors)
    if len(rows) < 2:
        return None
    # Discard only wholly empty outside columns; preserve internal blank cells.
    keep = [i for i in range(len(rows[0])) if any(row[i] for row in rows)]
    rows = [[row[i] for i in keep] for row in rows]
    colors = [[row[i] for i in keep] for row in colors]
    return {"title": "", "header_rows": [rows[0]], "rows": rows[1:],
            "cell_colors": colors}


def _line_table(items, width, height):
    if len(items) < 4:
        return None
    heights = [max(1, i["bottom"] - i["top"]) for i in items]
    tol = max(3.0, sorted(heights)[len(heights) // 2] * 0.65)
    ys = _clusters([i["cy"] for i in items], tol)
    if len(ys) < 2:
        return None
    rows = []
    for y in ys:
        words = [i for i in items if abs(i["cy"] - y) <= tol]
        words.sort(key=lambda i: i["left"])
        rows.append(words)
    # Position alignment is a useful table cue only when several lines share
    # repeated column starts. Do not declare ordinary prose a table.
    starts = [i["left"] for row in rows for i in row]
    x_tol = max(5.0, sorted(heights)[len(heights) // 2] * 0.75)
    columns = _clusters(starts, x_tol)
    if len(columns) < 2 or sum(len(r) >= 2 for r in rows) < max(2, len(rows) // 2):
        return None
    matrix = [[""] * len(columns) for _ in rows]
    for ri, row in enumerate(rows):
        for item in row:
            ci = min(range(len(columns)), key=lambda j: abs(columns[j] - item["left"]))
            matrix[ri][ci] = (matrix[ri][ci] + " " + item["text"]).strip()
    # Drop columns with no observations; keep blank cells in retained columns.
    keep = [j for j in range(len(columns)) if any(row[j] for row in matrix)]
    matrix = [[row[j] or None for j in keep] for row in matrix]
    return {"title": "", "header_rows": [matrix[0]], "rows": matrix[1:]}


def _flow_nodes(items, image):
    import cv2
    import numpy as np
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Saturated strokes can disappear in grayscale; include channel edges when
    # looking for rectangular flowchart nodes.
    sample = rgb[::max(1, rgb.shape[0] // 256), ::max(1, rgb.shape[1] // 256)]
    if np.any(np.ptp(sample, axis=2) > 12):
        for channel in cv2.split(rgb):
            binary = cv2.bitwise_or(binary, cv2.threshold(
                channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    image_area = image.width * image.height
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if len(polygon) != 4 or w < 45 or h < 22 or area < 0.0004 * image_area or area > 0.35 * image_area:
            continue
        if not 0.35 <= w / max(h, 1) <= 12:
            continue
        candidates.append((x, y, x + w, y + h))
    # Keep the largest version of a shape when nested contours describe its
    # border. Retain overlapping boxes when they are clearly separate nodes.
    boxes = []
    for box in sorted(candidates, key=lambda b: (b[1], b[0], -((b[2]-b[0])*(b[3]-b[1])))):
        area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        duplicate = False
        for old in boxes:
            ix = max(0, min(box[2], old[2]) - max(box[0], old[0]))
            iy = max(0, min(box[3], old[3]) - max(box[1], old[1]))
            overlap = ix * iy
            old_area = max(1, (old[2] - old[0]) * (old[3] - old[1]))
            if overlap / min(area, old_area) >= 0.88:
                duplicate = True
                break
        if not duplicate:
            boxes.append(box)
    nodes = []
    used = set()
    for x1, y1, x2, y2 in boxes:
        contained = [(idx, item) for idx, item in enumerate(items)
                     if idx not in used and x1 <= item["cx"] <= x2 and y1 <= item["cy"] <= y2]
        if not contained:
            continue
        used.update(idx for idx, _ in contained)
        contained.sort(key=lambda pair: (pair[1]["cy"], pair[1]["left"]))
        label = " ".join(item["text"] for _, item in contained).strip()
        nodes.append({"id": f"N{len(nodes)+1}", "label": label,
                      "bbox": [int(x1), int(y1), int(x2), int(y2)]})
    # Keep OCR regions outside detected boxes visible too. A diagram can mix
    # boxed steps with free-standing start/end labels or annotations.
    ungrouped = [item for idx, item in enumerate(items) if idx not in used and item["text"]]
    if nodes and ungrouped:
        ungrouped.sort(key=lambda item: (item["cy"], item["left"]))
        nodes.extend({"id": f"N{len(nodes)+i+1}", "label": item["text"],
                      "bbox": [int(item["left"]), int(item["top"]),
                               int(item["right"]), int(item["bottom"])]}
                     for i, item in enumerate(ungrouped))
    if not nodes:
        nodes = [{"id": f"N{i+1}", "label": item["text"],
                  "bbox": [int(item["left"]), int(item["top"]),
                           int(item["right"]), int(item["bottom"])]}
                 for i, item in enumerate(items) if item["text"]]
    return nodes, len(boxes)


def extract(image, prompt=""):
    items = _ocr(image)
    all_text = [item["text"] for item in items if item["text"]]
    lowered = prompt.lower()
    is_table = "focus on table" in lowered
    is_flow = "focus on flowchart" in lowered
    is_chart = "focus on chart" in lowered
    uncertainties = ["OCR and geometric rules are used; verify all recognized text against the image."]
    data = {"text": [], "tables": [], "flowcharts": [], "charts": [], "uncertainties": uncertainties}

    if is_table or (not is_flow and not is_chart and "focus on text" not in lowered):
        table = _grid_table(items, image) or _line_table(items, image.width, image.height)
        if table:
            data["tables"].append(table)
            data["text"] = []
            uncertainties.append("Rows and columns were inferred from text alignment; inspect blank, merged, or wrapped cells.")
        elif is_table:
            uncertainties.append("No consistent row and column alignment was detected; text was not forced into a table.")

    if is_flow:
        # Every OCR region is emitted as a node; connector inference is
        # intentionally conservative until reliable arrowhead tracing exists.
        if len(items) >= 2:
            nodes, detected_boxes = _flow_nodes(items, image)
            data["flowcharts"].append({"title": "", "nodes": nodes, "edges": []})
            uncertainties.append(f"Found {detected_boxes} candidate shape boxes and emitted {len(nodes)} text nodes; arrow directions and branch connections were not inferred.")

    if is_chart:
        if items:
            # Preserve OCR text as axis/tick candidates; never manufacture
            # numeric data points from pixel positions.
            top_band = [i["text"] for i in items if i["cy"] > image.height * 0.75]
            left_band = [i["text"] for i in items if i["cx"] < image.width * 0.22]
            data["charts"].append({"title": "", "x_axis": {"label": "", "scale": "unknown", "ticks": top_band},
                                   "y_axis": {"label": "", "scale": "unknown", "ticks": left_band}, "points": []})
            uncertainties.append("Chart text and likely tick labels were collected; plotted values and series were not digitized.")

    if not data["tables"] and not data["flowcharts"] and not data["charts"]:
        data["text"] = all_text
    low_confidence = [item["text"] for item in items if item["score"] < 0.80]
    if low_confidence:
        uncertainties.append(f"{len(low_confidence)} OCR regions have confidence below 0.80; check: " + ", ".join(repr(x) for x in low_confidence[:12]))
    ambiguous = [item for item in items if re.search(r"(?<!\w)(?:TT|T\s+T|π|Π)(?!\w)", item["text"])]
    for item in ambiguous:
        uncertainties.append(
            f"Possible TT/π symbol ambiguity in {item['text']!r} near "
            f"({int(item['left'])}, {int(item['top'])}); OCR does not return alternate candidates, so verify against the image."
        )
    return data
