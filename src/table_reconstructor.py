"""表格结构重建（rapid_table 封装）

对版面分析检出的 table 区域裁剪图做表格结构识别（SLANetPlus），
输出 Markdown 表格与单元格坐标/文本/跨度（坐标在 crop 坐标系，由调用方偏移回整图）。

- 小表：可复用全图 OCR 结果（ocr_result），不重复推理
- 大表：等比缩放到 SLANet 舒适范围后同坐标系重建，避免乱码/丢行
"""

import cv2
import numpy as np
from html.parser import HTMLParser


class _TableHTMLParser(HTMLParser):
    """提取 HTML 表格的行/单元格及 colspan/rowspan。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self._cur_row = None
        self._in_td = False
        self._cur_cell = None
        self._col = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._cur_row = []
            self._col = 0
        elif tag in ("td", "th") and self._cur_row is not None:
            self._in_td = True
            self._cur_cell = {
                "text": "",
                "colspan": int(a.get("colspan", 1)),
                "rowspan": int(a.get("rowspan", 1)),
                "row": len(self.rows),
                "col": self._col,
            }
            self._col += 1

    def handle_data(self, data):
        if self._in_td and self._cur_cell is not None:
            self._cur_cell["text"] += data

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cur_row is not None:
            if self._cur_cell is not None:
                self._cur_row.append(self._cur_cell)
                self._cur_cell = None
            self._in_td = False
        elif tag == "tr" and self._cur_row is not None:
            self.rows.append(self._cur_row)
            self._cur_row = None


def html_to_markdown(html):
    """RapidTable 输出的 HTML 表格 → (Markdown 表格, cell 元数据列表)。

    rowspan/colspan 无法在 Markdown 中表达，采用复制单元格填充的近似方案：
    - colspan：文本放入首列，后续列留空
    - rowspan：文本放入首行，后续行该列留空
    """
    if not html:
        return None, []
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
    except Exception:
        return None, []
    if not parser.rows:
        return None, []

    cells_meta = []
    matrix = []
    for row in parser.rows:
        if not row:
            continue
        cells = []
        for cell in row:
            cells_meta.append({
                "row": cell["row"], "col": cell["col"],
                "colspan": cell["colspan"], "rowspan": cell["rowspan"],
                "text": cell["text"].strip().replace("\n", " "),
            })
            for _ in range(max(1, cell["colspan"])):
                cells.append(cell["text"].strip().replace("\n", " "))
        if not cells:
            continue
        matrix.append(cells)

    if not matrix:
        return None, []

    ncols = max(len(r) for r in matrix)
    matrix = [r + [""] * (ncols - len(r)) for r in matrix]
    # 过滤全空行（如底部边框产生的幻影行）
    matrix = [r for r in matrix if any(c.strip() for c in r)]
    if not matrix:
        return None, []

    lines = ["| " + " | ".join(matrix[0]) + " |"]
    lines.append("|" + "|".join([" --- "] * ncols) + "|")
    for r in matrix[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines), cells_meta


class TableReconstructor:
    """基于 rapid_table 的表格重建。"""

    def __init__(self, model_type="slanet_plus"):
        self.model_type = model_type
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from rapid_table import ModelType, RapidTable, RapidTableInput
            self._engine = RapidTable(
                RapidTableInput(model_type=ModelType(self.model_type))
            )
        return self._engine

    def reconstruct(self, crop_img, ocr_engine, ocr_result=None, box_thresh=0.5, max_dim=1000):
        """对表格裁剪图重建表格。

        ocr_result: 可选 (boxes, txts, scores)，坐标须与 crop_img 一致
                    （可传全图 OCR 平移结果以复用，避免重复推理）。
        - crop 最大边 <= max_dim 且有 ocr_result：直接复用
        - 否则（大表）等比缩到 max_dim 内，对缩放图重跑 OCR，同一坐标系重建

        Returns (md_table, cells) 或 (None, None)。
        cells: [{row, col, colspan, rowspan, text, box}]，box 坐标为 crop 坐标系。
        """
        scale = 1.0
        img_for_model = crop_img

        if ocr_result is not None:
            boxes, txts, scores = ocr_result
        else:
            h, w = crop_img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img_for_model = cv2.resize(
                    crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            res = ocr_engine(img_for_model, box_thresh=box_thresh)
            if res.boxes is None or len(res.boxes) == 0:
                return None, None
            boxes, txts, scores = res.boxes, res.txts, res.scores

        if boxes is None or len(boxes) == 0:
            return None, None

        try:
            table_res = self._get_engine()(
                img_for_model,
                ocr_results=[(boxes, txts, scores)],
            )
        except Exception:
            return None, None

        if not table_res.pred_htmls:
            return None, None
        md_table, cells_meta = html_to_markdown(table_res.pred_htmls[0])
        if md_table is None:
            return None, None

        inv = 1.0 / scale
        cell_bboxes = table_res.cell_bboxes[0] if table_res.cell_bboxes else []
        cb = np.asarray(cell_bboxes)
        if cb.size == 0:
            cell_bboxes = []
        elif cb.ndim == 2 and cb.shape[1] == 8:
            cell_bboxes = cb.reshape(-1, 4, 2)
        elif cb.ndim == 3:
            cell_bboxes = cb
        else:
            cell_bboxes = []

        cells = []
        for i, box in enumerate(cell_bboxes):
            b = np.asarray(box)
            if b.size == 0 or b.ndim != 2 or b.shape[1] != 2:
                continue
            xs, ys = b[:, 0], b[:, 1]
            # row/col 显式置 None：cell_bboxes 可能多于 HTML 解析出的单元格
            # （rowspan/colspan 场景），这类"孤儿"单元格只有坐标、定不到行列，
            # 下游 Excel 导出会跳过它们，ocr_data 里保留坐标供调用方使用。
            cell = {
                "row": None, "col": None,
                "colspan": 1, "rowspan": 1, "text": "",
                "box": [int(xs.min() * inv), int(ys.min() * inv),
                        int(xs.max() * inv), int(ys.max() * inv)],
            }
            if i < len(cells_meta):
                m = cells_meta[i]
                cell.update({
                    "row": m["row"], "col": m["col"],
                    "colspan": m["colspan"], "rowspan": m["rowspan"],
                    "text": m["text"],
                })
            cells.append(cell)

        return md_table, cells


def crop_table_region(img, bbox, pad=8):
    """裁剪表格区域（加边距，越界钳制）。"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None, (0, 0)
    return img[y1:y2, x1:x2].copy(), (x1, y1)