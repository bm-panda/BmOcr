"""OCR-to-Markdown 转换器（配合 RapidLayout 版面检测与 RapidTable 表格重建）

接收 RapidOCR 的文字识别结果 + LayoutAnalyzer 的版面区域，
基于 region 标签决定排版结构。
"""

import re
import numpy as np


class MarkdownConverter:
    """基于版面检测的 OCR-to-Markdown 转换器。"""

    # 列表前缀模式（仅 row-level fallback 用）
    _ORDERED_PREFIX = re.compile(r"^(\d+)[.、)]\s*")
    _UNORDERED_PREFIX = re.compile(r"^[-*+•·▪▸→]\s*")

    # region label → line class 映射（内部统一标签集）
    _LABEL_MAP = {
        "title": "heading",
        "text": "normal",
        "table": "table",
        "figure_caption": "normal",
        "table_caption": "normal",
        "header": "normal",
        "footer": "normal",
        "reference": "normal",
        "equation": "normal",
        "figure": "normal",  # figure 内文本按普通文本渲染，避免误判丢内容
    }

    # 重叠 region 匹配优先级（table 优先，figure 最低）
    _REGION_PRIORITY = {
        "table": 5,
        "title": 4,
        "text": 4,
        "table_caption": 3,
        "figure_caption": 3,
        "equation": 3,
        "reference": 3,
        "header": 3,
        "footer": 3,
        "figure": 1,
    }

    def __init__(self, boxes, txts, regions=None, word_results=None, table_markdowns=None,
                 formula_markdowns=None):
        self.boxes = boxes
        self.txts = txts
        self.regions = regions or []
        self.word_results = word_results
        # {(x1,y1,x2,y2): (md_table, cells) | md_table} — 已重建的表格块
        self.table_markdowns = table_markdowns or {}
        # {(x1,y1,x2,y2): latex} — 已识别的公式块
        self.formula_markdowns = formula_markdowns or {}

    # ----------------------------------------------------------------
    # public
    # ----------------------------------------------------------------

    def convert(self) -> str:
        if self.boxes is None or self.txts is None or len(self.txts) == 0:
            return "没有检测到任何文本。"

        self._table_has_sep = False
        self._table_claimed = set()
        self._formula_claimed = set()

        # step 1: build typed items
        items = [
            {"text": text, "props": self._box_props(box)}
            for box, text in zip(self.boxes, self.txts)
        ]

        # step 2: sort by reading order (top→bottom, left→right)
        items.sort(key=lambda it: (it["props"]["center_y"], it["props"]["left"]))

        # step 3: group into lines
        lines = self._group_lines(items)

        # step 4: classify lines via layout regions
        self._classify_by_regions(lines)

        # step 4b: split headings into levels by relative height
        self._assign_heading_levels(lines)

        # step 5: fallback — list detection for lines still "normal"
        self._detect_lists_fallback(lines)

        # step 6: render
        return self._render(lines)

    # ----------------------------------------------------------------
    # box helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _box_props(box):
        ys, xs = box[:, 1], box[:, 0]
        t, b, l, r = float(np.min(ys)), float(np.max(ys)), float(np.min(xs)), float(np.max(xs))
        return {
            "top": t, "bottom": b, "left": l, "right": r,
            "height": b - t, "width": r - l,
            "center_y": (t + b) / 2, "center_x": (l + r) / 2,
        }

    @staticmethod
    def _merge_props(a, b):
        t = min(a["top"], b["top"])
        bm = max(a["bottom"], b["bottom"])
        l = min(a["left"], b["left"])
        r = max(a["right"], b["right"])
        return {"top": t, "bottom": bm, "left": l, "right": r,
                "height": bm - t, "width": r - l,
                "center_y": (t + bm) / 2, "center_x": (l + r) / 2}

    # ----------------------------------------------------------------
    # line grouping
    # ----------------------------------------------------------------

    @staticmethod
    def _same_line(cur, line):
        ot = max(line["top"], cur["top"])
        ob = min(line["bottom"], cur["bottom"])
        overlap = max(0.0, ob - ot)
        mh = max(1.0, min(cur["height"], line["height"]))
        return (overlap / mh) > 0.5 or abs(cur["center_y"] - line["center_y"]) < mh * 0.35

    def _group_lines(self, items):
        lines = []
        for item in items:
            target = None
            for ln in lines:
                if self._same_line(item["props"], ln["props"]):
                    target = ln
                    break
            if target is None:
                lines.append({"items": [item], "props": dict(item["props"]),
                              "class": "normal", "detail": None})
            else:
                target["items"].append(item)
                target["props"] = self._merge_props(target["props"], item["props"])
        lines.sort(key=lambda ln: (ln["props"]["top"], ln["props"]["left"]))
        return lines

    # ----------------------------------------------------------------
    # region-based classification
    # ----------------------------------------------------------------

    def _classify_by_regions(self, lines):
        """用 LayoutAnalyzer 检测到的 region 给每行分类。"""
        if not self.regions:
            return

        for ln in lines:
            cx = ln["props"]["center_x"]
            cy = ln["props"]["center_y"]

            # 在所有包含该行的 region 中按优先级+最小面积选最优
            best = None
            best_score = -1
            best_area = None
            for r in self.regions:
                x1, y1, x2, y2 = r["bbox"]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    pri = self._REGION_PRIORITY.get(r["label"], 2)
                    area = (x2 - x1) * (y2 - y1)
                    if (pri > best_score or
                            (pri == best_score and best_area is not None and area < best_area)):
                        best = r
                        best_score = pri
                        best_area = area
            if best is None:
                continue

            label = best["label"]

            if label == "table":
                key = tuple(round(float(v), 1) for v in best["bbox"])
                ln["region_bbox"] = best["bbox"]
                md_info = self.table_markdowns.get(key)
                if md_info is not None:
                    md_text = md_info[0] if isinstance(md_info, tuple) else md_info
                    if md_text:
                        if key not in self._table_claimed:
                            self._table_claimed.add(key)
                            ln["class"] = "table"
                            cells = md_info[1] if isinstance(md_info, tuple) else None
                            ln["detail"] = {
                                "md_table": md_text,
                                "cells": cells,
                                "region_bbox": best["bbox"],
                            }
                        else:
                            ln["class"] = "skip"
                            ln["detail"] = None
                        continue

            if label == "equation":
                key = tuple(round(float(v), 1) for v in best["bbox"])
                ln["region_bbox"] = best["bbox"]
                latex = self.formula_markdowns.get(key)
                if latex:
                    if key not in self._formula_claimed:
                        self._formula_claimed.add(key)
                        ln["class"] = "equation"
                        ln["detail"] = {"latex": latex, "region_bbox": best["bbox"]}
                    else:
                        ln["class"] = "skip"
                        ln["detail"] = None
                    continue

            ln["class"] = self._LABEL_MAP.get(label, "normal")
            ln["detail"] = None

            # propagate table class to all items in the same table region
            if ln["class"] == "table":
                ln["region_bbox"] = best["bbox"]

    def _assign_heading_levels(self, lines):
        """把 layout 判出的标题按相对字号分成 1-3 级。

        版面模型只输出一个 "title" 类，不带层级信息；这里用标题之间的相对行高
        推断层级：最高的一批算一级，依次往下。只有一个标题时按一级处理。
        """
        heads = [ln for ln in lines if ln["class"] == "heading"]
        if not heads:
            return
        tallest = max(ln["props"]["height"] for ln in heads)
        if len(heads) == 1 or tallest <= 0:
            for ln in heads:
                ln["detail"] = 1
            return
        for ln in heads:
            ratio = ln["props"]["height"] / tallest
            if ratio >= 0.85:
                ln["detail"] = 1
            elif ratio >= 0.65:
                ln["detail"] = 2
            else:
                ln["detail"] = 3

    # ----------------------------------------------------------------
    # list fallback (for lines the layout model didn't classify)
    # ----------------------------------------------------------------

    def _detect_lists_fallback(self, lines):
        """对仍为 normal 的行做列表检测。"""
        i = 0
        while i < len(lines):
            ln = lines[i]
            if ln["class"] not in ("normal",):
                i += 1
                continue
            text = "".join(it["text"] for it in ln["items"])
            prefix_type = None
            if self._ORDERED_PREFIX.match(text):
                prefix_type = "ordered"
            elif self._UNORDERED_PREFIX.match(text):
                prefix_type = "unordered"
            if not prefix_type:
                i += 1
                continue

            j = i + 1
            while j < len(lines):
                if lines[j]["class"] not in ("normal",):
                    break
                t = "".join(it["text"] for it in lines[j]["items"])
                if not (self._ORDERED_PREFIX.match(t) or self._UNORDERED_PREFIX.match(t)):
                    break
                j += 1
            if j - i >= 2:
                for k in range(i, j):
                    lines[k]["class"] = "list"
                    lines[k]["detail"] = prefix_type
                i = j
            else:
                i += 1

    # ----------------------------------------------------------------
    # table column helper
    # ----------------------------------------------------------------

    @staticmethod
    def _compute_table_columns(items):
        """从一行 items 中聚类出列结构。"""
        widths = [it["props"]["width"] for it in items]
        median_w = sorted(widths)[len(widths) // 2] if widths else 20
        gap = max(median_w * 0.25, 6)
        centers = sorted([it["props"]["center_x"] for it in items])
        clusters = []
        for c in centers:
            if not clusters:
                clusters.append([c])
            elif c - clusters[-1][-1] < gap:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        return [int(np.mean(cl) / 15) * 15 for cl in clusters]

    # ----------------------------------------------------------------
    # markdown rendering
    # ----------------------------------------------------------------

    @staticmethod
    def _gap_text(gap, prev_w, cur_w):
        if gap <= 1:
            return ""
        ref = max(1.0, min(prev_w, cur_w))
        n = max(1, int(round(gap / max(1.0, ref * 0.6))))
        return " " * min(n, 8)

    def _render(self, lines):
        md_lines = []
        prev_props = None
        in_table = False

        for idx, ln in enumerate(lines):
            # skip lines consumed by a reconstructed table block
            if ln["class"] == "skip":
                prev_props = ln["props"]
                continue

            # --- reconstructed table block (whole-table render) ---
            if (ln["class"] == "table"
                    and isinstance(ln["detail"], dict)
                    and ln["detail"].get("md_table")):
                if md_lines and md_lines[-1] != "":
                    md_lines.append("")
                md_lines.append(ln["detail"]["md_table"])
                md_lines.append("")
                in_table = False
                self._table_has_sep = False
                prev_props = ln["props"]
                continue

            # --- reconstructed formula block ---
            if (ln["class"] == "equation"
                    and isinstance(ln["detail"], dict)
                    and ln["detail"].get("latex")):
                if md_lines and md_lines[-1] != "":
                    md_lines.append("")
                md_lines.append("$$")
                md_lines.append(ln["detail"]["latex"])
                md_lines.append("$$")
                md_lines.append("")
                prev_props = ln["props"]
                continue

            # --- paragraph break ---
            if prev_props is not None:
                vgap = ln["props"]["top"] - prev_props["bottom"]
                max_h = max(ln["props"]["height"], prev_props["height"])
                if ln["class"] == "heading":
                    md_lines.append("")
                elif ln["class"] == "table" and not in_table:
                    md_lines.append("")
                elif vgap > max_h * 0.7 and ln["class"] != "table":
                    md_lines.append("")

            # track table block
            if ln["class"] == "table":
                in_table = True
            else:
                if in_table:
                    self._table_has_sep = False
                in_table = False

            # --- table ---
            if ln["class"] == "table":
                sig = self._compute_table_columns(ln["items"])
                if len(sig) < 2:
                    # fallback: treat as normal text
                    text = self._render_line_text(ln)
                    md_lines.append(text)
                    prev_props = ln["props"]
                    continue

                items = sorted(ln["items"], key=lambda it: it["props"]["center_x"])
                cells = [""] * len(sig)
                for it in items:
                    cx = it["props"]["center_x"]
                    dists = [abs(cx - s) for s in sig]
                    col = dists.index(min(dists))
                    if cells[col]:
                        cells[col] += " " + it["text"]
                    else:
                        cells[col] = it["text"]
                md_lines.append("| " + " | ".join(cells) + " |")

                if not self._table_has_sep:
                    md_lines.append("|" + "|".join(" --- " for _ in sig) + "|")
                    self._table_has_sep = True
                prev_props = ln["props"]
                continue

            # --- list ---
            if ln["class"] == "list":
                text = "".join(it["text"] for it in ln["items"])
                if ln["detail"] == "ordered":
                    text = re.sub(r"^\d+[.、)]\s*", "", text)
                    md_lines.append(f"1. {text}")
                else:
                    text = re.sub(r"^[-*+•·▪▸→]\s*", "", text)
                    md_lines.append(f"- {text}")
                prev_props = ln["props"]
                continue

            # --- heading ---
            if ln["class"] == "heading":
                text = self._render_line_text(ln)
                level = ln["detail"] if isinstance(ln["detail"], int) else 1
                md_lines.append(f"{'#' * min(max(level, 1), 3)} {text}")
                prev_props = ln["props"]
                continue

            # --- normal paragraph ---
            text = self._render_line_text(ln)
            if (prev_props is not None
                    and md_lines
                    and ln["props"]["top"] - prev_props["bottom"] < ln["props"]["height"] * 0.5
                    and ln["class"] == "normal"
                    and (idx == 0 or lines[idx - 1]["class"] == "normal")):
                md_lines[-1] += " " + text
            else:
                md_lines.append(text)
            prev_props = ln["props"]

        return "\n".join(md_lines)

    @staticmethod
    def _render_line_text(ln):
        items = sorted(ln["items"], key=lambda it: it["props"]["left"])
        if not items:
            return ""
        parts = [items[0]["text"]]
        for i in range(1, len(items)):
            gap = items[i]["props"]["left"] - items[i - 1]["props"]["right"]
            parts.append(MarkdownConverter._gap_text(
                gap, items[i - 1]["props"]["width"], items[i]["props"]["width"]))
            parts.append(items[i]["text"])
        return "".join(parts)
