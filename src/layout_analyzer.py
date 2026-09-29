"""版面检测分析器（rapid_layout 薄封装）

固定使用 PP-DocLayoutV3（24 类），输出统一标签（text/title/table/figure/
figure_caption/table_caption/equation/header/footer/reference）。
内置 letterbox 等比缩放、NMS 与跨类重叠去重。

模型由不忙脚本盒子统一下载，运行时经 /api/model/path 换取本地路径；
盒子外运行时 model_path 为空，退回 rapid_layout 自带的按需下载。
"""


# 固定模型：参数值即 rapid_layout ModelType 枚举取值（小写）
_MODEL_TYPE = "pp_doc_layoutv3"

# PP-DocLayoutV3 原始标签 → 统一标签
_LABEL_NORMALIZE = {
    "text": "text", "content": "text", "aside_text": "text",
    "abstract": "text", "algorithm": "text", "vertical_text": "text",
    "number": "text", "footnote": "text",
    "doc_title": "title", "paragraph_title": "title",
    "table": "table",
    "image": "figure", "figure": "figure", "chart": "figure", "seal": "figure",
    "figure_title": "figure_caption",
    "display_formula": "equation", "inline_formula": "equation",
    "formula_number": "equation",
    "header": "header", "header_image": "figure",
    "footer": "footer", "footer_image": "figure",
    "reference": "reference", "reference_content": "reference",
}


# region 标签优先级（重叠时保留高优先级，table > 文本 > 图）
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


def _nms(regions, iou_thresh=0.5):
    """跨类 NMS：按 (标签优先级, 置信度) 降序贪心，重叠超过阈值的低优先级 region 丢弃。"""
    if len(regions) <= 1:
        return regions

    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a["bbox"]
        bx1, by1, bx2, by2 = b["bbox"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        a_area = (ax2 - ax1) * (ay2 - ay1)
        b_area = (bx2 - bx1) * (by2 - by1)
        union = a_area + b_area - inter
        return inter / union if union > 0 else 0.0

    def _key(i):
        return (_REGION_PRIORITY.get(regions[i]["label"], 2), regions[i]["confidence"])

    order = sorted(range(len(regions)), key=_key, reverse=True)
    keep = []
    for i in order:
        if all(_iou(regions[i], regions[j]) <= iou_thresh for j in keep):
            keep.append(i)
    return [regions[i] for i in sorted(keep)]


class LayoutAnalyzer:
    """基于 rapid_layout（PP-DocLayoutV3）的版面分析器。"""

    def __init__(self, conf_thresh=0.5, model_path=None, iou_thresh=0.5):
        self.conf_thresh = conf_thresh
        self.model_path = model_path
        self.iou_thresh = iou_thresh
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from rapid_layout import EngineType, ModelType, RapidLayout
            kwargs = {}
            if self.model_path:
                # 盒子已下好的 onnx；给了就不触发 rapid_layout 自己的下载
                kwargs["model_dir_or_path"] = self.model_path
            self._engine = RapidLayout(
                model_type=ModelType(_MODEL_TYPE),
                engine_type=EngineType.ONNXRUNTIME,
                conf_thresh=self.conf_thresh,
                **kwargs,
            )
        return self._engine

    def analyze(self, img):
        """对 BGR ndarray 图片做版面分析，返回统一标签的 region 列表。"""
        try:
            results = self._get_engine()(img)
        except Exception:
            return []

        boxes = results.boxes if results.boxes is not None else []
        class_names = results.class_names if results.class_names is not None else []
        scores = results.scores if results.scores is not None else []
        regions = []
        for box, label, score in zip(boxes, class_names, scores):
            unify = _LABEL_NORMALIZE.get(label)
            if unify is None:
                continue
            regions.append({
                "label": unify,
                "confidence": float(score),
                "class_id": None,
                "bbox": [float(v) for v in box],
            })
        return _nms(regions, self.iou_thresh)
