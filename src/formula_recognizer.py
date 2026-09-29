"""公式识别（PP-FormulaNet_plus-M 封装）

对版面分析检出的 equation 区域裁剪图识别为 LaTeX 表达式。

推理代码 vendor 自 RapidDoc 的 rapid_formula_self 子包（Apache-2.0，
许可证见 rapid_formula_self/LICENSE），已剥离 torch / openvino 分支，
仅保留 onnxruntime 路径。模型不随包分发 —— 由不忙脚本盒子统一下载，
运行时经 /api/model/path 换取本地路径；盒子外运行时退回库自带按需下载。
"""

import os


def _fallback_model_dir():
    """盒子外运行时的模型落盘位置。

    vendor 包的默认落盘目录在自己的包目录里，那样盒子外跑一次就会往脚本
    目录塞 600MB 模型（还会被打包带走），所以统一挪到用户缓存目录。
    """
    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    return os.path.join(root, "BmOCR", "models")


class FormulaRecognizer:
    """基于 PP-FormulaNet_plus-M 的公式 → LaTeX 识别器。"""

    def __init__(self, model_path=None):
        self.model_path = model_path
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            # 必须在 import 前设好：vendor 包在导入时就会解析并创建模型目录
            os.environ.setdefault("RAPID_MODELS_DIR", _fallback_model_dir())

            from src.rapid_formula_self import (ModelType, RapidFormula,
                                                RapidFormulaInput)
            cfg = RapidFormulaInput(
                model_type=ModelType.PP_FORMULANET_PLUS_M,
                model_dir_or_path=self.model_path or None,
            )
            self._engine = RapidFormula(cfg)
        return self._engine

    def recognize(self, crop_img):
        """对公式裁剪图识别，返回 LaTeX 字符串或 None。"""
        try:
            out = self._get_engine()([crop_img])
            latex = (out[0].rec_formula or "").strip()
            return latex or None
        except Exception:
            return None


def crop_region(img, bbox, pad=6):
    """裁剪区域（加边距，越界钳制）。"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()
