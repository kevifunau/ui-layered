# RapidOCR（2.1.1 文字提取所用 OCR）

## 功能

- RapidOCR 是 PaddleOCR（PP-OCR）模型的 ONNX 化开源实现：不装 PaddlePaddle 框架即可跨平台推理，本仓库用 `rapidocr-onnxruntime` 后端（CPU）。
- 内部三段式管线：**文本检测**（DB 网络，输出文字区域四边形）→ **方向分类** → **文本识别**（CRNN，输出文本与置信度）。
- 在本管线中的职责：2.1.1 对原图做首次全量文字提取，其结果是 2.1.2 去字、2.1.3 样式提取、2.2.1 LLM 规划（OCR id 列表）的共同输入。

## 输入

- 一张图像，四种形式均可：cv2 解码的 BGR ndarray / 文件路径 / URL / bytes。本管线传 `cv2.imdecode` 得到的 uint8 BGR ndarray（H×W×3）。
- 可选参数：det / cls / rec 三段开关与阈值（`box_thresh`、`text_score`、`max_side_len` 等）。本管线用默认模型参数，置信度过滤在管线外部做（见下）。

```python
from rapidocr_onnxruntime import RapidOCR
ocr = RapidOCR()
result, elapse = ocr(bgr_ndarray)   # result: [[quad, text, score], ...] 或 None
```

## 输出

- `result, elapse = ocr(img)` 二元组：
    - `result`：列表，每项 `[quad, text, score]`
        - `quad`：4×2 浮点四点，文字框四边形顶点（左上→右上→右下→左下）；
        - `text`：识别出的字符串；
        - `score`：0–1 置信度；
        - 整图无文字时 `result` 为 `None`。
    - `elapse`：det / cls / rec 三段耗时，用于性能分析。
- 本管线对输出的后处理（2.1.1）：
    - `score < 0.35` 丢弃；
    - 单个拉丁字母且 `score < 0.85` 视为图标误识别，丢弃；
    - `score < 0.85` 且 高/宽 ≥ 2.4 的竖排装饰候选，丢弃；
    - ≥3 位纯数字放大 2x / 3x 裁剪复审一次；
    - 通过后赋稳定 id `text_001…`（按检出顺序），2.2.1 的 prompt 以该 id 列表引用 OCR 区域，`raster_text_ids` / `text_corrections` 也用同一套 id 回答。