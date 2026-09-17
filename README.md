# UI Layered — 游戏 UI 图层分解（对知乎方案5文章的复刻）

复刻对象：[游戏AI拼UI工作流（方案5-图层拆分算法突破）](https://zhuanlan.zhihu.com/p/2071328022855153618)（罗培羽，2026-08-21）。
单张 UI 预览图 → OCR 去字 → LLM 图层规划 → SAM 分割 → 遮挡修补 → 分层导出 + 回拼。

## 实现原理

目标是把一张"拍扁"的游戏 UI 截图还原成一套可编辑的分层素材（每层一张带 alpha 的 PNG + 文字样式参数 + 可无损回拼）。核心设计是五件事的分工：**文字与图形分离、LLM 出语义规划、SAM 出像素 mask、父子链驱动遮挡修补、回拼无损做质量闸**。

```
原图 ──OCR──► 文字mask ──inpaint──► 无字底板 ──┐
  │                                            ├─► 视觉LLM ─► plan.json(图层规划)
  └───────────────────────────────────────────┘        │ bbox+正负点+z_order+父子链
文字参数(内容/样式) 存档 ──────────────┐                 ▼
                                      │        SAM 逐实例分割 ─► 候选打分 ─► mask
                                      │                 ▼
                                      │     遮挡分析(父子链+像素归属) ─► alpha + 孔洞
                                      │                 ▼
                                      │     孔洞修补(平色填补 / 生图模型 / 传统NS)
                                      ▼                 ▼
                              文字重渲染 ◄── 底板+图层按 z_order 回拼 ─► 无损校验
```

1. **文字剥离（2.1.1 / 2.1.2）**：RapidOCR 找出全部文字区域（带置信度与形态过滤）。对每个区域取其四边窄带的中位数作为局部底色，按"像素与底色的 RGB 距离"分割出字形 mask（Otsu 自适应阈值 + 闭运算 + 椭圆膨胀），全部文字的 mask 并集做 `cv2.inpaint` 得到**无字底板**。文字从此不当像素处理——内容、位置、颜色、字号、字重、描边全部参数化存档（2.1.3），最后一步才重新渲染。去字既让 LLM 看清被文字挡住的图层结构，也让后续取色/修补不被笔画污染。
2. **LLM 图层规划（2.2.1）**：原图 + 无字底板 + 带稳定 id 的 OCR 列表一起送视觉大模型（Qwen-VL），输出每个图元的语义描述：bbox 与正/负点提示、`z_order`、`parent_query_id`（哪些控件挂在哪个面板上）、修补方式（平色 `surface` / 纹理 `image` / 不动 `none`）、哪些是美术字（`raster_text_ids`，保留原像素不去字）、OCR 漏字修正。LLM 只做判断，不画 mask。
3. **SAM 像素分割（2.2.2–2.2.4）**：SAM 对整图编码一次，按每个实例的 bbox+正负点提示出 3 个候选 mask，按原文公式打分（SAM IoU + 正点命中 − 负点命中 + 框内包含率 + 外接框 IoU）挑最优，再经闭运算和三条连通域规则清理散块。
4. **遮挡与成层（layer build）**：重叠 mask 之间按"离各自 bbox 中心最近"裁决像素归属，得到每层的 alpha；父子链加几何遮挡（z 更高且 mask 相触者）算出每层被谁盖住——被盖住的区域就是待修补的**孔洞**。
5. **孔洞修补（2.3）**：平色图元用自身可见像素的中位数/主簇色填补并混合接缝；纹理图元打包成图集送生图模型（Seedream / 本地 Flux），带黑洞率+材质色 sanity 自检、多 seed 重试、失败回退传统 NS 修补；背景按 LLM 给出的 `background_repair.mode` 单独修。
6. **导出与自检（2.4）**：每层导出带 alpha 的 PNG；底板 + 图层按 z_order 回拼 + 文字按存档样式重渲染。最后做**无损校验**：回拼图与原图在孔洞和接缝带之外必须零差异——分解的正确性由此闭环验证。

## 步骤与原文对应

| 原文 | 实现 | 说明 |
|---|---|---|
| 2.1.1 OCR 文字提取 | `steps/text.py TextExtractStep` | RapidOCR；置信度 0.35；单拉丁字母 <0.85 视为图标误识别；低置信且 h/w≥2.4 的竖排装饰剔除；≥3 位纯数字放大 2x/3x 复审（仓库扩展） |
| 2.1.2 预览图去除文字 | `steps/text.py` | OCR 四边形 allowed mask；边带 `clamp(min(h,w)/4,1,3)` 取 RGB **中位数**底色；`3d ≥ max(12, 0.65·Otsu)`；3×3 闭运算；面积 <1.5% 或 >68% → `coarse`；椭圆膨胀 `clamp(0.16h,3,8)`/`clamp(0.09h,2,6)`；`cv2.inpaint(r=5, TELEA)`。产物 `01_text_removed.png` + `01a_text_mask.png` |
| 2.2.1 LLM 图层规划 | `steps/plan.py PlanStep` + `prompts.PLAN_PROMPT` | 输入 = 原图 + **初次全 OCR 去字图** + 逐字复刻的 prompt + **带稳定 id 的 OCR 列表**；输出 schema 含 `raster_text_ids` / `text_corrections` / `parent_query_id` / 多实例 `geometry_hints` |
| 2.1.4 美术字 | `steps/text.py TextFinalizeStep` | `raster_text_ids` 命中的 OCR 区域从 mask 中删除（保留原像素）；按原文契约要求每个 raster id 必须同时有一个 `kind=logo` query，二者交叉校验后才生效（见下） |
| 2.1.3 文字样式 | `steps/text.py` | 32 级量化桶 ×(24+与背景距离)；YaHei/Arial；`0.88h/0.82h` 下限 8px；coverage≥0.17→700；Rec.709 描边色；描边宽 `0.045h/0.03h` clamp 0–2 |
| 2.2.2 图像分割 | `steps/segmentation.py` | SAM1 ViT-B，整图编码一次，逐实例提示分割（multimask=3 候选） |
| 2.2.3 候选提取 | `cand_score` | `SAM + 0.22·正点 − 0.38·负点 + 0.10·框内包含 + 0.06·外接框IoU`（逐字一致） |
| 2.2.4 修补+连通域 | `cc_filter` | 3×3 闭运算；①含正点必留 ②无正点留最大 ③**≥最大组件 8% 也保留**（旧版漏了③） |
| 2.3.1 大模型生图修补 | `steps/repair.py repair_image_gen` | 元素拼图集 + 二值 mask 送生图模型，prompt 逐字复刻；`--repair auto` 跟随 LLM 的 `element_repair_mode` |
| 2.3.2 传统算法修补 | `surface_fill` / `_ns_fill_region` | 用同元素可见像素的中位数/双簇主簇填补 + 接缝混合 |
| 2.3.3 背景修补 | `repair_background` | `background_repair.mode` 三分支；scene 打黑洞（+极窄柔化边）送 Flux；prompt 逐字复刻 |
| 2.4 组合 | `steps/export.py` + `text.render` | 底板 + 图层按 z_order 回拼 + 文字按 2.1.3 样式重渲染 |
| （原文耗时图） | `Pipeline` 计时 | 每步耗时写入 `audit.json → timings` |

## 用法

```powershell
$env:PYTHONIOENCODING='utf-8'

# 一键跑（有缓存 plan 就用缓存，没有则自动调 Qwen 生成）
& 'E:\CodeRep\Python\python311\python.exe' '_tools\run_pipeline.py' --src 'output\基础测试\test.png' --dest 'output\基础测试_fix3'

# 强制传统修补（快） / 强制生成式
... --repair ns --bg ns
... --repair gen --bg gen

# 单独生成/查看 LLM 规划（会先按 2.1.2 生成去字图再送 LLM）
& '...\python.exe' '_tools\gen_llm_plan.py' --src '...' --draw

# 产物校验
& '...\python.exe' '_tools\verify_output.py' --dest 'output\基础测试_fix3'
```

主要开关：`--repair {auto,ns,gen}`、`--bg {auto,ns,gen}`、`--refresh`（忽略缓存）、
`--bg-estimate {median,kmeans}`（原文为中位数，kmeans 是可选扩展）、
`--no-fg-core-filter`、`--no-atlas-prefill`（回到原文字面的纯黑洞输入）、
`--cc-keep-ratio`、`--safe-area N`、`--dropped-ids a,b`、`--model`。

密钥：环境变量 `DASHSCOPE_API_KEY`，或 `data/secrets/dashscope.key`（代码中不再硬编码）。

## 按图配置 `data/cases/<源图名>.json`

单图特有的东西不再写死在算法里：

```json
{ "dropped_ids": ["watermark_art"], "safe_area_height": 120,
  "v4_reference": true, "truth_bbox": {...}, "bg_prompt": "..." }
```

`bg_prompt` 是实测覆盖：通用背景提示词在 flux1-dev-fp8 上对覆盖率 >50% 的洞会填失败
（孔洞纯黑 >2% 被门限拒收），明信片收集这张图的背景需要显式描述材质才能通过自检。

## 质量闸与自检

* **分解无损性**：回拼图 vs 修补后 work，孔洞 + 7px 接缝带之外必须 0 差异（`audit.lossless_check`）。
* **生成式修补自检**：孔洞内纯黑占比 ≤2% 判通过；否则若孔洞有真实纹理（std≥6）**且**
  孔洞中位色与同元素可见材质中位色距离 ≤90 也通过（暗色材质合法）；否则换 seed 11/12/13，
  全失败逐元素回退传统算法。背景只用严格的 2% 门限。
* **fidelity 表由实际执行生成**：每个步骤调用 `ctx.note(...)` 记录自己真正做了什么，
  `audit.json → fidelity` 不再可能谎报未接线的功能。

## 实测结论 / 已知限制（本地 flux1-dev-fp8 + ComfyUI）

* 孔洞占比小（<10%）的元素，图集生成式修补质量好（见 `data/inpaint_cache_v6/atlas_*.png`）。
* 单元素孔洞占比 >50%（如几乎被子卡片盖满的面板）时，Flux 无论是否预填都会填出无关暗色内容，
  门限会拒收并回退传统算法 —— 这与旧版脚本"实测走传统 NS"的结论一致。
* 同理，通用背景提示词 + 60%+ 大洞的背景生成也会失败；需要按图 `bg_prompt` 或更好的修补模型。
* `raster_text_ids` 交叉校验：没有 OCR id 列表的旧 plan 里 LLM 会凭空编 id（例如把
  "金币：100" 标成美术字），因此 id 只有在「plan 携带本次 OCR id 列表」或「存在对应 logo query」时才生效，
  否则记入 `audit → fidelity` 的 rejected 列表。

## 仓库扩展（超出原文，均可关闭/可审计）

数字 OCR 复审、bbox guard、surface refine、重复精灵模板匹配（原文"后续改进"里的相似度召回）、
图集孔洞中位数预填（`--no-atlas-prefill` 关闭）、CJK 全角标点字体判定、状态条安全区图层、
contact sheet、`verify_output.py` 质量闸、per-case 配置。

## 目录

```
_tools/
├── run_pipeline.py        主管线（Step 顺序 = 原文顺序）
├── gen_llm_plan.py        独立 2.2.1（自动先去字再送 LLM）
├── draw_bbox.py           bbox 诊断图
├── verify_output.py       产物校验
└── pipeline/
    ├── config.py          原文全部数值常量 + 路径 + 按图配置
    ├── prompts.py         四段 prompt，逐字复刻原文
    ├── context.py         RunConfig / PipelineContext
    ├── llm_planner.py     OCR id 列表、plan 解析、多实例展开、父子链
    └── steps/             text / plan / segmentation / layer_build / repair / export / audit
data/
├── plans/                 LLM 规划缓存
├── cases/                 按图配置
├── secrets/               API key（不进代码）
└── inpaint_cache_v6/      图集/背景生成结果缓存（hash 命名）
```