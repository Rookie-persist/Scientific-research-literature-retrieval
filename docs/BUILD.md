# 子集化数据库构建技术文档 v2

> 配套文档：`PROJECT-PLAN.md`（项目计划，最终版）。
> 本文只负责「把数据库建出来」，不包含 M1/M2/M3 的代码改造；两者的衔接点见 §8。
> 命令默认在仓库根目录执行。

---

## 0. v2 修订说明

对照 `PROJECT-PLAN.md` 做了一致性审计，修正 9 处问题，其中 4 处会直接导致执行失败或结果失真：

| # | 问题 | 后果 | 修正 |
|---|---|---|---|
| 1 | 配置与计划冲突：文档写 bge-m3 / OCR 开 / docling_parse，计划要求 bge-small / OCR 关 / pypdfium2 | 下载量与耗时都对不上 | 拆成 `haiku.rag.yaml`（开发档）与 `haiku.rag.final.yaml`（最终档） |
| 2 | 语料规模不一致：文档用 150 篇，计划用 20 / 60 / 100 | 步骤与预算错位 | 统一为 `subset_<N>`，N ∈ {20, 60, 100}；旧的 150 篇目录已重命名为 `_legacy_subset_150` |
| 3 | 评测命令缺 `--no-telemetry` | **harness 直接拒绝启动**（`benchmark.py:138`） | 命令中补上 |
| 4 | 评测裁判默认 `ollama:qwen3.8`，本机没有 | 评测跑到一半报模型不存在 | 配置里显式设 `evaluations.judge` |
| 5 | 未使用 `download-models` | 模型在首次解析时才下载，失败点被推迟到摄取阶段 | 新增 §4.5 预下载步骤 |
| 6 | 时间估算混用两档口径 | 排期失真 | 按开发档 / 最终档分列 |
| 7 | `visualize` 会打开图片查看器 | 脚本化调用会阻塞 | 标注为交互命令 |
| 8 | A+B 补强未与建库文档衔接 | 读者以为建完库就完事 | 新增 §8 |
| 9 | 三档共用 `pdfs/` 目录 | **后一档把前一档的 PDF 一起索引，文档数与题目数错位** | 改为 `pdfs/smoke` / `pdfs/dev` / `pdfs/eval` 分目录 |

另外记录一个易错点：`--config` 是**全局参数**，必须写在子命令之前：

```bash
haiku-rag --config haiku.rag.final.yaml add-src ./pdfs/eval --db ./dbs/eval.lancedb
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ 在子命令前
```

---

## 1. 目标与档位

| 档位 | 论文数 | 可评分题（实测） | 约合 chunk | 数据库 | 配置 |
|---|---|---|---|---|---|
| **L0 smoke** | 20 | 199 | ~2,420 | `dbs/smoke.lancedb` | `haiku.rag.yaml` |
| **L1 dev** | 60 | 545 | ~7,260 | `dbs/dev.lancedb` | `haiku.rag.yaml` |
| **L2 eval** | 100 | 850 | ~12,100 | `dbs/eval.lancedb` | `haiku.rag.final.yaml` |

两档配置的差别只有解析质量（`pdf_backend` / `table_mode` / `images_scale`），**嵌入模型严格一致**——否则两个库无法互相打开，等于两套独立语料。

---

## 2. 环境基线（实测）

| 项 | 实测值 | 影响 |
|---|---|---|
| Python | 3.13.14 | 满足 `>=3.12` |
| CPU / GPU | 24 核 / RTX 5060 8 GB | 走 CPU 即可；GPU 非必需 |
| 磁盘 | D: 剩余 262 GB | 充裕 |
| `huggingface.co` | **不可达**（超时） | 必须设 `HF_ENDPOINT` |
| `hf-mirror.com` | 可用 | 元数据、Docling 模型、嵌入模型都从这里取 |
| `arxiv.org` | 可用但慢（8–21 秒/篇） | 下载需限速与重试 |
| Ollama | 运行中 0.32.3，仅 `qwen3.5:9b-q4_K_M` | 作问答模型；**不能作嵌入模型** |
| `uv` | 未安装 | 仅评测 harness 需要 |

**第一条必须做的事**：`export HF_ENDPOINT=https://hf-mirror.com`，且要在**下载元数据**与**首次运行 Docling** 之前都生效。

---

## 3. 数据源事实（实测，非估算）

| 指标 | 数值 | 来源 |
|---|---|---|
| ORB 语料论文数 | **1,000 篇** | `pdf_urls.json` |
| 查询总数 | 3,045 条 | `queries.json` |
| 剔除图像类查询后可评分 | **2,062 条 / 389 篇** | 子集脚本统计 |
| 每题金标文档 | 1 篇 | `qrels.json` |
| 官方库 chunk 总量 | 121,168 | 对应 1,000 篇 |
| 折算每篇 chunk | **约 121 个** | 121,168 ÷ 1,000 |

按 121 chunk/篇估算：20 篇 ≈ 2,420，60 篇 ≈ 7,260，100 篇 ≈ 12,100。全部低于 10 万 chunk 的建索引阈值，检索走暴力 kNN，与官方基准条件一致。

---

## 4. 构建步骤

### 4.1 环境与依赖（一次性）

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install uv -i https://mirrors.aliyun.com/pypi/simple
.venv/Scripts/uv.exe pip install --python .venv/Scripts/python.exe \
  --index-url https://mirrors.aliyun.com/pypi/simple --index-strategy unsafe-best-match \
  "haiku.rag==0.98.0" "sentence-transformers>=3.0,<6.0"
```

**必须走镜像 + uv**（D0 实测）：PyPI 直连只有约 30 kB/s，pip 单线程下载 103 MB 的 lancedb 就要 50 分钟以上，装完全部依赖需数小时。阿里云镜像单连接约 180–226 kB/s，`uv` 再并行化，实测 **4 分 53 秒**装完。清华镜像在本机不可用（索引返回空）。

**不要装 CUDA 版 torch**。PyPI 的 Windows 轮子 124 MB 已够用；CUDA 轮子要 2 GB 以上，而本项目的瓶颈在设计不在算力。

版本对齐：仓库副本是 `0.98.0`，装同版本可避免数据库 schema 迁移。

### 4.2 元数据

```bash
python scripts/fetch_orb_metadata.py --out meta
```

只用标准库，不需要先建虚拟环境。依次尝试 `$HF_ENDPOINT` → `hf-mirror.com` → `huggingface.co`，并校验首字节（防止把 HTML 错误页写成 JSON）。

**验收**：四个文件记录数为 1000 / 3045 / 3045 / 3045。

### 4.3 生成子集

```bash
# 三档按需生成，互不覆盖
python scripts/build_orb_subset.py --metadata-dir meta --target-papers 20  --out subset_20  --drop-image-queries
python scripts/build_orb_subset.py --metadata-dir meta --target-papers 60  --out subset_60  --drop-image-queries
python scripts/build_orb_subset.py --metadata-dir meta --target-papers 100 --out subset_100 --drop-image-queries
```

脚本做三件事：只保留金标文档真实存在的查询 → 按「覆盖题目最多」贪心选论文 → **同步裁剪四个文件**。

> 只删论文不删题目，那些题会变成必然失败，指标被系统性压低，而且不会报错。这是子集化最常见的坑。

**验收**：`stats.json` 中 `queries_without_answer` 必须为 0；题数应接近 199 / 545 / 850。

### 4.4 下载 PDF

```bash
python scripts/download_pdfs.py --subset subset_20 --out pdfs/smoke --workers 3 --retries 4
```

并发默认 3（arXiv 限流），403/429 走长退避，到达即校验 `%PDF` 文件头，已存在的合法文件直接跳过（断点续传）。

**实测速率**：2 篇 / 31 秒（2 并发）≈ 15 秒/篇。20 篇约 5 分钟，100 篇约 20–30 分钟。

**验收**：`ls pdfs/smoke/*.pdf | wc -l` 接近目标数；`manifest.jsonl` 里失败篇目要记录，不能悄悄忽略。

### 4.5 预下载模型（v2 新增，建议不要跳过）

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1

# 1) 嵌入模型与分词器：必须用 local_dir 下载（见下方说明）
.venv/Scripts/python.exe scripts/fetch_models.py

# 2) Docling 模型与其余依赖
.venv/Scripts/haiku-rag.exe download-models

# 3) 修复 Docling 模型在 HF 缓存中的快照视图
.venv/Scripts/python.exe scripts/fix_hf_snapshots.py
```

**D0 实测：本机 HF 缓存的快照目录为空。** `snapshot_download()` 报告成功、blob 落盘，但 `snapshots/<sha>/` 是空的（无法创建符号链接），于是：

- 分词器报 `Couldn't instantiate the backend tokenizer`
- Docling 报 `Image processor config not found: ...snapshots/<sha>/preprocessor_config.json`

`fetch_models.py` 用 `local_dir` 把模型复制到 `models/bge-small-en-v1.5`（配置已指向该路径）；`fix_hf_snapshots.py` 把 Docling 已经写好的真实文件复制进快照目录，避免重下 1.3 GB。两者都是幂等的，可重复执行。

这一步会按当前配置一次性拉齐：Docling 模型、分块分词器、sentence-transformers 嵌入模型、以及配置里启用的 reranker / Ollama 模型。

**为什么单独做**：不做的话，模型下载会被推迟到第一次摄取时，一旦镜像不通，你会在「解析卡住」的表象下排查很久。提前跑一次，5–15 分钟就能暴露网络问题。

**验收**：命令正常结束；`~/.cache/huggingface` 出现 docling 与 bge-small 的目录。

### 4.6 建库与摄取

```bash
# L0：开发档（自动读取 ./haiku.rag.yaml）
.venv/Scripts/haiku-rag.exe init --db ./dbs/smoke.lancedb
.venv/Scripts/haiku-rag.exe add-src ./pdfs/smoke --db ./dbs/smoke.lancedb 2>&1 | tee logs/ingest-smoke.log

# L1：同一个库、换 60 篇语料；也可另建 dev 库
.venv/Scripts/haiku-rag.exe init --db ./dbs/dev.lancedb
.venv/Scripts/haiku-rag.exe add-src ./pdfs/dev --db ./dbs/dev.lancedb 2>&1 | tee logs/ingest-dev.log

# L2：最终档（--config 必须写在子命令前）
.venv/Scripts/haiku-rag.exe --config haiku.rag.final.yaml init --db ./dbs/eval.lancedb
.venv/Scripts/haiku-rag.exe --config haiku.rag.final.yaml add-src ./pdfs/eval --db ./dbs/eval.lancedb 2>&1 | tee logs/ingest-eval.log
```

> 注意：每档用独立的 PDF 目录（`pdfs/smoke` / `pdfs/dev` / `pdfs/eval`）。共用一个目录会让后一档把前一档的 PDF 一起索引进去，文档数与题目数就对不上了。

三条硬约束：

1. **每个数据库只允许一个写进程**（项目自身实现约束，不是 LanceDB 限制）。不要并行跑两个 `add-src`。
2. **中断可续**：已入库文档会被跳过，重跑同一条命令即可。
3. **写进程必须与建库时用同一份配置**，否则嵌入器身份不一致会报错。

**时间估算（按档位区分）**：

| 档位 | 解析配置 | 每篇 | 总量 |
|---|---|---|---|
| 开发档 | pypdfium2 + fast + 无 OCR | **45 秒（实测）** | 20 篇 **15 分钟**；60 篇约 45 分钟 |
| 最终档 | docling_parse + accurate + 无 OCR | 60–90 秒（外推） | 100 篇约 100–150 分钟 |

> D0 实测 20 篇用了 15 分 04 秒，是原估算（8–15 秒/篇）的 3–5 倍。慢在 TableFormer 表格结构推理，与 OCR 无关。排期请按上表，不要按理论值。

### 4.7 验证

```bash
# 文档数与 chunk 数
.venv/Scripts/haiku-rag.exe info --db ./dbs/smoke.lancedb

# 列表（应接近目标篇数）
.venv/Scripts/haiku-rag.exe list --db ./dbs/smoke.lancedb | head -20

# 检索冒烟：结果应带 page_numbers 与 headings
.venv/Scripts/haiku-rag.exe search "what is the main contribution of this paper" \
  --db ./dbs/smoke.lancedb --limit 5

# 健康检查：索引覆盖、嵌入一致性、待迁移项
.venv/Scripts/haiku-rag.exe doctor --db ./dbs/smoke.lancedb
```

**判定标准**：chunk 数与档位估算同量级（±30% 正常，取决于图片与表格数量）；`search` 结果带页码与章节；`doctor` 无嵌入不匹配告警。

**引用高亮（交互命令）**：`haiku-rag visualize <chunk_id> --db ./dbs/smoke.lancedb` 会打开图片查看器，在原文页上画出命中框。它不适合脚本化调用，会阻塞等待窗口关闭。

### 4.8 与评测 harness 对接

harness 会加载**全量** 3,045 条查询，所以必须用 `--filter-ids` 限定到子集题目，否则子集外的题全部计为失败。

**先补裁判配置**（否则会去调用本机不存在的 `qwen3.8`）。在配置里加：

```yaml
evaluations:
  judge:
    provider: ollama
    name: qwen3.5:9b-q4_K_M
    thinking: false
    temperature: 0
    vision: false
```

> 自评偏差：让答题模型同时当裁判，会高估准确率。缓解方式是用 `--skip-qa` 只跑检索指标，或对 QA 结果做人工抽检校准。报告中应写明裁判是谁。

```bash
pip install uv
uv sync --all-packages        # 在仓库根目录执行

uv run evaluations run orb_text \
  --skip-db \
  --skip-retrieval \
  --no-telemetry \
  --db ./dbs/eval.lancedb \
  --config ./haiku.rag.final.yaml \
  --filter-ids ./subset_100/query_ids.txt \
  --limit 40
```

四个参数的理由：

- `--skip-db`：不重新建库，直接用我们自己的库。
- `--skip-retrieval`：检索基准不接受 `--filter-ids`，会跑全量查询；检索指标另算或写自定义 spec。
- `--no-telemetry`：**必需**，否则 harness 找不到 `LOGFIRE_TOKEN` 会直接拒绝启动。
- `--limit 40`：QA 逐题串行，850 题要跑数小时；40 题足以做前后对照。

**替代方案**：不用 harness，自己读 `subset_100/queries.json` + `qrels.json` + `answers.json` 算 recall@k / MAP / 引用命中率。代价是失去 `cited_map` 与 McNemar 配对检验，而这两项正是「与官方同口径」的说服力来源。

---

## 5. 预算

| 项目 | 体积 | 依据 |
|---|---|---|
| ORB 元数据 | 1.6 MB | 实测 |
| PDF（20 / 60 / 100 篇） | 13 / 38 / 64 MB | 实测 0.64 MB/篇 |
| torch（CPU） | 124 MB | 实测 PyPI |
| torchvision + scipy + numpy + onnxruntime + opencv + rapidocr | 200–300 MB | 估算（Docling 传递依赖） |
| lancedb | 83 MB | 实测 PyPI |
| docling 核心（parse / core / transformers） | 约 40 MB | 实测 |
| 其他 Python 依赖 | 约 60 MB | 估算 |
| Docling 模型 | **1.3 GB** | **实测**（CodeFormula 640 + TableFormer 358 + 两个 layout 各 170 + 分类器 34） |
| 嵌入 bge-small-en-v1.5 | **383 MB** | **实测**（含 onnx 与 pytorch_model.bin 重复格式；只取 safetensors 可降到约 140 MB） |
| 依赖安装后体积 | **2.0 GB** | **实测**（`.venv`，wheel 解压后远大于下载体积） |
| **下载合计** | **约 2.4 GB** | 实测 |
| **磁盘合计** | **约 3.9 GB** | 含 `.venv` 2.0 + 模型 0.38 + Docling 1.3 + PDF 0.05 + 库 0.13 |

数据库体积：开发档（1.0 图像）约 0.8–1.2 GB / 100 篇；最终档（1.5 图像）约 1.8–2.6 GB / 100 篇。

---

## 6. 故障对照表

| 现象 | 原因 | 处理 |
|---|---|---|
| 元数据下载超时 / `000` | `huggingface.co` 不可达 | 设 `HF_ENDPOINT=https://hf-mirror.com` |
| 摄取卡在模型加载 | Docling 模型下载被墙 | 先跑 §4.5 `download-models` |
| arXiv 返回 403 / 429 | 并发过高 | `--workers 2`，`--retries 5`，隔几分钟重跑 |
| `ConfigMismatchError` | 嵌入器与建库时不一致 | 用建库时那份配置；或 `rebuild --embed-only` |
| `ConverterWedgedError` | 大 PDF 转换超时卡住共享转换器 | 重启进程；调小 `split_pages` |
| 摄取时 OOM | vacuum 峰值 = max(目标片段, 最大片段) | 调小 `compaction_target_bytes`；分批摄入 |
| 两个进程写同一库报错 | 单写者约束 | 只保留一个写进程 |
| 检索结果没有页码 / 无法高亮 | 建库时关了 `generate_page_images` | 需重建 |
| 问答返回空答案，检索正常 | 模型思考模式吃光 token（`done_reason=length`） | `qa.model.thinking: false` |
| 问答报模型不存在 | 未设 `qa.model.name`，回落 `qwen3.8` | 显式写 `name` |
| 评测启动即退出 | 缺 Logfire token | 加 `--no-telemetry` |
| 评测报 judge 模型不存在 | 默认 `ollama:qwen3.8` | 配置 `evaluations.judge` |
| 评测指标接近 0 | 未用 `--filter-ids`，混入子集外题目 | 加 `--filter-ids` |
| `visualize` 卡住不返回 | 打开了图片查看器 | 交互使用，别放进脚本 |
| 分词器报 `Couldn't instantiate the backend tokenizer` | HF 缓存快照为空（符号链接失败） | 跑 `scripts/fetch_models.py`，配置指向本地目录 |
| Docling 报 `Image processor config not found` | 同上，Docling 模型快照为空 | 跑 `scripts/fix_hf_snapshots.py` |
| 摄取失败重试后数据库膨胀 3 倍 | LanceDB 保留历史版本（默认 1 天） | `haiku-rag vacuum --db ... --retention-seconds 0`（实测 411 MB → 125 MB） |

---

## 7. 决策记录

**为什么两档配置只差解析，不差嵌入。**
嵌入器一换，数据库就不能互相打开（`ConfigMismatchError`），等于两套语料，无法做「同一批题目、同一批文档」的对照。把差异限制在解析参数上，L2 重建只是为了提高解析质量，而不是换表示空间。

**为什么用 bge-small 而不是 bge-m3。**
省 2.1 GB 下载与一半嵌入耗时。代价是失去中英跨语言检索——若最终评测发现中文查询召回明显偏低，再升级到 bge-m3 并重建一次（这是唯一需要重建嵌入的场景）。

**为什么开发档关 OCR。**
arXiv PDF 自带文本层，OCR 纯属浪费。只有解析为空的文档才需要 `force_ocr` 单独重跑。

**为什么保留页面图像。**
`visualize_chunk` 把命中块画回原页，是本项目最核心的差异化能力，也是 M3 引用溯源的基础。关掉它等于放弃演示点。开发档用 1.0、最终档用 1.5 折中体积与清晰度。

**为什么不建向量索引。**
12,100 chunk 远低于 10 万阈值，暴力 kNN 精确且够快；官方基准本身也是无索引条件。

---

## 8. 与改造模块的衔接

本文档到「库建好、能检索」为止。后续三项改造不在此文档内，但与建库有明确接口：

| 模块 | 接口点 | 是否需要重建库 |
|---|---|---|
| M1 查询理解 | `Hook.before_search`（改 query / limit） | 否 |
| M2 结果治理 | `Hook.after_search`（改 score / annotations） | 否 |
| M3 引用校验 | 服务层后处理 `(answer, citations)` | 否 |
| A 引用策略 | `Agent(capabilities=[rag, citation_policy()])` | 否 |
| B 验证回灌 | M3 判定 → 改写查询 → 再检索（上限 2 轮） | 否 |

**这一列全部是「否」，是本项目最重要的成本控制点**：建一次库，之后所有迭代都在它上面做。只有换嵌入模型或改分块/解析参数才需要重建。

---

## 9. 复现清单

- [ ] `HF_ENDPOINT` 已在当前 shell 设置
- [ ] 元数据四文件记录数为 1000 / 3045 / 3045 / 3045
- [ ] `subset_<N>/stats.json` 中 `queries_without_answer` 为 0
- [ ] `pdfs/<档位>/` 合法 PDF 数 ≥ 目标数 × 0.95，失败篇目已记录
- [ ] `scripts/fetch_models.py` 与 `scripts/fix_hf_snapshots.py` 均正常结束
- [ ] `haiku-rag download-models` 正常结束
- [ ] 配置中 `vector_dim: 384` 与 bge-small 一致
- [ ] 摄取全程只有一个写进程
- [ ] `info` 的 chunk 数与档位估算同量级
- [ ] `search` 返回带 `page_numbers` 与 `headings`
- [ ] `doctor` 无嵌入不匹配告警（Ollama 未启动时会有 1 条 provider 失败，属预期）
- [ ] 摄取失败重试后执行过 `vacuum --retention-seconds 0`
- [ ] `python scripts/check_ollama.py --model qwen3.5:9b-q4_K_M` 返回 ready
- [ ] 评测命令带 `--no-telemetry` 与 `--filter-ids`

---

## 附：命令速查

```bash
# 0. 环境
python -m venv .venv
.venv/Scripts/python.exe -m pip install "haiku.rag==0.98.0" "sentence-transformers>=3.0,<6.0"
export HF_ENDPOINT=https://hf-mirror.com

# 1. 元数据
python scripts/fetch_orb_metadata.py --out meta

# 2. 子集（N = 20 / 60 / 100）
python scripts/build_orb_subset.py --metadata-dir meta --target-papers N \
  --out subset_N --drop-image-queries

# 3. PDF（断点续传）
python scripts/download_pdfs.py --subset subset_N --out pdfs/<档位> --workers 3

# 4. 预下载模型
.venv/Scripts/haiku-rag.exe download-models

# 5. 建库（开发档）
.venv/Scripts/haiku-rag.exe init --db ./dbs/dev.lancedb
.venv/Scripts/haiku-rag.exe add-src ./pdfs/dev --db ./dbs/dev.lancedb

# 6. 建库（最终档，注意 --config 在子命令前）
.venv/Scripts/haiku-rag.exe --config haiku.rag.final.yaml init --db ./dbs/eval.lancedb
.venv/Scripts/haiku-rag.exe --config haiku.rag.final.yaml add-src ./pdfs/eval --db ./dbs/eval.lancedb

# 7. 验证
.venv/Scripts/haiku-rag.exe info --db ./dbs/dev.lancedb
.venv/Scripts/haiku-rag.exe search "query" --db ./dbs/dev.lancedb
.venv/Scripts/haiku-rag.exe doctor --db ./dbs/dev.lancedb

# 8. Ollama 预检
python scripts/check_ollama.py --model qwen3.5:9b-q4_K_M
```
