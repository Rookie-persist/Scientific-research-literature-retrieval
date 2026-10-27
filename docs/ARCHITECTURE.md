# 架构与实现说明：科研文献检索基线（haiku.rag 二次开发）

> 本文档面向接手者与协作者，说明设计取舍、已完成的实验、实测结论与已知陷阱。
> 评测数字的原始记录见 `eval/`，各阶段细节见 `docs/` 下其余文档。

---

## 0. 一页速览

**一句话**：在开源 RAG 库 haiku.rag 上做二次开发，面向科研文献（arXiv）建可复现的检索/问答评测基线，
并验证「改查询」「改结果治理」「改解析」三类改造里，哪一类真的有效。

**当前状态**：D0（环境+建库）、D1（基线）、D2（M1 查询理解 Hook）完成。M2 未做，M3 未开始。

**最重要的一条结论**：**瓶颈不在检索，在解析。**
- 检索 recall@5 = 99.5%、MRR = 0.9912（20 篇 / 199 题）
- 但答案正确率只有 87.5%（14/16），两道失败题都不是检索问题
- 13.3%（425/3,193）的文本块在解析阶段就丢了公式，变成 `<!-- formula-not-decoded -->`

**第二重要**：`SearchResult.score` 是 RRF 融合分（1/61 的整数倍），**不含语义相似度**，
正负样本分数完全重叠，因此**不能用分数阈值做拒答判定**。

---

## 1. 背景与目标

科研文献检索的真实痛点不是「检不到」，而是「检到了但依据不可核验」：
向量检索能召回段落，却说不清答案对应原文哪一页、哪一句。

原计划是三个模块（M1 查询理解 / M2 结果治理 / M3 引用校验），都通过 haiku.rag 官方 Hook
扩展点实现，不改检索内核。**D1/D2 的实测把重心改了**：

- M1 的假设（缩写查询检不到全称论文）**被否掉**
- M2 的收益上限（Top-5 参考文献块只占 7.8%）**不足以支撑一个模块**
- 真正的短板是解析保真度（公式 + 表格）

---

## 2. 技术底座（haiku.rag 的关键机制）

读源码确认过，以下事实影响所有设计决策：

| 机制 | 事实 | 影响 |
|---|---|---|
| 解析 | Docling（`docling-local`），`ConversionOptions` **没有** `do_formula_enrichment` 字段 | 公式识别要改代码才能开 |
| 分块 | `chunker_type: hybrid`（结构感知），`chunk_size: 256` | 表格/公式按 Docling 节点切 |
| 检索 | LanceDB 混合（向量 + FTS），`RRFReranker()`，`k=60`（`store/repositories/chunk.py:316`） | score 是 `Σ 1/(60+rank)`，只有几个离散值 |
| 扩展点 | `haiku.rag.hooks` 入口点组；`config.hooks` 列表；`SearchRequest{query, filter, search_type, limit}` | 可装插件，不用 fork |
| after_search | 拿到的是**已被 limit 截断**的结果 | 降权类改造必须先在 before_search 放大 limit |
| annotations | `SearchResult.annotations` 存在，但 `format_for_agent()` **不渲染** | 无法通过 annotations 给模型传信号 |
| 问答 | `ask()` = `Agent(model, capabilities=[RAGCapability])`，工具 `search / execute_code / cite` | 模型自决检索轮数（`max_searches=5`） |
| 引用策略 | `CitationPolicyCapability` 默认**未注册** | 不注册则模型可以零引用作答 |
| 评测裁判 | 默认 `evaluations.judge = ollama:qwen3.8`（本机没有） | 需显式配置或用人工判分 |

---

## 3. 数据：ORB 子集化

**上游**：OpenRAG Bench，HuggingFace `vectara/open_ragbench`，子目录 `pdf/arxiv/`。
经 `hf-mirror.com` 下载（`huggingface.co` 直连超时）。

**真实规模**（实测，纠正了早期估算）：

- 语料 **1,000 篇** arXiv 论文（不是 3,045 篇）
- 查询 **3,045 条**，每条 1 篇金标文档
- 剔除图像类查询后，可评分上限 **389 篇 / 2,062 题**
- 每篇约 **121 chunk**（官方库 121,168 / 1,000）

**子集化原则**：减论文必须同步减题目。
只删论文不删题目，会制造「必然失败」的题，系统性压低指标。
`scripts/build_orb_subset.py` 按「覆盖题目最多」贪心选论文，并同步裁剪 `queries/qrels/answers`。

**当前档位**：

| 档 | 论文 | chunk | 可评分题 | 用途 |
|---|---|---|---|---|
| L0 | 20 | 3,193 | 199 | 已建（`dbs/smoke.lancedb`，125 MB） |
| L1 | 60 | ~7,300 | 545 | 未建 |
| L2 | 100 | ~12,100 | 850 | 未建 |
| 历史 | 150 | — | 1,185 | 已弃用，目录 `_legacy_subset_150` |

> 子集化后**不能与官方分数横向比**（候选池不同），只能做内部前后对照。

---

## 4. 环境与配置

**硬约束（都是实测踩出来的）**：

1. **PyPI 直连仅 ~30 kB/s**，pip 装 lancedb（103 MB）要 50 分钟。
   必须用阿里云镜像 + `uv pip install`（并行），实测 4 分 53 秒装完。清华镜像不可用。
2. **HF 缓存快照目录为空**（建不了符号链接）。表现为分词器报
   `Couldn't instantiate the backend tokenizer`、Docling 报 `Image processor config not found`。
   修法：`scripts/fetch_models.py`（`local_dir` 下载）+ `scripts/fix_hf_snapshots.py`（补快照）。
   环境变量：`HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_SYMLINKS=1`
3. **Ollama 两个必设项**（否则表现为「答案跑题/空答案」，极难排查）：
   - `num_ctx >= 16384`：默认 4096 装不下 ~10 KB 系统提示，模型会丢失问题
   - `presence_penalty = 0`：模型自带 1.5，会惩罚问题里的词导致话题漂移
   - 已固化为 `qwen35-9b-16k`（从 `qwen3.5:9b-q4_K_M` 派生）
4. **思考模式不可用**：3072 token 全被推理吃掉，`response` 为空。配置里必须 `thinking: false`。

**配置档位**：

| 文件 | 用途 | 关键项 |
|---|---|---|
| `haiku.rag.yaml` | 开发档（默认加载） | bge-small/384、`pypdfium2`、`do_ocr: false`、`table_mode: fast`、`max_tokens: 4096` |
| `haiku.rag.final.yaml` | 最终档（`--config` 指定） | `table_mode: accurate`，嵌入器与开发档**严格一致** |
| `haiku.rag.m1*.yaml` | M1 实验档 | 见 docs/M1-QUERY-HOOK.md |

> 嵌入模型**建库后不可换**（维度不可变，换维度报 `ConfigMismatchError`）。
> `--config` 是**全局参数**，必须写在子命令之前。

**体积实测**：Docling 模型 1.3 GB（含 CodeFormula 640 MB、TableFormer 358 MB）、
bge-small 383 MB、`.venv` 2.0 GB；下载合计约 2.4 GB，磁盘约 3.9 GB。

**摄取速度**：开发档 **45 秒/篇**（20 篇 15 分 04 秒），比原估算慢 3–5 倍，慢在 TableFormer。

---

## 5. 已完成工作

### D0：环境与建库（`docs/D0-SETUP.md`）

产出 `dbs/smoke.lancedb`（20 篇 / 3,193 chunk / 125 MB），检索带页码与章节，`doctor` 18 ok。
端到端 61 分钟。

**注意**：摄取失败重试后 DB 会膨胀 3.3 倍（LanceDB 保留历史版本），
必须 `vacuum --retention-seconds 0` 回收（411 MB → 125 MB）。
普通 `vacuum` 回收不了当天版本，实测反而涨到 535 MB。

### D1：基线（`docs/BASELINE.md`）

20 题 = 16 道 ORB 金标（extractive 8 / abstractive 8）+ 4 道负样本。

| 指标 | 值 |
|---|---|
| 答案正确率 | 14/16 = **87.5%** |
| 负样本拒答 | 4/4，且零引用 |
| 引用率 / 带页码率 | 93.8% / 93.8% |
| 延迟（正题） | 均值 20.9 s，p50 19.6 s |
| 检索 recall@5 / MRR | 99.5% / 0.9912 |
| Top-5 中参考文献块 | 0.39 / 5 = **7.8%** |
| Top-5 中公式残缺块 | 0.92 / 5 = **18.4%** |

**两道失败题的归因**（关键）：

- Q4「WTT 公式」：检索完全正确（Top-1、正确章节、正确页码），但块里是
  `<!-- formula-not-decoded -->`，模型编了 `0.32/√D`（金标 `(0.32)²/2D`）
- Q15「Tesla 交易建议」：金标在表格里，被压平成
  `Cumulative Return TSLA -0.1633 -0.1829 ...`，列头丢失，模型称「文档未给出建议」

### D2：M1 查询理解 Hook（`docs/M1-QUERY-HOOK.md`）

产出可安装插件 `m1_hook/`（入口点 `haiku.rag.hooks: m1-query`），
两个能力：双向缩写扩展 + 语料覆盖门控（`off / cap / starve / conservative` 四策略）。

**实验一（缩写扩展，12 对探针）**：基线两种问法 recall@5 **都是 100%**，MRR 0.9583；
开启后 MRR → 1.0。**收益可忽略，假设被否掉**。
（ORB 199 条查询里只有 18 条含词典缩写，用整个基准测不出信号，故自建探针。）

**实验二（覆盖门控，10 条负样本）**：

| 策略 | 均值 | 中位 | 拒答 |
|---|---|---|---|
| off | 49.0 s | 53.9 s | 10/10 |
| starve | **23.4 s** | **17.3 s（−68%）** | **10/10** |

**失效模式（重要）**：`tesla` 是词表外专名（语料只写 TSLA），被判定为「语料缺失」
→ 零结果 → 可答题引用从 2 条掉到 0。
`conservative` 策略（单个大写专名不触发）能保住引用，但 neg2 延迟反涨到 **269.6 s**
——**饿死检索会诱发更多重试**（`max_searches=5`）。

**决定**：`coverage_policy` 默认 `off`。要用选 `conservative` 且把 `max_searches` 降到 3。

---

## 6. 实测结论清单（最重要，可直接引用）

| # | 结论 | 数值 | 证据文件 |
|---|---|---|---|
| 1 | 检索不是瓶颈 | recall@5 99.5%、MRR 0.9912、Top-1 带页码 100% | `eval/retrieval-diag.json` |
| 2 | 公式在解析阶段丢失 | 425/3,193 = 13.3%，波及 15/20 篇（最严重 79/146） | `docs/BASELINE.md` |
| 3 | 公式识别可修复 | 开启后输出 `WTT = \frac{(0.32)^2}{2D}`，与金标一致；代价每页 +15 s | 手动实验 |
| 4 | score 是 RRF 分，不能判拒答 | 0.0164=1/61、0.0328=2/61；正负样本分布完全重叠 | `docs/BASELINE.md` 发现 2 |
| 5 | 参考文献挤占远低于预期 | Top-5 中仅 7.8% | `eval/retrieval-diag.json` |
| 6 | 缩写扩展无检索收益 | 两种问法基线均 100%，MRR 0.958→1.0 | `eval/probe-acronym-*.json` |
| 7 | 覆盖门控降延迟 68% | 中位 53.9→17.3 s，拒答 10/10 | `eval/neg10-*.jsonl` |
| 8 | 门控会误杀专名 | `tesla` → 引用 2→0；`conservative` 保住但延迟不稳 | `eval/ab-m1-*.jsonl` |
| 9 | 拒答成本是正题的 2–4 倍 | 负样本均值 49.0 s vs 正题 20.9 s | `eval/neg10-off.jsonl` |
| 10 | 表格被压平导致该答却拒答 | 金标在表格，列头丢失 | `docs/BASELINE.md` |

---

## 7. 目录与文件职责

仓库根目录（本文档中所有相对路径均相对仓库根）

| 路径 | 职责 |
|---|---|
| `haiku.rag.yaml` / `.final.yaml` | 开发档 / 最终档配置 |
| `haiku.rag.m1{,.cap,.starve,.conservative}.yaml` | M1 四档实验配置 |
| `dbs/smoke.lancedb` | L0 数据库（20 篇 / 3,193 chunk） |
| `subset_20/` | 20 篇子集（queries/qrels/answers/paper_ids/query_ids） |
| `meta/` | ORB 原始元数据（1.6 MB） |
| `pdfs/smoke/` | 20 篇 PDF（50 MB） |
| `models/bge-small-en-v1.5/` | 本地嵌入模型（绕开 HF 快照问题） |
| `m1/acronyms.json` | 32 条领域缩写词典 |
| `m1/negatives10.json` | 10 条负样本 |
| `m1/trace-*.jsonl` | Hook 每次决策的 trace（排查「配置没生效」的唯一手段） |
| `m1_hook/` | 可安装插件（`pyproject.toml` + `src/m1_query/hook.py`） |
| `eval/` | 全部评测记录（baseline / retrieval-diag / probe / ab-m1 / neg10） |
| `scripts/` | 11 个脚本，见下表 |
| `docs/BUILD.md` | 建库技术文档（含 9 条故障对照） |
| `docs/D0-SETUP.md` / `docs/BASELINE.md` / `docs/M1-QUERY-HOOK.md` | 各阶段记录 |

**脚本清单**：

| 脚本 | 作用 |
|---|---|
| `fetch_orb_metadata.py` | 经镜像下载 ORB 元数据（仅标准库） |
| `build_orb_subset.py` | 生成子集，同步裁剪题目 |
| `download_pdfs.py` | 限速 + 重试 + 断点续传 |
| `fetch_models.py` | `local_dir` 模式下载嵌入模型 |
| `fix_hf_snapshots.py` | 修补 HF 快照（免重下 1.3 GB） |
| `check_ollama.py` | Ollama 预检（服务/模型/think=false/嵌入端点） |
| `diag_retrieval.py` | 纯检索诊断（recall@k、MRR、参考文献占比、公式残缺占比） |
| `run_baseline.py` | 问答基线（支持 `--qids` / `--negatives-file` / `--policy`） |
| `ask_policy.py` | 带 `CitationPolicyCapability` 的 ask 封装 |
| `build_acronym_dict.py` | 从语料抽缩写词典（含首字母子序列校验） |
| `probe_acronym.py` | 缩写/全称对照探针 |

---

## 8. 复现命令

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_SYMLINKS=1
export HAIKU_RAG_CONFIG_PATH=./haiku.rag.yaml

# 检索诊断（不调 LLM，约 1 分钟）
.venv/Scripts/python.exe scripts/diag_retrieval.py --db ./dbs/smoke.lancedb \
  --subset subset_20 --limit 5 --out eval/retrieval-diag.json

# 问答基线（20 题，约 10 分钟）
.venv/Scripts/python.exe -u scripts/run_baseline.py --db ./dbs/smoke.lancedb \
  --subset subset_20 --out eval/baseline.jsonl --policy

# M1 插件安装
.venv/Scripts/uv.exe pip install --python .venv/Scripts/python.exe --no-build-isolation -e ./m1_hook
```

---

## 9. 已知坑（照抄会失败）

1. **YAML 1.1 把裸写的 `off` 解析成布尔 `False`** → 配置里必须写 `"off"`，代码里也要归一化。
   表现为 trace 里 `"policy": false` 且门控静默失效，**不报错**。
2. **Hook 的 `lifespan` 必须 `@asynccontextmanager`**，否则
   `AttributeError: 'async_generator' object has no attribute '__aenter__'`。
3. **`limit=0` 会抛 `ValueError: k must be positive`**；要让检索返回空，用
   `filter="uri = '__m1_no_match__'"`。
4. **`Document.content` 是空的**，正文只在 chunk 里；`SearchResult` 字段是
   `document_uri`（不是 `doc_uri`）；`Chunk` 没有 `headings`（只有 `SearchResult` 有）。
5. **`annotations` 不会渲染给模型**（`format_for_agent` 不含它），别指望用它传信号。
6. **`--config` 必须写在子命令之前**；`add-src` 支持目录与 `--meta KEY=VALUE`。
7. **`sentence-transformers` provider 不接受** `query_prefix` / `document_prefix`，设置会抛异常。
8. **`visualize` 会弹图片查看器**，不能放进脚本。
9. 评测加 `--filter-ids` 只作用于 QA；检索基准不接受它。官方 harness 需 `--no-telemetry`
   （找不到 `LOGFIRE_TOKEN` 会拒绝启动）。

---

## 10. 未完成与下一步（按性价比排序）

| 顺序 | 事项 | 预期产出 | 成本 |
|---|---|---|---|
| 1 | **解析保真度**：`do_formula_enrichment=True` + `table_mode: accurate`，重摄取 | 正确率 87.5% → 93.75%/100% | 80 分钟摄取 + 需改代码暴露开关（约 20 行） |
| 2 | **M3 引用校验**：句子切分 → 三态判定（supported/partially/unsupported）→ 原文深链 | 无依据断言占比可量化；项目核心卖点 | 2 天 |
| 3 | 升到 L1（60 篇）+ 重跑基线 | 数字可信度 | 45 分钟 + 15 分钟 |
| 4 | Demo（引用深链可视化） | 可现场演示 | 1 天 |

**M2（结果治理）建议彻底放弃**：参考文献块只占 Top-5 的 7.8%，收益上限不足。

**注意**：第 1 项会改变基线数字，必须重跑 D1 才能与后续改动比较。
D2 那条「不重建数据库」的约束需要为此放开一次。

---

## 11. 术语

| 术语 | 含义 |
|---|---|
| ORB | OpenRAG Bench，arXiv 语料 + 单金标文档的问答基准 |
| RRF | Reciprocal Rank Fusion，`Σ 1/(k+rank)`，本库 k=60 |
| L0/L1/L2 | 20 / 60 / 100 篇三档语料 |
| 门控（gate） | 依据语料覆盖度判断「该不该答」，在 `before_search` 干预检索 |
| 负样本 | 语料外问题，用来测拒答 |
| 探针（probe） | 为隔离单一变量而自建的等价查询对 |
