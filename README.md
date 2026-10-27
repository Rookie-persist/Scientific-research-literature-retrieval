# 科研文献检索基线（haiku.rag 二次开发）

面向 arXiv 论文的检索与问答评测基线，基于开源 RAG 库 [haiku.rag](https://github.com/ggozad/haiku.rag) 二次开发。

项目要回答一个具体问题：**在科研文献检索里，"检得到"和"答得对"分别被什么卡住。**

为此提供了三样东西：

- **可复现的评测链路**：语料子集化、检索诊断、问答基线，全部指标可追溯到原始记录（`eval/`）
- **一个可插拔的查询理解 Hook**：走 haiku.rag 官方 `haiku.rag.hooks` 入口点，不改检索内核，支持双向缩写扩展与语料覆盖门控
- **一组实测结论**：哪些改造有效、哪些假设不成立，都带数字与证据文件

> 当前结论：该语料上检索 recall@5 已达 99.5%，瓶颈不在检索而在**解析保真度**——13.3% 的文本块在解析阶段丢失了公式。详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 技术栈

| 层 | 选型 |
|---|---|
| RAG 框架 | haiku.rag 0.98.0（Pydantic AI Agent + 官方 Hook 扩展点） |
| 向量库 / 检索 | LanceDB 0.39（向量 + FTS 混合检索，RRF 融合 k=60） |
| 文档解析 | Docling 2.127（版面 / 表格 TableFormer / 公式 CodeFormula） |
| 嵌入 | sentence-transformers 5.7 + `BAAI/bge-small-en-v1.5`（384 维） |
| 生成 | Ollama `qwen3.5:9b`（16k 上下文变体） |
| 语料 | OpenRAG Bench（HuggingFace `vectara/open_ragbench`，arXiv 子集） |
| 运行环境 | Python 3.11+，Windows / Linux / macOS 均可 |

## 目录结构

```
.
├── README.md
├── requirements.txt
├── haiku.rag.yaml                    # 开发档配置（默认加载）
├── haiku.rag.final.yaml              # 最终档（高质量解析）
├── haiku.rag.m1*.yaml                # M1 Hook 四档实验配置
├── scripts/                          # 数据准备、建库、评测脚本（11 个）
│   ├── fetch_orb_metadata.py         # 经镜像下载 ORB 元数据
│   ├── build_orb_subset.py           # 子集化，同步裁剪题目
│   ├── download_pdfs.py              # PDF 下载（限速/重试/断点续传）
│   ├── fetch_models.py               # 嵌入模型下载（local_dir 模式）
│   ├── fix_hf_snapshots.py           # 修补 HF 快照缓存
│   ├── check_ollama.py               # Ollama 预检
│   ├── diag_retrieval.py             # 检索诊断（recall@k / MRR / 块构成）
│   ├── run_baseline.py               # 问答基线
│   ├── ask_policy.py                 # 带引用策略的 ask 封装
│   ├── build_acronym_dict.py         # 从语料抽取缩写词典
│   └── probe_acronym.py              # 缩写/全称对照探针
├── m1_hook/                          # 可安装插件（entry point: m1-query）
│   ├── pyproject.toml
│   └── src/m1_query/hook.py
├── meta/                             # ORB 原始元数据（1.6 MB）
├── subset_20/                        # 20 篇子集（queries/qrels/answers/ids）
├── m1/                               # 缩写词典、负样本
├── eval/                             # 全部评测记录（含 traces/）
└── docs/
    ├── ARCHITECTURE.md               # 架构、设计取舍、实测结论、已知陷阱
    ├── BUILD.md                      # 建库与故障对照
    ├── BASELINE.md                   # 基线评测记录
    ├── M1-QUERY-HOOK.md              # M1 Hook 的实验记录
    └── D0-SETUP.md                   # 环境搭建与四个环境问题
```

## 快速开始

### 0. 环境要求

- Python 3.11+
- [Ollama](https://ollama.com) 已安装（问答阶段需要；建库与检索不需要）
- 磁盘约 4 GB（依赖 2.0 GB + Docling 模型 1.3 GB + 嵌入模型 0.4 GB + 语料与库）

### 1. 安装依赖

PyPI 直连在国内较慢，建议走镜像并用 `uv` 并行安装：

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install uv -i https://mirrors.aliyun.com/pypi/simple
.venv/Scripts/uv.exe pip install --python .venv/Scripts/python.exe \
  --index-url https://mirrors.aliyun.com/pypi/simple --index-strategy unsafe-best-match \
  -r requirements.txt
```

Linux / macOS 把 `.venv/Scripts/` 换成 `.venv/bin/`。

### 2. 设置环境变量

```bash
export HF_ENDPOINT=https://hf-mirror.com      # HuggingFace 镜像
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_SYMLINKS=1              # 部分文件系统无法建软链
export HAIKU_RAG_CONFIG_PATH=./haiku.rag.yaml
```

Windows 用 `set` 或在 PowerShell 中用 `$env:NAME="value"`。

### 3. 准备模型

```bash
# 嵌入模型（下载到 ./models/，配置已指向该目录）
python scripts/fetch_models.py

# Docling 版面 / 表格 / 公式模型
haiku-rag download-models

# 若上一步报 "Image processor config not found" 或 tokenizer 错误，执行快照修补
python scripts/fix_hf_snapshots.py
```

### 4. 准备语料

仓库已包含 20 篇的子集元数据（`subset_20/`），PDF 需自行下载（约 48 MB）：

```bash
python scripts/download_pdfs.py --subset subset_20 --out pdfs/smoke --workers 3
```

如需更换规模（推荐 60 / 100 篇）：

```bash
python scripts/build_orb_subset.py --metadata-dir meta --target-papers 60 \
  --out subset_60 --drop-image-queries
python scripts/download_pdfs.py --subset subset_60 --out pdfs/dev --workers 3
```

### 5. 建库

```bash
haiku-rag init --db ./dbs/smoke.lancedb
haiku-rag add-src ./pdfs/smoke --db ./dbs/smoke.lancedb

# 回收 LanceDB 版本历史（失败重试后体积会膨胀数倍）
haiku-rag vacuum --db ./dbs/smoke.lancedb --retention-seconds 0

haiku-rag info --db ./dbs/smoke.lancedb
haiku-rag search "retrieval augmented generation" --db ./dbs/smoke.lancedb --limit 3
```

开发档约 45 秒/篇（20 篇约 15 分钟），慢在表格结构推理。

### 6. 运行评测

```bash
# 检索诊断：不调用 LLM，约 1 分钟
python scripts/diag_retrieval.py --db ./dbs/smoke.lancedb \
  --subset subset_20 --limit 5 --out eval/retrieval-diag.json

# 问答基线：20 题（16 金标 + 4 负样本），约 10 分钟
python scripts/run_baseline.py --db ./dbs/smoke.lancedb \
  --subset subset_20 --out eval/baseline.jsonl --policy
```

问答需要 Ollama。本项目使用 16k 上下文变体：

```bash
cat > Modelfile <<'EOF'
FROM qwen3.5:9b
PARAMETER num_ctx 16384
PARAMETER presence_penalty 0
PARAMETER temperature 0.3
EOF
ollama create qwen35-9b-16k -f Modelfile
python scripts/check_ollama.py --model qwen35-9b-16k
```

> 这两个参数不是可选项。Ollama 默认 `num_ctx=4096` 装不下系统提示，模型会丢失问题；
> 模型自带的 `presence_penalty=1.5` 会惩罚问题中的词，导致话题漂移。两者都表现为"答案跑题"，难以定位。

### 7. 启用查询理解 Hook（可选）

```bash
uv pip install --python .venv/Scripts/python.exe --no-build-isolation -e ./m1_hook

# 用带 Hook 的配置启动
export HAIKU_RAG_CONFIG_PATH=./haiku.rag.m1.yaml
python scripts/run_baseline.py --db ./dbs/smoke.lancedb --subset subset_20 \
  --out eval/ab-m1-off.jsonl --policy --qids 01416eb6,58ee5bb5,e3c274b9,b54940c9
```

四种门控策略通过 `haiku.rag.m1*.yaml` 切换，实测对比见 [docs/M1-QUERY-HOOK.md](docs/M1-QUERY-HOOK.md)。

## 配置说明

| 文件 | 用途 | 关键差异 |
|---|---|---|
| `haiku.rag.yaml` | 开发档，默认加载 | `pypdfium2`、`do_ocr: false`、`table_mode: fast`、`max_tokens: 4096` |
| `haiku.rag.final.yaml` | 最终档，`--config` 指定 | `table_mode: accurate`，解析更慢更准 |
| `haiku.rag.m1.yaml` | Hook：仅缩写扩展 | `coverage_policy: "off"` |
| `haiku.rag.m1.cap.yaml` | Hook：门控收窄 | `limit` 减半 |
| `haiku.rag.m1.starve.yaml` | Hook：门控短路 | 返回零结果 |
| `haiku.rag.m1.conservative.yaml` | Hook：实体感知门控 | 单个大写专名不触发 |

两档配置的**嵌入模型必须一致**，否则两个库无法互相打开。

## 评测结果（20 篇 / 3,193 chunk）

| 指标 | 值 |
|---|---|
| 检索 recall@5 / MRR | 99.5% / 0.9912 |
| 答案正确率（16 道金标题） | 14/16 = 87.5% |
| 负样本拒答（4 道） | 4/4，零引用 |
| 引用率 / 带页码率 | 93.8% / 93.8% |
| 延迟（正题 / 负样本） | 20.9 s / 49.0 s |
| Top-5 中公式残缺块 | 18.4% |

完整数据与归因见 [docs/BASELINE.md](docs/BASELINE.md) 与 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 已知陷阱

1. **嵌入模型建库后不可更换**：维度不可变，换维度会抛 `ConfigMismatchError`，只能全量重嵌。
2. **`SearchResult.score` 是 RRF 融合分**（`Σ 1/(60+rank)`），不含语义相似度，不能当相关性阈值使用。
3. **`--config` 是全局参数**，必须写在子命令之前：`haiku-rag --config x.yaml init`。
4. **YAML 1.1 会把裸写的 `off` 解析成布尔 `false`**，配置里请加引号。
5. **摄取失败重试后数据库会膨胀**（LanceDB 保留版本历史），用 `vacuum --retention-seconds 0` 回收。
6. **Hook 的 `lifespan` 需要 `@asynccontextmanager`**，否则客户端打开时报 `async_generator` 无 `__aenter__`。

其余陷阱（含 Docling 缓存、评测参数、API 字段差异）见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 第 9 节。

## 数据与致谢

- 语料与评测问题来自 [OpenRAG Bench](https://huggingface.co/datasets/vectara/open_ragbench)（arXiv 子集）
- RAG 框架为 [haiku.rag](https://github.com/ggozad/haiku.rag)（MIT），本仓库在其之上做扩展与评测，未修改其源码
- 文档解析使用 [Docling](https://github.com/docling-project/docling)

本仓库不包含 PDF 原文与向量库文件，二者可由上述脚本重新生成。
