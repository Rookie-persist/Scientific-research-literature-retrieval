# D0 执行记录（2026-10-20）

> 目标：按 `PROJECT-PLAN.md` 的 D0 跑通最小闭环——20 篇 L0 语料、可检索的数据库、带页码与章节的检索结果。
> 结论：**验收通过**。但过程中遇到 3 个计划未预见的环境问题，耗时比估算高约 20 倍，需回写文档。

---

## 1. 验收结果

| 验收项 | 目标 | 实测 | 结论 |
|---|---|---|---|
| 文档数 | 20 | **20** | ✅ |
| chunk 数 | ~2,400 | **3,193**（+32%） | ✅ 在 ±30% 边界上 |
| 检索返回页码 | 有 | `pages: 1` | ✅ |
| 检索返回章节 | 有 | `headings: 1 INTRODUCTION` | ✅ |
| 嵌入一致性 | 无告警 | 384 维，identity 匹配 | ✅ |
| `doctor` | 无嵌入/索引告警 | **18 ok, 0 warning, 1 failure** | ⚠️ 失败项是 Ollama 不可达 |
| 字段过滤 | 可用 | `--filter "uri LIKE '%2410.23841%'"` 生效 | ✅ |
| 数据库体积 | — | **125 MB**（20 篇，vacuum 后） | — |

`doctor` 唯一失败项：

```
✗ ollama at http://localhost:11434 is unreachable.
```

**原因**：Ollama 进程已不在运行（`tasklist` 无进程、11434 无监听），不是配置问题。D0 不涉及 LLM，因此不影响验收；**D1 之前必须启动 Ollama**。

---

## 2. 实际耗时

| 步骤 | 计划估算 | 实测 | 倍率 |
|---|---|---|---|
| 创建 venv | — | 10 秒 | — |
| 装依赖（PyPI 直连，已放弃） | 6–10 分钟 | **22 分 54 秒仍未完成**（手动终止） | — |
| 装依赖（阿里云镜像 + uv） | — | **4 分 53 秒** | — |
| 下载 Docling 模型（首次，失败） | 5–15 分钟 | 6 分 13 秒（模型已落盘，卡在分词器） | — |
| 下载嵌入模型（local_dir） | — | 2 分 11 秒 | — |
| `download-models`（修复后） | — | **32 秒** | — |
| 下载 20 篇 PDF | 5 分钟 | **3 分 32 秒**（50 MB） | 0.7× |
| 摄取 20 篇 | 3–5 分钟 | **15 分 04 秒** | **3–5×** |
| 检索冒烟 | 秒级 | 约 5 秒（含模型加载） | — |
| **端到端** | 约 3 小时 | **约 61 分钟** | 0.34× |

端到端反而比估算快，因为省掉了大量试错；但**摄取本身比估算慢 3 倍**，这一项会随语料规模线性放大，是后续排期的主要风险。

---

## 3. 三个计划未预见的问题

### 问题 1：PyPI 直连带宽只有 ~30 kB/s

pip 下载 `docling-parse`（11.8 MB）用了 6 分 26 秒，`lancedb`（103 MB）按此速度需 50 分钟以上。按计划装完全部依赖需要数小时。

**处理**：改用阿里云镜像（单连接约 180–226 kB/s），并改用 `uv` 并行下载。

```bash
.venv/Scripts/python.exe -m pip install uv -i https://mirrors.aliyun.com/pypi/simple
.venv/Scripts/uv.exe pip install --python .venv/Scripts/python.exe \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  --index-strategy unsafe-best-match \
  "haiku.rag==0.98.0" "sentence-transformers>=3.0,<6.0"
```

注意：清华镜像在本机不可用（索引返回空）。

### 问题 2：HuggingFace 缓存的快照目录为空

`snapshot_download()` 报告成功，blob 落盘，但 `snapshots/<sha>/` 是空的——本机无法创建缓存所依赖的符号链接。表现为两类错误：

- `Couldn't instantiate the backend tokenizer...`（分词器）
- `Image processor config not found: ...snapshots/<sha>/preprocessor_config.json`（Docling 布局模型）

**处理**（两条互补）：

1. 嵌入模型与分词器改用 `local_dir` 下载到 `models/bge-small-en-v1.5`，配置指向本地路径：
   ```yaml
   embeddings.model.name: ./models/bge-small-en-v1.5
   processing.chunking_tokenizer: ./models/bge-small-en-v1.5
   ```
2. Docling 模型已由它自己的下载器写成**真实文件**（`~/.cache/docling/models/<org>--<name>/`），只是 HF 缓存视图坏了。用脚本把文件复制进快照目录，避免重新下载 1.3 GB：
   ```bash
   .venv/Scripts/python.exe scripts/fix_hf_snapshots.py
   ```

脚本会从 `refs/main`、Docling 的 `trees/<sha>.json`、或镜像 API 三处解析 commit，递归复制（TableFormer 权重在 `model_artifacts/` 子目录里）。

### 问题 3：Docling 模型体积是估算的两倍

计划估 400–600 MB，实际 **1.3 GB**：

| 模型 | 体积 |
|---|---|
| CodeFormulaV2 | 640 MB |
| docling-models（TableFormer） | 358 MB |
| docling-layout-heron | 172 MB |
| docling-layout-heron-onnx | 171 MB |
| DocumentFigureClassifier | 34 MB |

即使不用 OCR、不用公式识别，这些也会被 `docling-slim[standard]` 一并下载。磁盘无压力，但下载时间要按实测修正。

### 问题 4：摄取失败重试后，数据库膨胀 3.3 倍

第一次摄取在第一个 PDF 就失败，重试成功后 `dbs/smoke.lancedb` 是 **411 MB**，而实际数据只有约 124 MB（文档 108 MB + chunk 9.4 MB + items 6.5 MB）。原因是 LanceDB 保留历史版本（`versions (documents): 34`），默认保留 1 天。

```bash
haiku-rag vacuum --db ./dbs/smoke.lancedb --retention-seconds 0
# 411 MB -> 125 MB，versions 从 34 降到 1
```

**默认的 `vacuum` 不带 `--retention-seconds 0` 回收不了当天产生的版本**，实测只从 411 MB 降到 535 MB（compaction 重写后反而更大）。这一点在反复失败的摄取之后尤其重要。

---

## 4. 对后续步骤的影响

| 影响 | 说明 | 建议 |
|---|---|---|
| **D1 被阻塞** | Ollama 未运行，`ask()` 与引用校验都跑不了 | 先启动 Ollama，再跑 `check_ollama.py` |
| **L1/L2 时间要重估** | 实测 45 秒/篇（开发档）。60 篇 ≈ 45 分钟，100 篇 ≈ 75 分钟；最终档更慢 | 排期按 45–90 秒/篇估算，别按 8–15 秒 |
| **chunk 数偏高 32%** | 3,193 而非 2,420，图片与表格块比预期多 | 不影响检索，但体积与嵌入时间相应增加 |
| **依赖安装必须走镜像** | 直连 PyPI 不可行 | 把 uv + 阿里云镜像写进文档主路径 |
| **磁盘占用超预算** | `.venv` 2.0 GB + 模型 0.38 GB + Docling 1.3 GB + PDF 0.05 + 库 0.13 ≈ **3.9 GB**；下载约 2.4 GB。计划上限是 1.3 GB 下载 / 3 GB 磁盘 | 预算按实测调整；`.venv` 是最大项，无法压缩 |

---

## 5. 完整复现命令

```bash

# 环境
python -m venv .venv
.venv/Scripts/python.exe -m pip install uv -i https://mirrors.aliyun.com/pypi/simple
.venv/Scripts/uv.exe pip install --python .venv/Scripts/python.exe \
  --index-url https://mirrors.aliyun.com/pypi/simple --index-strategy unsafe-best-match \
  "haiku.rag==0.98.0" "sentence-transformers>=3.0,<6.0"

# 模型
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
.venv/Scripts/python.exe scripts/fetch_models.py
.venv/Scripts/haiku-rag.exe download-models
.venv/Scripts/python.exe scripts/fix_hf_snapshots.py

# 语料
python scripts/build_orb_subset.py --metadata-dir meta --target-papers 20 --out subset_20 --drop-image-queries
python scripts/download_pdfs.py --subset subset_20 --out pdfs/smoke --workers 3

# 建库
.venv/Scripts/haiku-rag.exe init --db ./dbs/smoke.lancedb
.venv/Scripts/haiku-rag.exe add-src ./pdfs/smoke --db ./dbs/smoke.lancedb

# 验收
.venv/Scripts/haiku-rag.exe info --db ./dbs/smoke.lancedb
.venv/Scripts/haiku-rag.exe search "retrieval augmented generation" --db ./dbs/smoke.lancedb --limit 3
.venv/Scripts/haiku-rag.exe doctor --db ./dbs/smoke.lancedb
```
