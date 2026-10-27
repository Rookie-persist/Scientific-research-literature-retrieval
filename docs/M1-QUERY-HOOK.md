# D2 记录：M1 查询理解 Hook

> 执行日期 2026-10-21 · 语料 L0（20 篇 / 3,193 chunk）· 数据库 `dbs/smoke.lancedb`
> 产出：`m1_hook/`（可安装插件）、`m1/acronyms.json`（32 条领域缩写）、4 份配置、3 个脚本、6 份评测记录

## 一、结论先行

**M1 在这套语料上没有检索收益可拿——不是没做好，是没有空间。**

- 检索 recall@5 已经 99.5%（D1 实测），任何"改查询"的改动最多影响剩下那 1 道题。
- 缩写扩展的实测收益：12 对探针里 MRR 从 **0.9583 → 1.0**（1 个位次前移），recall 无变化。**真实但可忽略**。
- 覆盖率门控确有实效：不可答题的中位延迟 **53.9 s → 17.3 s（−68%）**，拒答保持 **10/10**。但它有明确的失效模式（见第五节），不能默认开启。

因此本模块的交付形态是：**机制建成、默认关闭、策略可选**，并用实测数据说明为什么默认关闭。

## 二、建了什么

```
m1_hook/                       # 可安装插件，入口点 haiku.rag.hooks: m1-query
  pyproject.toml
  src/m1_query/hook.py         # QueryUnderstandingHook
m1/acronyms.json               # 32 条，从本库正文抽取
haiku.rag.m1.yaml              # 扩展开，门控 off
haiku.rag.m1.cap.yaml          # 扩展开，门控 cap（limit 减半）
haiku.rag.m1.starve.yaml       # 扩展开，门控 starve（零结果）
haiku.rag.m1.conservative.yaml # 扩展开，门控 conservative（实体感知）
scripts/build_acronym_dict.py  # 抽词典
scripts/probe_acronym.py       # 缩写/全称对照探针
scripts/diag_retrieval.py      # 检索诊断（D1 已建）
```

安装与验证：

```bash
.venv/Scripts/uv.exe pip install --python .venv/Scripts/python.exe --no-build-isolation -e ./m1_hook
.venv/Scripts/python.exe -c "import importlib.metadata as md; print([e.name for e in md.entry_points(group='haiku.rag.hooks')])"
# -> ['m1-query']
```

### 两个能力

**① 双向缩写扩展**。查询里出现词典缩写就补全称，出现全称就补缩写。

采用**追加**而非替换：`What does SICR measure?` → `What does SICR measure? [SICR (Strict Instruction Compliance Ratio)]`。理由是混合检索两路都吃这个改写——保留原词让 FTS 仍命中缩写（缩写在高频文档里出现上千次），追加全称让向量侧拿到语义更完整的查询。

**② 语料覆盖门控**。查询实词在本库词表里找不到（允许 4 字符前缀的词干匹配），判定为"大概率不在语料内"。四种策略：

| 策略 | 行为 | 用途 |
|---|---|---|
| `off` | 只记录不干预 | 默认 |
| `cap` | limit 减半 | 温和收窄 |
| `starve` | 注入 `uri = '__m1_no_match__'`，返回零结果 | 最快拒答 |
| `conservative` | 同 starve，但单个大写专名不触发 | 避免误杀 |

`starve` 用"不可能命中的过滤条件"而非 `limit=0`：实测 `limit=0` 会抛 `ValueError: k must be positive`，而假过滤条件正常返回空列表。

### 词典怎么来的

不引外部词表，直接从本库正文用 `全称 (缩写)` 模式抽，再加一条**首字母子序列校验**。没有这条校验会产出 `US → Food and Drug Administration` 这类跨句错配。校验后 77 对降到 54 对，保留 32 条（出现 ≥3 次）。

## 三、实验一：缩写扩展（探针，n=12）

**为什么不用 ORB 测**：199 条 ORB 查询里只有 **18 条（9.0%）**含词典缩写，用整个基准测这个机制，信号会被稀释到看不见。

**探针设计**：从词典反查金标文档（要求该文档同时含全称与缩写、且唯一），为每个概念造一对等价查询：

- `What is WER?`
- `What is Word Error Rate?`

金标相同，于是差距只能来自查询措辞。

| 指标 | 基线（Hook 关） | Hook 开 |
|---|---|---|
| 缩写形式 recall@5 | 1.000 | 1.000 |
| 全称形式 recall@5 | 1.000 | 1.000 |
| 缩写形式 MRR | 0.9583 | **1.0000** |
| 全称形式 MRR | 0.9583 | **1.0000** |

**解读**：缩写与全称两种问法在基线下**已经等价**（recall 都是 100%）。原因是混合检索的 FTS 侧会字面命中缩写，RRF 只需一路命中就能进 Top-5。扩展开启后 MRR 前移一个位次，**真实但不足以支撑一个模块**。

这条负结果的价值在于：它否掉了"缩写查询检不到论文"这个看似合理的假设，而这个假设是原计划里 M1 的全部理由。

## 四、实验二：覆盖率门控（n=10 负样本）

**动机**：D1 发现负样本延迟是正题的 4 倍（82.5 s vs 20.9 s），因为 `rag.md` 那条"结果不匹配就换词再搜"的规则只在不可答题上触发。

**为什么不能只用 score 判**：D1 已证明 `score` 是 RRF 融合分（1/61 的整数倍），正负样本分布完全重叠。所以改用**语料覆盖度**这个确定性信号。

**信号可分性**（199 条 ORB 正题 vs 10 条负样本）：

| 词干前缀长度 | 负样本命中 | ORB 误报 |
|---|---|---|
| 4 | **10/10** | 5/199（2.5%） |
| 5 | 10/10 | 9/199 |
| 6 | 10/10 | 10/199 |

取 4。误报的 5 条里有 4 条是 LaTeX 片段（`gamma`、`theta`、`mathbf`）——这些查询本来就检索不准。

**结果**（10 条负样本，`starve` vs `off`）：

| 策略 | 均值 | 中位 | 最小 | 最大 | 拒答 |
|---|---|---|---|---|---|
| off | 49.0 s | 53.9 s | 9.8 s | 82.1 s | 10/10 |
| starve | **23.4 s** | **17.3 s** | 8.7 s | 61.0 s | 10/10 |

逐题 10/10 全部持平或更快（最大改善 neg9：82.1 s → 44.8 s），**拒答正确性没有损失**。

## 五、门控的失效模式（必须知道）

在 4 正题 + 4 负样本的对照里，`starve` 和 `cap` 都**误杀了一个可答题**：

| 题 | off | cap | starve | conservative |
|---|---|---|---|---|
| Tesla 交易建议（b54940c9） | 17.7 s，**2 条引用** | 13.6 s，**0 条引用** | 12.8 s，**0 条引用** | 19.8 s，**2 条引用** |
| neg1 metformin | 94.5 s | 66.4 s | 12.5 s | 6.9 s |
| neg2 FIFA | 50.3 s | 19.5 s | 12.0 s | **269.6 s** |
| neg3 FlashAttention | 114.0 s | 27.6 s | 10.2 s | 127.8 s |
| neg4 tungsten | 20.4 s | 31.4 s | 7.8 s | 8.9 s |

**两个发现**：

1. **`tesla` 是词表外的专名**（语料里只写 TSLA），于是被当成"语料缺失"证据，零结果 → 引用从 2 条掉到 0。这题本来就答错（模型称文档没给建议），但**证据被抽走让错误更难被发现**。这就是 `conservative` 策略存在的原因：单个大写专名不算缺失。实测它确实保住了 2 条引用。

2. **`conservative` 的延迟反而失控**：neg2 从 50.3 s 涨到 **269.6 s**。原因是"零结果"没有触发模型的收手，反而让它换了 4 次关键词（`max_searches=5`），每次都空手而归。**饿死检索不等于让它更快放弃**，在个别情况下会诱发更多重试。n=4 的样本量无法判定这是系统性现象还是单次波动，但足以说明这个策略的延迟收益不稳定。

## 六、决定

**默认关闭**（`coverage_policy: "off"`）。理由不是保守，是三条实测：

1. 检索 recall@5 已 99.5%，M1 的收益上限只有 1 道题。
2. 门控确有 −68% 中位延迟，但代价是把"语料外专名"这类查询的证据一起抽走——而这类查询里混着可答题。
3. 更省事的替代方案存在：不可答题的 4 倍延迟，根源是**重试规则**，不是检索质量。改 `rag.md` 的重试条件（或调 `qa.max_searches`）比加一个门控更直接，且不碰检索。

**如果后续要用**：选 `conservative`，并把 `max_searches` 从 5 降到 3，压制"饿死后反复重试"的路径。

## 七、对计划的修正

D2 原计划把 M1 定为"缩写扩展 + over-fetch"，理由是"缩写类查询检不到全称论文"。**这个理由被实验否掉了**（第三节）。

同时 D1 的两个失败案例都是解析保真度问题：

| 失败 | 原因 | 归属 |
|---|---|---|
| Q4 WTT 公式答错 | 块里是 `<!-- formula-not-decoded -->`，13.3% 的块如此 | 解析（公式识别） |
| Q15 Tesla 该答却拒答 | 金标来自表格，被压平成 `Cumulative Return TSLA -0.1633 -0.1829 ...`，列头丢失 | 解析（表格结构） |

**建议**：M1 到此收口（机制保留、默认关闭），把原定给 M1 的时间转到**解析保真度**——公式识别（`do_formula_enrichment`）与表格精确模式（`table_mode: accurate`）。这两项都不需要新下载（CodeFormula 与 TableFormer 已在 1.3 GB 缓存里），代价只是一次重摄取。

这也意味着 D2 的"不重建数据库"约束需要放开一次。

## 八、复现

```bash
export HAIKU_RAG_CONFIG_PATH=./haiku.rag.yaml

# 词典（32 条）
.venv/Scripts/python.exe scripts/build_acronym_dict.py --db ./dbs/smoke.lancedb --out m1/acronyms.json

# 实验一：缩写探针（基线 / 开启）
.venv/Scripts/python.exe scripts/probe_acronym.py --db ./dbs/smoke.lancedb --min-occurrences 20 --limit 5 --out eval/probe-acronym-baseline.json
HAIKU_RAG_CONFIG_PATH=./haiku.rag.m1.yaml .venv/Scripts/python.exe scripts/probe_acronym.py --db ./dbs/smoke.lancedb --min-occurrences 20 --limit 5 --out eval/probe-acronym-hook.json

# 实验二：10 条负样本，开关对照
.venv/Scripts/python.exe -u scripts/run_baseline.py --db ./dbs/smoke.lancedb --subset subset_20 \
  --out eval/neg10-off.jsonl --policy --negatives-only --negatives-file m1/negatives10.json
HAIKU_RAG_CONFIG_PATH=./haiku.rag.m1.starve.yaml .venv/Scripts/python.exe -u scripts/run_baseline.py \
  --db ./dbs/smoke.lancedb --subset subset_20 --out eval/neg10-starve.jsonl --policy \
  --negatives-only --negatives-file m1/negatives10.json
```

## 九、一个工程坑

配置里 `coverage_policy: off` 会被 **YAML 1.1 解析成布尔 `False`**，Hook 拿到的不是字符串。表现为 trace 里 `"policy": false`、门控逻辑静默失效。修法两条：配置里加引号 `"off"`，以及在 Hook 里对布尔做归一化。这类"配置写了但没生效"的问题不会报错，只能靠 trace 发现——这也是把每次决策写进 `m1/trace-*.jsonl` 的价值。
