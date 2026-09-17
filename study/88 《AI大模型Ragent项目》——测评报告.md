# 《AI大模型Ragent项目》——测评报告

上一篇把 RAGAS 的五个坑讲完了——同源偏置、单跑方差、中文 NaN、成本失控、Judge 误判。你现在知道哪些分数差异是真的，哪些是噪声，怎么用多轮取均值压制方差，怎么用人工校准修正误判。

指标也全算完了。`_scores.json` 里躺着十几个指标的 overall、by_intent_l1、by_intent_l2、per_sample——从意图准确率到 TTFT P50，从 faithfulness 到 answer_correctness，该有的都有了。

然后呢？

打个比方，你考完试了，答题卡已经批完了，每道题的分数都有了。但你把一张写满分数的 Excel 发到班级群里，家长看得懂吗？班主任能拿着这张表在家长会上讲两分钟吗？你自己能从里面一眼看出哪个知识点最弱、下一步该补哪块吗？

评测报告也是一样的道理。`_scores.json` 是原始成绩单，但不同的人需要不同的呈现方式。CI 脚本只需要一个 JSON 判阈值，研发需要看逐样本明细找具体问题，老板需要打开浏览器看几个大数字和一张漂亮的看板。

这篇讲怎么把指标变成不同角色能消费的报告——三种人，三种产物，一套数据源。

## 报告产物总览

### 1. 三类受众，各看什么

想一想你平时工作里的场景：

- CI 流水线只需要一个 JSON，判断 Hit@5 有没有跌破 90%，跌了就阻断合入——它不需要看图表，不需要看失败样本，只需要一个布尔值
- 研发同事想知道哪几条样本最差、差在哪个环节——需要逐样本的明细表和失败归因，能直接定位到具体 query
- 技术负责人要在周会上花两分钟看一眼系统质量——需要几个大数字加一张漂亮的看板，翻几页就能讲完

一份报告服务不了这三种需求。所以评测产出拆成了五个文件，各司其职：

| 产物 | 文件路径 | 受众 | 格式 | 用途 |
| --- | --- | --- | --- | --- |
| _scores.json | reports/<run>/ | CI / 自动化脚本 | JSON | 所有指标的全量数据，给脚本读 |
| report.md | 同目录 | 研发 / Reviewer | Markdown | 指标总览 + 意图切片表 |
| per_sample.csv | 同目录 | 研发 / Reviewer | CSV | 逐样本明细 + 人工复核列 |
| failures.jsonl | 同目录 | 研发 / Reviewer | JSONL | 失败样本归因，一行一条 |
| slides.html | 同目录 + reports/latest_slides.html | 老板 / 周会 | HTML | 16:9 横向翻页，浏览器全屏演示 |

所有产物都在 `reports/<run>/` 目录下。`<run>` 是 runs 文件的名称，类似 `v1_20260601_143022`——哪一次录制产生的结果，一目了然。

### 2. 数据怎么流的：三段缓存

五个产物不是各自独立生成的，它们共享同一个数据源 `_scores.json`。而 `_scores.json` 本身也不是凭空出来的——整个评测的数据流是三段式的，每段产物独立缓存：

![图片.png](assets/88-01.png)

用一句话概括这个设计的核心思路：**录制一次，评分 N 次，报告 M 次**。

runner 跑一次调两个接口，成本最高（RAGAS 要花钱调 judge），录好之后落 `runs/*.jsonl`。score 阶段从 runs 文件算指标，改了 judge 模型或调了参数之后可以复用同一份 runs 重跑——不花接口费。report 阶段从 `_scores.json` 渲染各种产物，秒级出结果——改失败判定阈值、调报告格式、重跑人工校准，都不需要重新调被评系统的接口。

三段缓存互不耦合，这是优化过程中反复跑评测的基础。

## 给研发看的三份报告

研发需要的是**能直接定位问题的信息**：整体水平怎么样、哪个场景最差、哪几条样本出了什么问题。这三份文件从粗到细，覆盖了研发 review 评测结果的完整流程。

### 1. `report.md`：一页纸总览

`report.md` 由 `markdown.py` 的 `write_all()` 函数生成，包含三部分内容：

**自建指标总览表**：按 `BASELINE_ORDER` 列出 13 个自建指标——`intent_top1`、Hit@1/3/5/10、Recall@1/3/5/10、MRR@10、误拒率、错答率、TTFT 等，每个给出 overall 值。一眼看到系统各个环节的健康度。

**RAGAS 指标表**：5 个 RAGAS 指标的 overall 值——`faithfulness`、`answer_relevancy`、`answer_correctness`、`context_precision`、`context_recall`。语义层面的质量判断。

**二级意图切片表**：这是这份报告最有价值的部分。每个二级意图一行，列出该意图下的核心指标值。

切片表的价值在哪？举个例子：overall Hit@5 = 90%，看起来还不错。但切到 by_intent_l2 一看，`S1_选购推荐` 只有 60%，`S14_退换货咨询` 是 100%。退换货类问题检索得很好，但选购推荐类一半以上没命中——90% 的 overall 把一个严重问题给掩盖了。

优化资源永远有限。切片表让你知道该把精力集中在哪几个意图场景上，而不是眉毛胡子一把抓。

### 2. `per_sample.csv`：逐样本明细 + 人工复核列

`report.md` 告诉你哪个意图最差，`per_sample.csv` 让你钻到每一条样本里去看。

每行一条评估样本，列分三组：

| 列类型 | 包含内容 |
| --- | --- |
| 基础信息列 | query_id、intent_l1、intent_l2、difficulty、requires_rag、final_status |
| 指标值列 | 所有指标的 per_sample 值（Hit@5 是 0 或 1、faithfulness 是 0~1 的浮点数等） |
| 人工复核列 | 5 个 *_manual 空列（对应 5 个 RAGAS 指标） |

前两组好理解，重点说人工复核列。上一篇讲过 RAGAS 偶尔会误判——judge 给某条样本的 faithfulness 打了 0.3，但你人工看过之后发现回答完全忠实于上下文，只是拆 claim 的方式有问题。这时候不需要重跑 RAGAS（贵且慢），直接在 CSV 里改就行。

#### 2.1 人工复核的工作流程

整个流程分五步：

- `score` 阶段跑完后生成 `per_sample.csv`，5 个 RAGAS 指标各自对应一个 `{metric_name}_manual` 列，初始为空
- Reviewer 打开 CSV（Excel 或任何编辑器都行），对有疑问的样本在对应 `*_manual` 列填入人工分数——支持 `0.8`、`80%`、`0.8000` 这几种格式
- 重跑 `python -m eval rag report` 命令
- `load_manual_overrides()` 自动读取已填写的人工分数
- `apply_manual_overrides()` 按人工列优先、空值回退 RAGAS 的策略重算——替换 `per_sample` 对应值，重算 `overall`、`by_intent_l1`、`by_intent_l2`

重跑后的报告和 slides 自动使用修正后的分数。`meta` 里会记录 `manual_overrides` 数量和 `manual_policy`，所有人都能看到有多少分数被人工修正过——透明可审计。

#### 2.2 为什么需要这个设计

听起来有点绕——为什么不直接重跑 RAGAS 让 judge 再评一次？

原因很现实：重跑一次 RAGAS 要十几分钟、花几十块钱，而且重跑也不保证 judge 这次就判对了（上一篇讲过，方差还在）。人工校准是最后一道防线——零成本、秒出、结果可控。

整个流程不需要重新调 judge API，只是在 report 阶段用人工分替换自动分再重算。

> 不需要每次跑完都逐条审。建议在 Release 前对低分样本抽查、在分数异常波动时查下降最多的指标、在高风险意图（售后政策、保修条款、价格类）上不管分高低都看一眼。大部分样本的自动评分是靠谱的。

### 3. `failures.jsonl`：失败样本归因

`report.md` 告诉你整体水平，`per_sample.csv` 让你逐条看分数，`failures.jsonl` 更进一步——直接告诉你哪些样本出了问题、问题出在哪个环节，是研发最直接的行动指引。

#### 3.1 五种失败类型

一条样本只要命中以下任一条件就算失败：

| 失败类型 | 判定条件 | 含义 |
| --- | --- | --- |
| hit@5_miss | Hit@5 == 0 | 检索前 5 完全未命中目标文档 |
| answer_correctness_low | answer_correctness < 0.5 | 端到端答案严重不对 |
| faithfulness_low | faithfulness < 0.5 | 严重幻觉，过半内容没有上下文支撑 |
| refused_when_required | requires_rag=true 且检索结果为空 | 应该走 RAG 但被误拒了 |
| over_retrieved | requires_rag=false 且检索结果非空 | 不该走 RAG 但过召回了 |

这五种覆盖了 RAG 系统最常见的故障模式：检索没命中、答案不对、幻觉严重、误拒、过召回。

#### 3.2 多原因合并：看到完整故障链路

关键设计：**一条样本可以同时命中多条失败原因**。

举个例子，某条样本的 `failure_reasons` 是 `["hit@5_miss", "answer_correctness_low(0.35)"]`——检索前 5 完全没命中目标文档，答案正确性也只有 0.35。这两个原因不是巧合，而是因果链：检索没命中 → 给模型的上下文里没有正确信息 → 答案自然也差了。

如果只标一个原因，你可能会以为是生成环节的问题（answer_correctness 低），去调 prompt。但真正的根因在检索——多原因合并让你看到完整的故障链路，直接定位根因。

每条失败记录包含完整上下文：

```
{
  "query_id": "S1-03",
  "intent_l1": "SUPPORT",
  "intent_l2": "S1_选购推荐",
  "requires_rag": true,
  "user_input": "3000 元左右的手机有什么推荐？",
  "reference_doc_ids": ["product_phone_mid"],
  "retrieved_doc_ids": ["product_phone_high", "product_phone_low"],
  "first_token_ms": 2340,
  "latency_ms": 8500,
  "final_status": "success",
  "response_preview": "为您推荐以下手机...",
  "failure_reasons": ["hit@5_miss", "answer_correctness_low(0.35)"],
  "scores": {"hit@5": 0.0, "answer_correctness": 0.35, "faithfulness": 0.72}
}
```

拿到这条记录，研发怎么用？看 `failure_reasons` 知道哪个环节出了问题。看 `retrieved_doc_ids` vs `reference_doc_ids` 知道召回偏了多少——该找 `product_phone_mid`（中端手机），实际找到了 `product_phone_high` 和 `product_phone_low`（高端和低端），价格区间路由错了。看 `response_preview` 知道模型具体回了什么。

一条记录，从故障现象到可能的根因，信息全了。

## 给负责人看的 slides.html

研发看 markdown 和 CSV 够了，但老板不会打开这些文件。周会汇报需要的是：打开浏览器，全屏，翻页，几个大数字，几张图表，两分钟讲完。

`slides.html` 就是干这个的。由 `slides.py` 的 `build_slides()` 函数生成，16:9 比例的横向翻页 HTML 文件，浏览器全屏打开即可演示。同时会复制一份到 `reports/latest_slides.html`，方便快速打开最新报告。

### 1. 两套模板风格

| 风格 | 模板文件 | 视觉特征 |
| --- | --- | --- |
| 瑞士国际主义（默认） | swiss_template.html | 无衬线字体、网格点阵背景、IKB 蓝 / 柠檬黄 / 柠檬绿 / 安全橙高亮 |
| 电子杂志 | magazine_template.html | 衬线字体、流体 WebGL 背景、暖色调 |

模板提供 CSS 样式（明暗主题切换、flex 布局、动画标记 `data-anim`）和可选的 WebGL canvas 背景。内容通过模板里的 `<!-- SLIDES_HERE -->` 占位符注入——模板管样式，代码管内容，互不干扰。

### 2. 页面编排

整个 deck 从封面到收束，编排如下：

| 页序 | 内容 | 说明 |
| --- | --- | --- |
| P1 | 封面 Hero | 项目名、样本总数、状态分布（success / refused / error） |
| P2 | 核心 KPI 四宫格 | Intent Top-1、Hit@5、Answer Correctness、Faithfulness |
| P3 | 次级 KPI 九宫格 | Recall@5、MRR@10、Context Precision / Recall、Answer Relevancy、TTFT、误拒率、错答率、兜底率 |
| P4 | 幕间过渡页 | Act II 引出检索主题 |
| P5+ | 二级意图切片 | by_intent_l2 表格，每页 6 行，多页自动分页（标题带 1/3 分页标记） |
| P(n-3)+ | 失败样例卡片 | 每页 4 条，明暗交替配色。无失败样本时显示 clean sweep 祝贺页 |
| P(n-1) | 逐样本明细表 | 每页 8 行，query_id、意图、核心指标值 |
| Pn | 收束页 | 四宫格回顾核心 KPI + next steps |

### 3. 编排背后的设计思路

为什么这么排？几个关键决策值得说说。

**核心 KPI 为什么只放 4 个？** 负责人的注意力是有限的。Intent Top-1（意图对不对）、Hit@5（检索准不准）、Answer Correctness（答案对不对）、Faithfulness（有没有编）——这四个数字覆盖了 RAG 链路的四个关键环节。多一个都是干扰，少一个都不完整。次级 KPI 放第三页，想看的人翻过去，不想看的人直接跳过。

**失败样例为什么要放？** 没有具体例子的报告不可信。只说 faithfulness 0.92，没有体感；但给一条具体的失败样本——用户问退货政策，系统编了一个不存在的 7 天无理由退款，faithfulness 只有 0.3——相关人员立刻理解这个数字背后意味着什么。数字建立认知，案例建立信任。

**收束页为什么有 next steps？** 报告不是结论，是行动的起点。收束页回顾核心 KPI，给出下一步要做什么（比如 `S1_选购推荐` Hit@5 偏低，建议优化选购推荐类文档的路由策略），让汇报有一个行动收口，而不是看完数字就散会了。

## 分层看板与参考目标

### 1. 三层粒度：从全局到单条

前面说的 `report.md` 和 `slides.html` 都包含分层看板。分层的核心是三层粒度的切片——像地图一样，先看全国，再看省份，再看市区等。

所有指标通过 `_common.py` 的 `slice_mean()` 函数统一聚合，每个指标都输出四个维度：

| 层 | 维度 | 典型用途 |
| --- | --- | --- |
| L1 | overall | 系统整体健康度，一个数字看全局 |
| L2 | by_intent_l2 | 按 22 个二级意图分组，找最差的几个场景，集中优化 |
| L3 | per_sample | 逐条定位具体问题，配合 failures.jsonl 使用 |

`by_intent_l1`（按 SUPPORT / FEEDBACK / CHAT 三个一级意图分组）也有，但实战中 by_intent_l2 更有用——一级意图太粗，二级意图刚好对应一个具体业务场景。

三层粒度的价值在于**逐层下钻**。overall 告诉你系统整体健不健康，by_intent_l2 告诉你哪几个场景拖了后腿，per_sample 告诉你这个场景里具体哪条 query 出了问题。从全局到局部再到单条，一路钻下去，问题就浮出水面了。

### 2. 一页纸看板参考目标

`report.md` 和 `slides.html` 都包含一张核心看板表格，按四个维度组织：

| 维度 | 指标 | 来源 | 参考目标 |
| --- | --- | --- | --- |
| 意图 | Top-1 准确率 | 自建 | >= 92% |
| 检索 | Doc Hit@5 | 自建 | >= 90% |
| 检索 | Recall@5 (must) | 自建 | -- |
| 检索 | context_recall | RAGAS | >= 0.80 |
| 检索 | context_precision | RAGAS | >= 0.75 |
| 生成 | faithfulness | RAGAS | >= 0.90 |
| 生成 | answer_correctness | RAGAS | >= 0.80 |
| 生成 | answer_relevancy | RAGAS | >= 0.85 |
| 行为 | 误拒率 | 自建 | <= 3% |
| 行为 | 错答率（过召回） | 自建 | <= 3% |
| 性能 | 首字 P95 (TTFT) | 自建 | <= 6s |

这些阈值是起步参考值，第一次 baseline 跑完后应该回头校准——实际系统水平可能比参考目标高或低，校准之后才知道哪些指标还有提升空间、哪些已经够好了。

## A/B 对比与持续优化闭环

指标有了，报告也出了。但评测不是跑一次就结束的——每次改动都要重新评测，而且要跟改之前的结果做对比。

### 1. diff：改了什么，看数字说话

改了 prompt、换了模型、调了检索参数之后，怎么知道效果是变好还是变差？

`eval/rag/report/diff.py` 提供两份 `_scores.json` 的 A/B 对比：

```
python -m eval rag diff reports/v1_before/_scores.json reports/v1_after/_scores.json
```

输出每个指标的 before / after / delta，一目了然。Hit@5 从 0.85 升到 0.90，delta +5%，好的。faithfulness 从 0.92 降到 0.88，delta -4%，需要关注。

diff 让每次改动都有数据支撑——不再靠感觉说好像变好了，而是用数字说 Hit@5 提了 5 个点，但 faithfulness 降了 4 个点，需要 trade-off。

### 2. 持续优化闭环

报告不是终点，是新一轮迭代的起点。完整的闭环是这样的：

![图片.png](assets/88-02.png)

这个闭环有三个关键点。

**禁止劣化合入**。任何核心指标退化超过阈值（自建指标看绝对值，RAGAS 看多轮均值差 ≥ 3%），不允许合入主分支。这是评测体系最硬的价值——不是跑个分数看看就完了，而是用数据守住质量底线。你改了一版 prompt，faithfulness 提了 3 个点但 Hit@5 掉了 5 个点？不行，回去继续调。

**扩充评测集**。每一轮评测发现的 bad case，都应该补进评估集。比如这次发现 `S1_选购推荐` 检索效果差，修好之后把这几条样本加到评估集里——下一次有人改了选购推荐相关的代码，这些样本就会自动被检测到，防止回归。评估集是活的，会随着系统迭代越来越完善。

**录制与评分分离的价值**。优化过程中经常需要反复跑 score 和 report，但不需要每次都重跑 runner。改了失败判定阈值想看效果？重跑 report。换了 judge 模型想对比？重跑 score。都不需要再调被评系统的接口——这就是前面说的三段缓存设计带来的效率。

## 评测系列总结

11 篇写完了。回头看一眼，每篇解决了什么问题：

| 篇号 | 主题 | 解决的核心问题 |
| --- | --- | --- |
| 1 | 总览 | 评测项目长什么样？两仓库 × 四流程 × 两套指标 |
| 2 | 评估集 | 150 条评估样本怎么设计？每个字段为什么存在？ |
| 3 | 初始化 | 知识库 / 文档 / 意图树怎么灌进被测系统？ |
| 4 | Runner 链路 | 一条 query 怎么跑出检索证据 + 真实答案 + 性能数据？ |
| 5 | 自建指标 | 意图准确率 + Hit@K / Recall@K / MRR 怎么算？ |
| 6 | 性能指标 | 对话产品的卡顿到底看什么？P95 看首字而非整流 |
| 7 | RAGAS 选型 | 几十个指标为什么只挑 5 个？跟自建指标怎么协同？ |
| 8 | RAGAS 跑起来 | Python 环境 + 依赖安装 + 跑通第一个分数 |
| 9 | RAGAS 全解读 | 5 个指标的算法 + 配对读法 + 优化决策树 |
| 10 | RAGAS 的坑 | 同源偏置、方差、中文 NaN、成本、Judge 误判 |
| 11 | 出报告 | 分层看板 + 失败归因 + PPT 风格 HTML + 持续优化闭环 |

11 篇不是 11 个独立的知识点，而是一条完整的链路。评估集定义了测什么，runner 收集了测到了什么，指标量化了好不好，报告呈现了给谁看，闭环保证了能变好。

从评估集的地基，到 runner 的数据采集，到两套指标的互补评分，到报告的分层呈现，到 A/B 对比驱动迭代——每一个环节都不可或缺。跑通 RAG 链路是第一步，前 18 篇讲的就是这一步。但跑通不代表跑得好。证明它跑得好，靠的不是抽几条 case 看看效果，而是建立一套持续运转的质量保证体系——有评估集兜底覆盖面，有指标量化好坏，有报告驱动行动，有闭环防止劣化。

> Source: https://t.zsxq.com/PrCzD
> Resolved: https://articles.zsxq.com/id_b574n3rq1p9l.html
