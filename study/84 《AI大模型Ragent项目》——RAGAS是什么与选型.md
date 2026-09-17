# 《AI大模型Ragent项目》——RAGAS是什么与选型

上一篇收束了自建指标的全部四组——意图、检索、性能、行为，覆盖了路由对不对、文档找到没、用户等多久、系统行为合不合理。这四组指标有一个共同特点：全是集合运算或数值运算，跑一遍不到 1 秒，可以挂 CI 每次提交都跑。

但它们也有一个共同的盲区——评不了内容。

假设某条 query 的自建指标全绿：Hit@5 = 1.0，Recall@5 = 1.0，`intent_top1` 正确，TTFT 也在阈值内。打开实际回答一看：模型在 chunk 信息的基础上多编了一段话。chunk 里只写了保修 1 年，模型回答保修 1 年且 AppleCare+ 可延保至 3 年——后半句是编的。自建指标对这种幻觉完全无感。检索确实命中了正确文档，意图确实分对了，响应确实够快——但答案是错的。

这就是需要 RAGAS 的理由。

## 自建指标的盲区：评不了内容

### 1. 三种自建指标看不出的问题

用比特严选的场景举三个例子：

**幻觉（编造细节）**

用户问“AirPods Pro 2 的保修期是多久”，检索到的 chunk 里只写了“标准保修期 1 年”。模型回答：“AirPods Pro 2 标准保修 1 年。如果购买了 AppleCare+ 服务，可延保至 3 年，并享受意外损坏维修服务。”——后面两句是模型从预训练知识里补的，不在 chunk 里。

自建指标：Hit@5 = 1.0，chunk 命中了。看不出答案里有编造。

**跑题（答非所问）**

用户问“退货怎么寄回去”，检索到的 chunk 确实是退货退款政策文档。模型回答：“退款通常在 35 个工作日到账，如果使用信用卡支付可能需要 710 天。”——回答了退款时间，但用户问的是退货流程（怎么寄回去、走什么渠道）。

自建指标：Hit@5 = 1.0，文档是对的。`intent_top1` 也对，确实是售后意图。但答案跑偏了。

**遗漏（关键信息缺失）**

用户问“哪些情况不能退货”，chunk 里列了三个条件：已拆封的耳机、超过 7 天、无购买凭证。模型只回答了前两个，漏了第三个。

自建指标：Recall@5 = 1.0，所有相关文档都召回了。但答案不完整。

三种情况的共同点：**自建指标全绿，答案有问题**。需要一个能读懂文本、判断内容质量的工具。

### 2. 需要一个能读懂文本的评估工具

要检测幻觉、跑题、遗漏，本质上需要做语义级的判断——这段话能不能从上下文推出来？回答切不切题？标准答案里的信息有没有被覆盖？

用正则匹配太脆弱（同义词换个说法就失效），人工逐条审太慢（150 条 × 每次迭代都看一遍不现实）。可行的方案是：用另一个大模型来当裁判，系统化地拆解答案、逐个判断。

这就是 LLM-as-judge 的思路，也是 RAGAS 的核心。

## RAGAS：用 LLM 当裁判的评估框架

### 1. 一句话定义

RAGAS（Retrieval Augmented Generation Assessment）是一个开源的 RAG 评估框架，GitHub 仓库 [vibrantlabsai/ragas](https://github.com/vibrantlabsai/ragas)。它不是简单让 judge 模型给出好、中、差的整体评分，而是把 RAG 的不同环节拆成更细粒度的判断：有的指标会把回答拆成原子级声明（atomic claims）或陈述（statements），逐条判断是否被上下文或标准答案支持；有的指标会反向生成问题，再计算与原问题的语义相似度；有的指标会逐个判断 retrieved contexts 是否和 reference 相关。

比如评忠实度：先把回答拆成几个独立的事实声明，再让 judge 逐一判断每个声明能否从检索到的上下文推导出来。判完之后算比例——10 个 claim 里 8 个有据可查，忠实度 = 0.8。比整体打分更可靠、更可解释、也更容易定位问题出在哪。

### 2. 为什么不能全跑：成本账

RAGAS 0.4.3 提供了不少可选指标和变体，覆盖 RAG、文本对比、Agent / Tool、SQL、Rubric 等场景。全跑？先算一笔账。

RAGAS 里的 LLM-based 指标通常不是一次模型调用就结束。它可能会做 claim 拆解、逐条判断、反向生成问题、计算 embedding 相似度、逐个 context 判相关性等步骤。不同指标、不同样本长度、不同 `strictness`、不同 contexts 数量，都会影响调用量和成本。

粗算本项目的开销：

| 维度 | 数值 |
| --- | --- |
| 选定指标数 | 5 个 |
| 每条样本 × 5 指标的 judge / embedding 步骤 | 约十几次量级 |
| 评估集规模 | 20 条 |
| 单轮评测总步骤 | 约 200+ 次量级 |
| 压制方差跑 3 轮（--ragas-n 3） | 约 600+ 次量级 |
| 时间 | 十几分钟~半小时，取决于并发、限流和上下文长度 |
| 费用 | 取决于 judge 模型、embedding 模型、输入输出 token 和重试次数 |

5 个指标已经是一个不小的离线评测任务。如果跑 10 个指标，成本和耗时大概率翻倍，评测报告也会变得更难读。更要命的是：指标越多，信号越分散。10 个指标里有 3 个低了，你不知道该先优化哪个——选型的核心不是越多越好，而是每个指标都能指向一个明确的优化动作。

## 选 5 个的判断标准

三个硬约束：

**跑得起**——20 条 × 3 轮的成本可控，开发阶段改一版 prompt 能当天看到结果，不需要长时间高费用验证。

**看得懂**——每个指标的含义一句话能解释清楚。拿给不做评测的同事看报告，不需要翻长篇大论才能理解这个 0.7 大概意味着什么。

**能驱动迭代**——指标低了能定位到具体的优化方向。忠实度低 → 可能是模型编造或上下文不足；上下文召回低 → 可能是检索策略、chunk 切分或知识库覆盖有问题；答案相关性低 → 回答跑题了，继续查意图、query rewrite 或 prompt 聚焦。每个指标对应一组优先排查方向，而不是笼统地说“需要改进”。

一句话概括选型哲学：不追求全覆盖，追求每个指标都能指向一个明确的优化动作。

## 5 个指标的分组与诊断闭环

### 1. 生成层：3 个指标评答案质量

| 报告字段 | v0.4.3 实现口径 | 中文含义 | 比较对象 | 一句话定义 |
| --- | --- | --- | --- | --- |
| faithfulness | Faithfulness | 忠实度 | response vs retrieved_contexts | 回答里的 claim 能不能从检索上下文推出来——幻觉检测 |
| answer_relevancy | AnswerRelevancy | 答案相关性 | response vs user_input | 回答是否切题，有没有跑偏、遗漏焦点或加入不必要内容 |
| answer_correctness | AnswerCorrectness | 答案正确性 | response vs reference | 回答与标准答案的事实和语义一致程度——端到端答案质量信号 |

三个指标从不同角度评答案：

- `faithfulness` 管的是有没有编——回答里的事实声明应该能被 retrieved contexts 支撑
- `answer_relevancy` 管的是切不切题——回答应该直接回应用户问的问题，少跑偏、少啰嗦、少答非所问
- `answer_correctness` 管的是对不对——回答应该和 reference 在事实层面、语义层面尽量一致

> 注意：`answer_relevancy` 只看切题，不直接判断事实正确性。一个答案可以切题但错误（比如用户问保修期，模型回答了保修期但数字编的）。它必须和 `faithfulness`、`answer_correctness` 配套看，单看会误判。

> 也要注意：`answer_correctness` 不是传统机器学习里的严格“正确率”。它是基于 reference 的自动化评分，适合作为端到端答案质量信号，但仍然依赖 reference 质量、judge 模型、prompt、embedding 和样本难度。

### 2. 检索层：2 个指标评上下文质量

| 报告字段 | v0.4.3 实现口径 | 中文含义 | 比较对象 | 一句话定义 |
| --- | --- | --- | --- | --- |
| context_precision | ContextPrecision | 上下文精度 | retrieved_contexts vs reference | retrieved contexts 里与 reference 相关的 chunk 是否排得靠前，噪声是否过多 |
| context_recall | ContextRecall | 上下文召回 | retrieved_contexts vs reference | reference 需要的信息，retrieved contexts 覆盖了多少 |

这两个跟自建的 Hit@K / Recall@K 有什么区别？

- **自建 Hit@K / Recall@K**：文档 ID 级的集合运算。只看这个文档 ID 有没有出现在召回列表里。
- RAGAS `context_precision` / `context_recall`：语义级判断。看 retrieved contexts 里的内容是否和 reference 相关、reference 里的信息点是否被 contexts 覆盖。

区别在于：自建指标说文档 D3 召回了，RAGAS 说文档 D3 里的关键信息确实支撑了正确答案。前者是 ID 匹配，后者是语义匹配。有时候文档召回了但切的 chunk 不对，Hit@K = 1.0 但 `context_recall` 很低——这就是 chunk 切分的问题，自建指标看不出来。

### 3. 诊断闭环：一张图串起 5 个指标

5 个指标不是独立的，它们形成了一条诊断链路。答案出了问题，按指标分支往下查：

![图片.png](assets/84-01.png)

这张图的实际用法：跑完 RAGAS 发现某条样本 `answer_correctness` 低，按图往下走——先看 `faithfulness`，如果低，优先怀疑回答中有 context 不支撑的内容；如果 `faithfulness` 正常但 `context_recall` 低，说明检索阶段可能没找到 reference 需要的信息，问题出在上游。每个分支对应一组优先排查方向，不用完全靠猜。

但这张图不是机械的 if-else。真实诊断要看指标组合。比如：

| 指标组合 | 更可能的问题 |
| --- | --- |
| answer_correctness 低 + faithfulness 低 | 回答中有幻觉，或 retrieved contexts 无法支撑回答 |
| answer_correctness 低 + context_recall 低 | 检索漏召、chunk 切分不合适、知识库缺信息 |
| answer_relevancy 低 + 自建意图指标异常 | 意图识别或路由错误 |
| answer_relevancy 低 + 自建意图指标正常 | Prompt 没有聚焦问题，或 query rewrite 把问题改偏了 |
| context_precision 低 + context_recall 高 | 召回到了相关信息，但噪声多、排序差，适合优化 rerank / TopK |
| faithfulness 高 + answer_correctness 低 | 回答忠于 context，但 context 可能过期、reference 更完整，或知识库本身有问题 |
| 5 个指标都正常但人工仍觉得错 | 检查 reference、业务规则、评估维度是否遗漏，或 judge 是否误判 |

自动指标负责缩小排查范围，不负责替代工程判断。

## 为什么不选其他指标

RAGAS 0.4.3 还有很多指标，逐个说明为什么当前不选：

`answer_similarity` / `semantic_similarity`（文本相似）

只衡量回答和标准答案像不像，不判断事实对不对。比如模型回答“保修期 2 年”，标准答案“保修期 1 年”，两句话结构几乎一样，语义相似度会很高——但事实是错的。当前已经选了 `answer_correctness`，它内部包含语义相似度分量，再单独跑一个纯相似度指标是重复计算，还容易误导。

`context_entity_recall`（检索）

它的逻辑是：从标准答案里抽出所有实体（品牌、型号、人名等），再看召回的 chunk 里覆盖了几个。比如标准答案提到了 AirPods Pro 2、AppleCare+、比特严选三个实体，召回的 chunk 里出现了两个，实体召回率就是 2/3。

问题在于：本项目知识库是同一批商品的描述和政策，几乎每篇文档都反复出现 iPhone、AirPods、比特严选这些实体。随便召回几个 chunk，实体命中率就能到 0.9 以上——不管内容对不对，实体总是在的。区分度太差，加了等于没加。

`noise_sensitivity`（检索鲁棒性）

它测的是：如果召回的 chunk 里混进了不相关的噪声文档，答案质量会掉多少。比如用户问 AirPods 保修期，正常召回 5 个 chunk 里 4 个相关、1 个不相关，答案没问题。现在故意把不相关的 chunk 从 1 个加到 3 个，看模型会不会被干扰——如果答案质量明显下降，说明系统对噪声敏感。

这个指标本身有价值，但需要额外构造噪声上下文再跑一轮评测，调用量和分析复杂度都会上升。20 条评估集规模下暂时不值得，等扩到 500 条或专门做鲁棒性评测再考虑。

`multi_turn_*` / 多轮相关指标（多轮）

评估集是单轮设计（每条 query 独立），会话记忆由 Ragent 内部管理，评测侧不模拟多轮。后续有多轮评估集再开。

`aspect_critic` / rubrics 类自定义指标（自定义）

前面 5 个指标都是 RAGAS 内置的，评分逻辑和 prompt 框架已经写好，直接用就行。`aspect_critic` 不一样——它是一个空壳，你得自己定义评什么、怎么算分。比如你想评“回答语气是否专业”，就得自己写一段 prompt 告诉 judge 什么算专业、什么算不专业、边界情况怎么判。再比如评“回答是否包含免责声明”，又得写另一段 prompt。

问题在于：每加一个自定义维度，就多一份 prompt 要写、要调、要校准。不同 judge 模型对同一段 prompt 的理解可能不一样，换个模型分数就漂移了。当前 5 个核心指标已经覆盖了主要问题，先不把维护成本拉高。

`summarization_score` / `summary_score`（摘要）

摘要场景专用，本项目当前是问答场景。

**Agent / Tool 指标（工具调用）**

本文评的是单轮电商 RAG 问答，不评 agent tool chain。工具调用准确率、Tool F1、Agent Goal Accuracy 后续可以给 agent 场景单独开一篇。

一句话收束：不是这些指标不好，是本项目当前评估集规模小、单轮 query、电商问答场景下用不上。5 个核心指标覆盖了幻觉、切题、正确、检索精度、检索覆盖五个维度，对当前阶段来说信息量已经足够密。评估集扩大或场景变复杂后再加。

## 自建指标与 RAGAS 的协同

### 1. 互补关系

两套指标不是二选一，是互补：

| 维度 | 自建指标 | RAGAS |
| --- | --- | --- |
| 速度 | 秒级，纯内存计算 | 十几分钟~半小时，取决于模型、并发和限流 |
| 成本 | 零（无 API 调用） | 有 judge LLM / embedding 调用成本 |
| 可重复性 | 确定性，跑 100 次结果一样 | 有方差，同一批数据重复跑可能有轻微波动 |
| 覆盖面 | 意图 / ID 命中 / 性能 / 行为 | 生成忠实度 / 答案相关性 / 答案正确性 / 上下文语义质量 |
| CI 友好 | 适合，每次提交都跑 | 适合小样本 smoke test；完整集更适合离线或 release gate |
| 能力边界 | 算不了内容质量 | 不适合替代低成本、确定性的业务指标和性能指标 |

### 2. 使用场景分工

**自建指标当 CI 闸门**——每次提交自动跑。检索指标低于阈值直接拦住不让合并。秒出结果，零成本，开发者能立即看到反馈。

**RAGAS 当离线深度评估**——改完 prompt、换完 embedding 模型、调完检索参数之后手动触发。等十几分钟拿到语义级的评分，看生成质量有没有退化。

**RAGAS 小样本可以进 CI**——不是说 RAGAS 完全不能用于 CI，而是不建议每次提交都跑 20 条 × 3 轮的完整评测。更合理的分层是：

| 场景 | 建议跑法 |
| --- | --- |
| 每次 PR | 自建指标全量 + RAGAS 小样本 smoke test |
| 每晚定时 | RAGAS 20 条完整评测 |
| Release 前 | RAGAS 20 条 × 3 轮均值 + 高风险样本人工抽查 |
| Prompt / embedding / rerank 大改 | 强制完整 RAGAS 评测 |

这个分工反映在命令行设计上：

```
# CI 用：快速跑评测 + 自建指标 + 报告
python -m eval rag all --skip-ragas

# 手动触发：完整评测含 RAGAS，十几分钟
python -m eval rag all

# 压制 RAGAS judge 方差：RAGAS 独立跑 3 次取均值
python -m eval rag all --ragas-n 3
```

设计理念是录制与评分分离——跑一次 runner 落 `runs/*.jsonl`，后续可以反复 score，不重复调被评接口。加 RAGAS 只是在 score 阶段多跑一步，不影响前面的自建指标。即使 RAGAS 因为 API key 没配、网络超时或限流失败了，自建指标照样落盘。

```
# eval/rag/pipeline/score.py
print("[1/4] intent ...")
results += intent.compute(records)
print("[2/4] retrieval ...")
results += retrieval.compute(records)
print("[3/4] behavior ...")
results += behavior.compute(records)
print("[4/4] latency ...")
results += latency.compute(records)

if not skip_ragas:
    print("[5/5] ragas (LLM-as-judge) ...")
    results += ragas_judge.compute(records, limit=ragas_limit, n_runs=ragas_n)
```

### 3. 最低验收线：先给指标一个解释框架

指标能驱动迭代的前提，是团队知道“多少算低”。RAGAS 的绝对分数不能脱离业务和评估集解释，但可以先给一组工程上的参考线：

| 指标 | 参考解释 |
| --- | --- |
| faithfulness | 低于 0.8 优先查幻觉；高风险售后、价格、保修类样本应更严格 |
| answer_relevancy | 低于 0.75 优先查是否答非所问、漏答核心问题或过度展开 |
| answer_correctness | 低于 0.75 说明端到端答案质量需要人工看样本，不应只看均值 |
| context_precision | 低说明 contexts 噪声多或排序差，优先查 TopK / rerank / metadata filter |
| context_recall | 低说明 reference 需要的信息没被召回，优先查 chunk、召回策略、知识库覆盖 |

> 注意，不同的业务场景对于分数是不同的。严格意义上来说，越是在规章制度、金额等场景要求越是严格。反之，即使分数稍低，可能也只是语义上描述不同，并不影响系统或者用户。

更稳的发布门禁不是只看均值，而是看均值 + 分位数 + 高风险 fail case：

- 均值看整体趋势；
- P10 / 最差 10% 看长尾风险；
- 高风险意图（退货、保修、价格、售后承诺）单独设更严格阈值；
- 关键样本低分必须人工复核。

这组阈值不是一次定死的。第一版可以用来发现问题，后续用人工标注和历史回归结果校准。

### 4. LLM-as-judge 的边界

RAGAS 分数不是事实真理。它依赖 judge 模型、prompt、reference、embedding、claim 拆解质量和语言适配。中文场景还可能出现表达方式差异、实体别名、格式化答案、枚举项顺序等问题，导致分数波动或误判。

所以本项目才设计了 `--ragas-n 3`：同一批数据跑多轮，取均值压制随机性。同时，关键发布决策不能只看自动分数，还要保留高风险样本的人审抽查。

第 10 篇会专门讲这些实操坑：单次方差、同源偏置、中文 NaN、JSON 解析失败、reference 质量、judge 模型选择，以及如何把 RAGAS 分数和人工标注对齐。

## 小结与下一篇预告

自建指标评不了内容，RAGAS 补上了这块语义盲区。5 个指标 = 3 个看生成（幻觉 / 切题 / 正确）+ 2 个看检索（精度 / 覆盖），形成从答案不对到定位原因的完整诊断闭环。跟自建指标的分工是 CI 闸门 vs 离线深度评估——两者互补，不替代。

本文的版本口径已经对齐到 `ragas==0.4.3`：报告字段继续保持可读的 `snake_case`，实现层使用 v0.4 的 collections-based API；评测结果不再作为绝对准确数据，而是作为自动化语义信号，配合自建指标和人审抽样一起使用。

选型确定了，但 RAGAS 还没跑起来。Python 3.11 是本项目的环境约束还是 RAGAS 的硬性要求？三个核心依赖怎么装？judge 模型怎么配？0.4.3 的 `llm_factory`、`embedding_factory`、`MetricResult.value` 怎么接进现有 runner？跑通第一个分数需要准备什么？

> Source: https://t.zsxq.com/nUK3K
> Resolved: https://articles.zsxq.com/id_unr55u7hqgzc.html
