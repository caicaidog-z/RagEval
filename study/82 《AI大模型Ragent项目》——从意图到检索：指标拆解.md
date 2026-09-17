# 《AI大模型Ragent项目》——从意图到检索：指标拆解

上一篇讲了 runner 的双接口聚合——一条 query 跑两个接口，SSE 拿答案和 TTFT，JSON 拿检索证据和意图分类，合并成 `EvalRecord` 落进 `runs/*.jsonl`。20 条样本跑完，20 行 JSON 静静躺在那里。

打开 JSONL 随便看一行，里面有 `intent_pred` 等于 `S14_退换货咨询`、`retrieved_doc_ids` 包含 `POLICY_RETURN_002` 和 `FAQ_RET_001`、`reference_doc_ids` 包含 `POLICY_RETURN_002`。这些字段怎么变成一个能量化好坏的分数？

评测体系有两套指标。这篇先讲自建指标里最核心的两组：**意图分类准确率**和**检索质量**。意图是链路最上游的闸门——判错了后面全白干；检索是召回阶段的底线——文档都没找对，答案不可能对。这两组指标有个共同特点：不依赖 LLM，纯集合运算，秒级出结果，每次提交都能跑。

## Part 1：意图分类——错了后面全白干

### 1. 为什么意图是最上游闸门

RAG 系统的处理链路是一条单向管道：意图识别 → 路由 → 检索 → 生成。意图是管道的第一个阀门，错了之后水流方向就偏了，后面每一步都在错误的方向上努力。

拿一个具体例子来说。用户问“买了两周的耳机能退吗”，正确意图是 `S14_退换货咨询`，系统应该去政策库里查退换货规则。但如果意图错判成了 `S2_商品详情`，系统会去商品库里检索耳机的产品参数——频响范围、蓝牙版本、续航时间——然后一本正经地回答一堆跟退货毫无关系的参数信息。

用户看到这个回答只会觉得：这系统是不是傻？

![图片.png](assets/82-01.png)

关键在于：RAGAS 的 5 个指标（`faithfulness` / `answer_relevancy` / `answer_correctness` / `context_precision` / `context_recall`）完全不评意图分类。它们只看给了什么上下文、答了什么内容，不管路由对不对。意图错判但恰好检索到了沾边的内容，RAGAS 可能还会给个不错的分数——但用户体验已经崩了。

所以意图指标必须自建，而且要放在所有指标的最前面看。

### 2. Top-1 Accuracy 怎么算

#### 2.1 先搞懂这个名字

Top-1 Accuracy 拆开看两个词：

- **Top-1**：只看系统给出的第一个（最有信心的）预测结果
- **Accuracy**：准确率，就是猜对了多少

合起来就是：**系统最有信心的那个意图预测，猜对了多少比例**。

为什么要强调 Top-1？因为系统有时候会对一个问题给出多个候选意图（比如复杂问题被拆成几个子问题，每个子问题各识别出一个意图）。Top-1 的意思是只拿排第一的那个来判对错，是最严格的标准。如果将来做 Top-3 Accuracy，就是看前三个预测里有没有包含正确答案——条件更宽松，分数自然更高。

代码里把这个指标命名为 `intent_top1`，和 Top-1 Accuracy 是同一个东西，只不过一个是概念名，一个是代码里的变量名。

#### 2.2 怎么算

算法极其简单：逐条样本对答案，算命中比例。

假设有 10 条测试题，每条事先标注好了正确意图，让系统去核对，核对完逐条对答案：

| 题号 | 正确答案（人工标注） | 系统的预测 | 对了吗 |
| --- | --- | --- | --- |
| 1 | 退换货咨询 | 退换货咨询 | 对 |
| 2 | 商品详情 | 商品详情 | 对 |
| 3 | 退换货咨询 | 保修维修 | 错 |
| 4 | 闲聊 | 闲聊 | 对 |
| 5 | 选购推荐 | 对比选购 | 错 |
| 6 | 商品详情 | 商品详情 | 对 |
| 7 | 退换货咨询 | 退换货咨询 | 对 |
| 8 | 对比选购 | 对比选购 | 对 |
| 9 | 闲聊 | 闲聊 | 对 |
| 10 | 选购推荐 | 选购推荐 | 对 |

对了 8 条，总共 10 条：

```
intent_top1 = 8 / 10 = 80%
```

没有加权，没有部分得分——完全匹配就是对，不匹配就是错。

用公式写就是：

```
intent_top1 = count(intent_pred == intent_l2) / count(有标注的样本)
```

两个字段分别来自 `EvalRecord` 的不同段：

- `intent_pred`（系统预测值）：来自 `/rag/eval` 接口返回的 `intentLeafIds`，经过 `build_record` 取第一个非空值
- `intent_l2`（人工标注值）：从评估集 `eval_set_v1.jsonl` 原样复制的正确答案

对齐方式就是字符串直接比较——`intent_pred` 和 `intent_l2` 都是 `S14_退换货咨询` 就算对，不一致就算错。

#### 2.3 哪些样本参与计算

所有有 `intent_l2` 标注的样本都参与计算，不限 `requires_rag`。闲聊类（CHAT）和反馈类（FEEDBACK）样本虽然不走 RAG 检索，但它们同样需要正确的意图分类来路由——闲聊判成知识检索，系统会无意义地去查知识库，白白浪费算力和用户等待时间。

#### 2.4 多子问题的取值

Ragent 处理复杂 query 时会做子问题拆分，每个子问题独立跑意图识别。`EvalRecord` 里的 `intent_pred_all` 记录了所有子问题的意图列表，`intent_pred` 取的是第一个非空值：

```
intent_pred = next((c for c in intent_codes if c), None)
```

绝大多数情况下只有一个子问题，`intent_pred_all` 长度为 1，`intent_pred` 就等于列表里唯一的那个值。多子问题的场景极少，目前评估集里的 query 都是单轮单意图的，不需要过多纠结这个细节。

### 3. 分层聚合：找最差的几个意图定向优化

总体准确率如果是 85%，意味着 20 条里有 3 条判错了。但这 3 条错在哪？是均匀分散在各个意图里，还是集中在某一两个意图上？

答案几乎总是后者——错分集中在少数几个容易混淆的意图里。所以指标不只看总体数字，还按二级意图（`intent_l2`）切片，每个意图单独算准确率。

拿到分层结果后，按准确率从低到高排序，直接找到最差的几个：

| 二级意图 | 准确率 | 命中 / 总数 | 典型错分方向 |
| --- | --- | --- | --- |
| S3_对比选购 | 50.0% | 1 / 2 | 错分到 S1_选购推荐 |
| S14_退换货咨询 | 66.7% | 2 / 3 | 错分到 S15_保修维修 |

> 以上数字仅为示意，实际结果取决于 ragent 的意图模型效果。

20 条样本量不大，单个意图可能只有两三条，一条判错准确率就掉到 50%。数字本身的置信度有限，但错分方向是有价值的——它告诉你哪两个意图之间的边界最模糊。后续扩充评估集时，优先补这些易混淆意图的样本，让数字更可靠。

两种典型错分模式反复出现：

**跨意图相近词**。`S1_选购推荐` 和 `S3_对比选购` 都跟购买决策相关，“3000 元手机推荐”容易在两者之间漂移——S1 是单品推荐，S3 是多品对比，边界本身就模糊。

**上下文歧义**。“能退吗”三个字脱离上下文，无法判断是退换货（S14）还是会员退订（S16）。评估集里的 query 是单轮的，这类歧义只能靠 query 本身的措辞来区分。

实操上的优化手段很直接：对最差的几个意图，调整意图树节点的描述文案让区分度更大，或者在意图分类 Prompt 里补 few-shot 示例，明确告诉模型这两类的区别。改完之后重跑一次评测，看这几个意图的准确率有没有上来——这就是评测驱动优化的基本循环。

### 4. 代码实现

`intent.py` 的核心实现只有十几行：

```
def compute(records: list[EvalRecord]) -> MetricResult:
    """意图分类 Top-1 准确率。"""

    def _score(r: EvalRecord) -> float | None:
        if not r.intent_l2:
            return None  # 没有标注的样本不参与计算
        return 1.0 if r.intent_pred == r.intent_l2 else 0.0

    overall, by_l1, by_l2, per_sample = slice_mean(records, _score)
    return MetricResult(
        name="intent_top1",
        overall=overall,
        by_intent_l1=by_l1,
        by_intent_l2=by_l2,
        per_sample=per_sample,
        is_pct=True,
    )
```

逻辑一目了然：逐样本判 `intent_pred == intent_l2`，命中给 1.0，不命中给 0.0，`intent_l2` 为空返回 `None` 表示跳过。然后交给 `slice_mean` 做聚合。

`slice_mean` 是所有指标共用的分层聚合工具，接受一个评分函数和一个可选的过滤函数，返回四元组：总体均值、按一级意图分组的均值、按二级意图分组的均值、逐样本明细字典。意图指标和下面要讲的检索指标都用它来聚合，一套逻辑，两组指标。

## Part 2：检索指标——召回了多少，排得好不好

### 1. 为什么用文档级而不是 chunk 级

评估集标注的 `expected_doc_ids` 是文档维度的，比如 `FAQ_VAC_001`、`POLICY_RETURN_002`——不是 chunk 维度的。原因很现实：chunk 级标注不稳定。你改了 chunk size 从 512 到 1024，或者调了 overlap 比例，同一段文本的 chunk ID 就变了，之前标注的 `expected_chunk_ids` 全部作废。

文档级标注跟分块策略解耦，换 chunk size 照样能用。所以检索指标统一在 doc 级别计算：拿 runner 返回的 `retrieved_doc_ids`（已去重的业务码列表）跟评估集的 `expected_doc_ids` 做集合运算。

### 2. 先过滤：不是所有样本都该算

20 条评估样本里，有 CHAT 类（你好、谢谢）和部分 FEEDBACK 类样本，它们的 `requires_rag` 是 `false`，`expected_doc_ids` 是空的——这些 query 本来就不该走 RAG 检索。如果把它们算进检索指标，每条的 Hit@5 都是 0.0，会无意义地拉低整体数字，制造假信号。

过滤逻辑封装在 `is_retrieval_eligible` 函数里：

```
def is_retrieval_eligible(record: EvalRecord, *, inclusive: bool = False) -> bool:
    """判断样本是否参与检索指标计算。"""
    if not record.requires_rag:
        return False
    if inclusive:
        return bool(record.reference_doc_ids or record.reference_doc_ids_nice)
    return bool(record.reference_doc_ids)
```

默认模式：`requires_rag=true` 且 `reference_doc_ids` 非空——这条 query 应该走 RAG，而且标注了必须命中的文档，才参与检索指标。包容模式多接受 `reference_doc_ids_nice` 不为空的样本，用于计算宽松口径的 Recall。

> 将来 Ragent 升级到 v2 版本后，可能会出现仅通过 Agent 循环获取答案，不走 RAG，所以 requires_rag 也会等于 false。

20 条里大约 15~17 条满足默认过滤条件，这些才是检索指标的分母。

### 3. 三个指标一次讲清

用一个具体例子串起来。假设某条 query：

- 标注的相关文档集：`{D3, D7, D9}`（3 篇）
- ragent 召回的 Top-5 文档列表：`[D1, D3, D5, D7, D8]`

#### 3.1 Hit@K：至少找到一篇了没

```
Hit@K = 1.0  如果 Top-K 里有至少 1 篇相关文档
       0.0  否则
```

上例中 Top-5 里有 D3 和 D7 两篇命中了，所以 `Hit@5 = 1.0`。

Hit@K 是最宽松的指标，只问一个问题：检索找到了任何一点有用的东西没有？哪怕 5 篇里只有 1 篇沾边，也算过。它衡量的是检索的底线——连一篇相关文档都找不到，后面就没得玩了。

#### 3.2 Recall@K：找全了没

```
Recall@K = |Top-K ∩ 相关集| / |相关集|
```

上例中 Top-5 命中了 D3、D7，漏了 D9。相关集共 3 篇，命中 2 篇：`Recall@5 = 2/3 ≈ 0.67`。

Recall@K 关注的是覆盖完整性。Hit@K 只问有没有，Recall@K 接着问：找全了没有？上例中虽然找到了 2 篇，但 D9 漏了——如果 D9 里有一条关键的退货时限说明，模型生成的答案就会缺一块。

#### 3.3 MRR@10：排在第几位

```
MRR@10 = 1 / 第一篇相关文档的排名（1-indexed）
       = 0.0  如果 Top-10 里一篇相关都没有
```

上例中 Top-5 列表 `[D1, D3, D5, D7, D8]`，第一篇相关文档是 D3，排在第 2 位：`MRR@10 = 1/2 = 0.5`。

如果 D3 排在第 1 位，MRR 就是 1.0；排在第 5 位就是 0.2。MRR 衡量的是排序质量——相关文档越靠前，模型越容易用到它（上下文窗口有限，排在后面的 chunk 容易被截断或被 Lost in the Middle 效应稀释）。

三个指标各管一件事：

| 指标 | 回答的问题 | 值域 | 特点 |
| --- | --- | --- | --- |
| Hit@K | 至少找到一篇相关的了没有？ | 0 或 1 | 最宽松，底线指标 |
| Recall@K | 相关的都找全了没有？ | 0~1 连续值 | 关注覆盖完整性 |
| MRR@10 | 第一篇相关的排第几？ | 0~1 连续值 | 关注排序质量 |

### 4. K 怎么选，数字怎么读

**Hit@5 是主指标**，参考目标 ≥ 90%。5 是 ragent 默认的 Top-K 召回数量，跟生产配置一致。

同时计算 K = 1、3、5、10 四档，对比着看能发现更多信息：

- **Hit@1 高**：最相关的文档经常排在第一位，检索精度很好
- **Hit@5 高但 Hit@1 低**：相关文档被找到了，但没排到前面去——rerank 有优化空间，或者向量相似度的区分度不够
- **Hit@10 比 Hit@5 高很多**：说明有不少样本的相关文档排在第 6~10 位，放宽 Top-K 到 10 能救回来——可以考虑调大生产环境的 K 值，或者优化检索策略

一个典型的健康分布长这样：`Hit@1 = 68%`，`Hit@3 = 84%`，`Hit@5 = 91%`，`Hit@10 = 95%`——从 1 到 10 逐步收敛，5 已经覆盖了绝大多数。如果 `Hit@5 = 75%` 但 `Hit@10 = 93%`，说明大量相关文档卡在 6~10 名，K=5 在不同场景下可能会比较激进。

> 如果通过意图识别树匹配意图，这个 topk 能够自定义，相对来说好的多。

### 5. must / nice 两档 Recall

前面讲 Recall@K 的时候，公式里有一个相关集——就是标注的期望文档。但实际标注时会碰到一个问题：有些文档是必须找到的，有些文档是锦上添花的，两者重要程度不一样。

拿一个具体例子来说。用户问“买了两周的耳机能退吗”，标注员标了三篇相关文档：

- `POLICY_RETURN_002`（退换货政策）——必须找到，没有它就回答不了退货条件和时限
- `FAQ_RET_001`（退货常见问题）——必须找到，里面有具体的退货流程步骤
- `GUIDE_AFTERSALE_005`（售后服务总览）——有则更好，里面提到了退货入口在哪，但缺了它也能基本回答问题

前两篇是核心证据，缺了答案就不完整；第三篇是补充信息，有了更好，没有也不影响回答质量。

评估集用两个字段来区分这两类：

- `expected_doc_ids`（must，必须命中）：前两篇，`[POLICY_RETURN_002, FAQ_RET_001]`
- `expected_doc_ids_nice`（nice，锦上添花）：第三篇，`[GUIDE_AFTERSALE_005]`

对应两个口径的 Recall，在报告和代码里分别叫 `recall@K` 和 `recall_inclusive@K`：

`recall@K`（must 口径）——严格口径，只拿 must 文档算。分母是 2（两篇必须命中的），如果系统 Top-5 里找到了 `POLICY_RETURN_002` 但漏了 `FAQ_RET_001`，Recall = 1/2 = 50%。这个数字直接反映核心证据的召回情况。

`recall_inclusive@K`（inclusive 口径）——宽松口径，把 must 和 nice 合并算。分母变成 3（两篇 must + 一篇 nice），同样的召回结果 Recall = 1/3 = 33%。分母变大了，数字自然更低，但它能告诉你补充信息的召回情况。报告里显示为 `Recall@5 (含 nice)`。

两个口径配合着看：如果 must Recall 高但 inclusive Recall 低，说明核心文档都找到了，只是补充信息没捞到——问题不大。如果 must Recall 本身就低，那是核心证据都没找全，需要优先解决。

> 当前评估集的实际情况：20 条中只有 S1-01 到 S1-05 这 5 条做了 must / nice 拆分标注，其余 15 条把所有期望文档都放在 `expected_doc_ids` 里（等于全部算 must）。所以目前两个口径的数值差异很小。随着后续标注规范升级，差异会逐步显现。

### 6. 代码实现

检索指标对 K = 1、3、5、10 四档统一计算。入口函数一目了然：

```
K_VALUES = [1, 3, 5, 10]

def compute(records: list[EvalRecord]) -> list[MetricResult]:
    results = []
    for k in K_VALUES:
        results.append(_hit_at_k(records, k))
        results.append(_recall_at_k(records, k))
        results.append(_recall_inclusive_at_k(records, k))
    results.append(_mrr(records, max_k=10))
    return results
```

Hit@K 的核心逻辑：

```
def _hit_at_k(records: list[EvalRecord], k: int) -> MetricResult:
    def _score(r: EvalRecord) -> float | None:
        topk = (r.retrieved_doc_ids or [])[:k]
        gold = set(r.reference_doc_ids or [])
        if not gold:
            return None
        return 1.0 if (set(topk) & gold) else 0.0

    overall, by_l1, by_l2, per_sample = slice_mean(
        records, _score, eligible=is_retrieval_eligible
    )
    return MetricResult(
        name=f"hit@{k}", overall=overall,
        by_intent_l1=by_l1, by_intent_l2=by_l2,
        per_sample=per_sample, is_pct=True,
    )
```

取 `retrieved_doc_ids` 的前 K 个，跟 `reference_doc_ids` 做集合交集——有交集就是 1.0，没有就是 0.0。通过 `eligible=is_retrieval_eligible` 参数，只有 `requires_rag=true` 且有标注文档的样本才参与计算。

Recall@K 类似，分子是交集大小，分母是相关集大小：

```
def _recall_at_k(records: list[EvalRecord], k: int) -> MetricResult:
    def _score(r: EvalRecord) -> float | None:
        topk = set((r.retrieved_doc_ids or [])[:k])
        gold = set(r.reference_doc_ids or [])
        if not gold:
            return None
        return len(topk & gold) / len(gold)

    overall, by_l1, by_l2, per_sample = slice_mean(
        records, _score, eligible=is_retrieval_eligible
    )
    return MetricResult(
        name=f"recall@{k}", overall=overall,
        by_intent_l1=by_l1, by_intent_l2=by_l2,
        per_sample=per_sample, is_pct=True,
    )
```

MRR 只算一个 `max_k=10`，逻辑是遍历 Top-K 列表，找到第一个命中的相关文档，返回排名的倒数：

```
def _mrr(records: list[EvalRecord], max_k: int = 10) -> MetricResult:
    def _score(r: EvalRecord) -> float | None:
        topk = (r.retrieved_doc_ids or [])[:max_k]
        gold = set(r.reference_doc_ids or [])
        if not gold:
            return None
        for rank, doc_id in enumerate(topk, start=1):
            if doc_id in gold:
                return 1.0 / rank
        return 0.0

    overall, by_l1, by_l2, per_sample = slice_mean(
        records, _score, eligible=is_retrieval_eligible
    )
    return MetricResult(
        name=f"mrr@{max_k}", overall=overall,
        by_intent_l1=by_l1, by_intent_l2=by_l2,
        per_sample=per_sample,
    )
```

所有指标都通过 `slice_mean` 做分层聚合，跟意图指标共享同一套逻辑。传入不同的 `_score` 函数和 `eligible` 过滤条件，就能算出不同指标的总体值和分层切片——不需要每个指标自己写一遍分桶逻辑。

跑一次 score 的命令也很简单：

```
# 对最新一次 run 计算自建指标（跳过 RAGAS，秒级出结果）
python -m eval rag score --skip-ragas

# 一键跑全流程
python -m eval rag all --limit 20 --skip-ragas
```

输出里能直接看到各项指标和分层排名，最差的几个意图一目了然。

## 跟 RAGAS 的一次对照

自建的 Recall@K 和 RAGAS 的 `context_recall` 都叫召回，但衡量的东西完全不同：

- **自建 Recall@K**：纯集合运算，`|Top-K ∩ gold_docs| / |gold_docs|`，看的是文档级命中
- **RAGAS context_recall**：把标准答案（`reference`）拆成一条条原子陈述（atomic statement），让 LLM judge 判断每条陈述能否由 `retrieved_contexts` 推出来，`covered / total`

两个指标交叉对照非常有价值：

| 场景 | 自建 Doc Hit@5 | RAGAS context_recall | 说明 |
| --- | --- | --- | --- |
| 文档找到了但 chunk 切错了 | 高 | 低 | 正确文档被召回了，但相关内容不在这些 chunk 里——分块策略需要调整 |
| 文档没精确命中但碰巧有相关信息 | 低 | 还行 | 别的文档的 chunk 碰巧覆盖了部分信息——检索没对齐，答案靠运气 |
| 检索和内容都到位 | 高 | 高 | 理想状态 |
| 全面崩溃 | 低 | 低 | 检索策略需要大改 |

自建指标的优势在于**快和确定**：秒级出结果，跑 100 次结果一样，适合挂 CI 当闸门。RAGAS 的优势在于**语义级**：能发现文档找到了但 chunk 里没有关键信息这类自建指标看不出来的问题。两套配合着看，检索阶段的问题基本能够发现。

## 小结与下一篇预告

意图 Top-1 是最上游的闸门——判错了路由就错，后面全盘皆输。Hit@K / Recall@K / MRR 是检索阶段的三把尺子，分别量有没有找到、找全了没有、排得好不好。

这些指标全部是纯集合运算，不调 LLM，秒级出结果。每次改了意图描述、换了 Embedding 模型、调了 rerank 参数，跑一遍就知道效果是变好还是变差——这正是自建指标存在的意义。

自建指标还剩一组——性能指标。对话产品的卡顿到底是什么？P95 应该看首字延迟（TTFT）还是整流耗时？均值和 P95 的差距说明什么？

> Source: https://t.zsxq.com/txWHU
> Resolved: https://articles.zsxq.com/id_d8nox622770f.html
