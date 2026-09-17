# 《AI大模型Ragent项目》——性能指标的口径选择

上一篇讲完了意图准确率和检索指标——意图 Top-1 衡量路由对不对，Hit@K / Recall@K / MRR 衡量文档找到没、找全没、排得好不好。自建指标还剩最后一组：性能。

对话产品的性能度量跟传统 API 有一个本质区别：用户不需要等完整回答出来才知道系统有没有响应。SSE 流式输出下，第一个字蹦出来的那一刻，用户的等待焦虑就解除了——后面的字像打字机一样逐个出现，体感是流畅的。所以性能指标的核心不是整流耗时，而是首字延迟——TTFT（Time To First Token）。

> 首字延迟相对来说是很重要的，作为一个C端用户来说，肯定是越快返回内容用户体验越好。如果问个简单问题，系统需要10-20秒的输出才能首字响应，除非很重要的问题，否则用户是没有耐心等待的。这种会间接造成用户的购买以及好感度流式。

## 整流耗时的误导：答案越长越慢

先看一个对比：

|  | query A：简短回答 | query B：详细回答 |
| --- | --- | --- |
| 回复长度 | 50 字 | 500 字 |
| TTFT | 4800ms | 5100ms |
| 整流耗时 | 10s | 30s |

如果只看整流耗时（`latency_ms`），query B 的性能比 A 差了 3 倍。但 TTFT 几乎一样——用户等答案首字的时间差不多，都在 5 秒左右，只是 B 的答案更详细，字蹦得更久而已。

整流耗时的问题很清楚：它跟生成的 token 数量线性相关。答案越长，整流耗时越大——但这不代表用户体验差了。拿它做性能基线会误导优化方向：你会发现性能最差的 query 全是回答最详细的那些，而它们的用户体验可能恰恰最好。

用一张时间轴来看两者的覆盖范围：

![图片.png](assets/83-01.png)

TTFT 覆盖的是从请求发出到答案首字出现的这段时间——Query 改写、意图识别、向量检索、Rerank、LLM 生成第一个 token，这些串行步骤全算在里面。这条链路走下来，5 秒左右的 TTFT 是常态。后面的 token 逐个输出只是增加了整流耗时，不影响用户对响应速度的感知。

## TTFT：用户体感的唯一卡点

### 1. 口径回顾

TTFT 的精确采集口径在第 4 篇已经详细讲过，这里只回顾关键定义：**只算 `type=response` 的首个非空 delta 到达时间，不算 thinking**。

如果模型开了深度思考模式，thinking 首字可能 23 秒就到了，但 response 首字要等到 68 秒（思考过程本身也需要时间）。TTFT 记的是 response 首字的时间——这是用户看到答案开始出现的时间，不是思考过程开始的时间。思考链对用户来说是隐藏的，不构成体感卡点。

### 2. 取不到时的回退

两种情况下 `first_token_ms` 会是 `None`：

- **老版本 run 文件**：早期 runner 还没实现 TTFT 采集，所有样本的 `first_token_ms` 都为空
- **异常请求**：被 reject、出错、或者全程只有 thinking 没有 response

回退逻辑只有 3 行：

```
def _first_token_or_total(r: EvalRecord) -> int | None:
    """老 runs 没采 first_token_ms 时回退到 latency_ms。"""
    return r.first_token_ms or r.latency_ms or None
```

优先用 `first_token_ms`；没有就退回到 `latency_ms`（整流耗时）；都没有返回 `None`，该样本不参与性能指标。

回退到 `latency_ms` 不完美——整流耗时远大于 TTFT——但有两个保底：同一批老 run 文件的所有样本都会统一回退，不会出现一半用 TTFT 一半用整流的混搭；而且只要重新跑一次 runner 就能拿到真实 TTFT，老数据只是过渡。

## 小样本下的分位数陷阱

### 1. P95 / P99 为什么不靠谱

先解释一下 P95 和 P99 是什么。把所有请求的 TTFT 从小到大排一排，P95 就是排在第 95% 位置的那个值——意思是 95% 的请求都比这个值快，只有最慢的 5% 比它慢。P99 同理，99% 的请求都比它快。P50 就是中位数，一半快一半慢。

P95 在业界用得很广，通常作为服务端性能的核心监控指标——它反映的是大部分用户能感知到的最差体验，比均值更敏感，能捕捉到尾部慢请求。

但这套指标有一个前提：**样本量要够大**。

之前想的是基于 P95 看首字或者看整流，但实际实现里没有用 P95——用的是 **P50 + 均值**。这不是偷懒，是小样本下的有意选择。

算一下 P95 在不同样本量下对应的排序位置：

| 样本数 | P95 索引 | 对应位置 | 代表什么 |
| --- | --- | --- | --- |
| 20 条 | int(20 × 0.95) = 19 | 倒数第 2 | 接近最慢的那条 |
| 50 条 | int(50 × 0.95) = 47 | 倒数第 4 | 接近最慢的几条 |
| 150 条 | int(150 × 0.95) = 142 | 倒数第 9 | 仍然接近尾部极值 |
| 1000 条 | int(1000 × 0.95) = 950 | 倒数第 51 | 真正代表 5% 慢请求的边界 |

20~150 条的规模下，P95 反映的不是大部分请求的尾部延迟，而是最倒霉的那几条碰巧有多慢。可能是网络抖了一下，可能是某个 LLM 调用卡了两秒，可能是刚好轮到了一条触发全局检索的超长 query——一条异常请求就能让 P95 波动好几秒。拿这个数字做性能基线，两次跑完结果不一致，你分不清是真退化了还是随机抖动。

P99 就更夸张——20 条下退化成跟 P95 一样的位置，150 条下是倒数第 3，几乎就是最大值本身。

> 不是说 P95 本身有问题。样本量到 5001000 条以上，P95 就能稳定代表第 95 分位的延迟水位了。后续评估集扩量之后可以加回来。当前 20150 条的规模，P50 + 均值更稳健。

### 2. P50 + 均值怎么读

关键是看两者的差距：

**情况一：均值 ≈ P50（健康）**

```
ttft_p50_ms    4820.0
ttft_mean_ms   5130.5
```

均值比 P50 高不到 7%，说明分布集中。大部分请求的 TTFT 在 5 秒上下，没有严重的离群值把均值拉偏。性能表现稳定。

**情况二：均值 >> P50（长尾）**

```
ttft_p50_ms    4820.0
ttft_mean_ms   7856.3
```

均值比 P50 高了 63%。一半请求在 5 秒以内就响应了，但另外一些请求特别慢，把均值拉到了将近 8 秒。这时候需要排查：哪几条请求 TTFT 特别高？

`latency.py` 会把逐样本的 TTFT 挂在 `per_sample_ttft` 字典里，按 `query_id` 索引。找到最慢的 3~5 条，看它们的意图分类和 query 内容——是不是都是多子问题拆分的复杂 query？是不是都走了全局检索？还是某个意图分支的检索链路特别慢？定位到原因才能优化。

## 代码实现：36 行的全部

`latency.py` 完整代码：

```
"""性能：首字延迟 (TTFT) 的 P50/均值 + 整流均值。

对话产品的体感卡点是 **正式回答首个 token 到达**（type=response 的首个 delta），
而不是完整流的总耗时（总耗时随 token 数线性增长，不反映"卡顿"）。
小样本下 P95/P99 退化为极值，改用均值。
"""
import statistics

from eval.common.schemas import EvalRecord, MetricResult

def compute(records: list[EvalRecord]) -> list[MetricResult]:
    ttfts = sorted(_first_token_or_total(r) for r in records if _first_token_or_total(r))
    totals = sorted(r.latency_ms for r in records if r.latency_ms)

    def pct(xs: list[int], q: float) -> float | None:
        return float(xs[min(len(xs) - 1, int(len(xs) * q))]) if xs else None

    ttft_p50 = pct(ttfts, 0.50)
    ttft_mean = float(statistics.mean(ttfts)) if ttfts else None
    total_mean = float(statistics.mean(totals)) if totals else None

    per_sample_ttft = {r.query_id: float(_first_token_or_total(r) or 0) or None for r in records}
    return [
        MetricResult("ttft_p50_ms", ttft_p50, is_pct=False),
        MetricResult("ttft_mean_ms", ttft_mean, is_pct=False, per_sample=per_sample_ttft),
        MetricResult("total_mean_ms", total_mean, is_pct=False),
    ]

def _first_token_or_total(r: EvalRecord) -> int | None:
    """老 runs 没采 first_token_ms 时回退到 latency_ms。"""
    return r.first_token_ms or r.latency_ms or None
```

几个设计点值得注意：

- `pct` 函数的分位数计算：对排序后的列表取 `int(len * q)` 位置的元素。P50 取中间那个值
- `per_sample_ttft` 挂在 `ttft_mean_ms` 这个 MetricResult 上，报告阶段可以按样本查看谁最慢
- 三个 MetricResult 的 `is_pct=False`——性能指标单位是毫秒，不是百分比

整个 compute 函数不做任何样本过滤（不像检索指标要排除 `requires_rag=false` 的样本）。所有有 TTFT 或 latency 数据的样本都参与，因为性能跟 query 类型无关——闲聊和知识检索都需要快速响应。

## 性能指标在报告里的层级

报告体系里，指标有明确的展示层级：

| 层级 | 指标 | 定位 |
| --- | --- | --- |
| 主 KPI | 意图准确率 / Hit@5 / faithfulness / answer_correctness | 决定系统能不能用 |
| 二级 KPI | TTFT 均值 | 对话体感卡点 |
| 参考 | 整流均值 | 仅供了解，不作为优化目标 |

slides.html 里 TTFT 均值标注为“TTFT · 对话体感卡点”，整流均值标注为“完整流式耗时 · 参考”——措辞本身就在引导读者把注意力放在 TTFT 上。

> 理想状态是 Ragent 侧也按 TTFT 做监控报警——在 `RAGChatService.streamChat()` 里记录第一个 response delta 写入 SSE 流的时间戳，Grafana 按 TTFT P95 设阈值。评测和监控看同一个口径，结论才能对得上。当前 Ragent 还没有服务端 TTFT 埋点，这是后续优化项。

## 自建指标全景回顾

写到这里，自建指标四组全部讲完了。回头看一眼全景：

| 组 | 核心指标 | 衡量什么 | 详见 |
| --- | --- | --- | --- |
| 意图 | intent_top1 | 路由对不对 | 第 5 篇 |
| 检索 | hit@K / recall@K / mrr@10 | 文档找到没 / 找全没 / 排得好不好 | 第 5 篇 |
| 性能 | ttft_p50 / ttft_mean / total_mean | 用户等多久 | 本篇 |
| 行为 | 误拒率 / 兜底率 / 过召回率 | 系统行为合不合理 | 第 11 篇 |

四组指标覆盖了意图、检索、性能、行为四个维度，全部是纯内存运算，秒级出结果，适合挂 CI 当闸门。

但有一个维度它们完全评不了：**内容对不对**。召回的 chunk 都对，检索指标全绿，但模型胡编了一段没在 chunk 里的话——自建指标看不出来。Hit@5 = 100%，`intent_top1` = 100%，TTFT 也在阈值内，可答案是错的。

这种语义级的评估需要另一套工具。一个不靠集合运算而靠 LLM 判断的框架——RAGAS。

## 小结与下一篇预告

对话产品看 TTFT 不看整流，小样本用 P50 + 均值不用 P95 / P99。口径选对了，优化方向才不会跑偏——你优化的应该是检索 + 意图 + LLM 首 token 生成这条串行链路的耗时，而不是答案生成了多少个字。

自建指标到这里全部讲完了。它们能算命中没命中、排第几、路由对不对、等了多久——但算不了内容对不对。召回的 chunk 都对，模型却胡编了一段，自建指标完全无感。这种语义级的评估需要另一套工具——RAGAS，一个全靠 LLM-as-judge 驱动的 RAG 评估框架。它怎么工作？几十个指标为什么只挑 5 个？跑一次要花多少钱？

> Source: https://t.zsxq.com/jrYMw
> Resolved: https://articles.zsxq.com/id_65dk42mwwq6l.html
