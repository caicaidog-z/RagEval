# 《AI大模型Ragent项目》——RAGAS五个指标全解读

RAGAS 跑通了，终端里出现了 5 个 0~1 的浮点数。faithfulness 0.85、answer_correctness 0.69——好还是不好？0.69 比上次的 0.72 低了 3 个点，是真退化还是随机波动？

要回答这些问题，得先搞清楚这些数字是怎么算出来的。

这篇把 5 个指标的算法拆透。每个指标讲清楚算法步骤，配一个比特严选的手算例子，让数字有体感。最后用三组配对读法告诉你怎么根据指标组合定位问题——单看某个指标的高低只有一半信息，组合起来看才能精准归因。

## 总览：5 个指标的坐标系

先建一个全局认知，后面每一节展开一行：

| 指标 | 层 | 比较对象 | 算法核心 | 不达标第一反应 |
| --- | --- | --- | --- | --- |
| faithfulness | 生成 | response vs contexts | 拆 response 的 claims → judge 判能否由 contexts 推出 | prompt 没限定知识来源 / 模型用预训练知识补了 contexts 里没有的内容 |
| answer_relevancy | 生成 | response vs user_input | 从 response 反向生成问题 → 与原问题算余弦相似度 | 回答跑题 / 啰嗦 / 答非所问 |
| answer_correctness | 生成 | response vs reference | 0.75 × claim F1 + 0.25 × embedding similarity | 端到端答案差，综合性信号 |
| context_precision | 检索 | contexts + user_input vs reference | judge 逐个 chunk 判有用没用 + Average Precision 排序得分 | rerank 没生效 / 过滤太宽 / 噪声 chunk 靠前 |
| context_recall | 检索 | contexts vs reference | 拆 reference 的 statements → judge 判能否由 contexts 覆盖 | chunk 切太细 / 检索策略漏 / 知识库缺内容 |

> **版本说明**：以下算法描述适用于 RAGAS 0.1.x ~ 0.4.x，核心计算逻辑跨版本未变。版本间的差异主要在 API 层面：0.2+ 将字段名从 `question`/`answer`/`contexts`/`ground_truth` 改为 `user_input`/`response`/`retrieved_contexts`/`reference`；对 context_precision 和 context_recall 新增了 NonLLM、ID-based 等变体（本文描述的是默认的 LLM-based 版本）。

## 五个指标的算法详解

### 1. faithfulness：回答里的话有据可查吗

faithfulness 是 5 个指标里最核心的一个——它检测幻觉。算法分两步。

#### 1.1 Step 1：拆 claims

RAGAS 调 judge 模型，把 response 拆成一组原子级事实声明（atomic claims）。每个 claim 是一条独立的、可验证的事实陈述。

**什么叫原子级**：拆到不能再拆、只包含一个可验证事实的粒度。一个好的 claim 应该可以独立判定真假，不依赖其他 claim 的上下文。

用一个例子感受粒度：

| 原始句子 | 拆出的原子级 claims |
| --- | --- |
| AirPods Pro 2 标准保修 1 年，自购买之日起算 | ① AirPods Pro 2 标准保修期为 1 年 ② 保修自购买之日起算 |
| 推荐购买 AppleCare+，延保至 3 年 | ③ AppleCare+ 可将保修延长至 3 年 |

注意第二句中推荐购买是主观建议，不是事实声明，不会成为 claim；但句子里蕴含的客观事实（AppleCare+ 的延保时长）仍然会被拆出。只有可以判定对还是错的陈述才算 claim。

拿比特严选的完整例子。response = AirPods Pro 2 标准保修 1 年，自购买之日起算。如果购买 AppleCare+ 可延保至 3 年。

judge 拆出三个 claims：

- claim 1：AirPods Pro 2 标准保修期为 1 年
- claim 2：保修自购买之日起算
- claim 3：购买 AppleCare+ 可延保至 3 年

#### 1.2 Step 2：逐个判 supportable

对每个 claim，judge 判断它能否从 retrieved_contexts 推导出来。结果是二值的：supported 或 not supported。

假设 contexts 里只有一段话：“标准保修期：自购买之日起 1 年。保修范围包括非人为损坏的硬件故障。”

- claim 1：**supported**（上下文明确写了保修 1 年）
- claim 2：**supported**（上下文写了自购买之日起）
- claim 3：**not supported**（上下文没提 AppleCare+ 延保——这是模型从预训练知识里补的）

计算：

```
faithfulness = supported / total = 2 / 3 ≈ 0.667
```

0.667 意味着回答里 1/3 的事实声明在上下文中找不到依据。claim 3 就是幻觉——模型编了一个 contexts 里没有的信息。

**不达标怎么办**：faithfulness 低 → 看 response 里哪些 claims 是 not supported → 如果是模型补了预训练知识 → 改 prompt 加强“只基于给定上下文回答”的约束、加兜底指令。如果 claim 本身应该被支持但 contexts 里漏了 → 问题出在检索阶段，转查 `context_recall`。

### 2. answer_relevancy：回答切题吗

answer_relevancy 衡量的是回答有没有跑偏。算法的思路很巧——反向推导。

**为什么不直接比 response 和 user_input 的相似度？**

直觉上会觉得：直接把 response 和 user_input 都做 embedding，算余弦相似度不就行了？问题是，一个好的回答跟原始问题在词汇和句式上往往差异很大。

比如用户问“退货怎么寄回去”，好的回答是“请在订单页点击申请退货，选择上门取件或到付寄回，地址是 xxx”。这段回答跟问题的文本相似度并不高——因为回答是信息展开，不是重复问题。直接比两者的 embedding 会得到偏低的分数。

RAGAS 的思路是：如果这个回答是切题的，那从回答反推出的问题应该跟原始问题很像。这比直接比 response 和 user_input 更能捕捉语义切题——回答的信息指向了正确的问题方向。

#### 2.1 Step 1：反向生成问题

judge 从 response 出发，反向生成 N 个问题（默认 N=3）。意思是：如果有人看到这个回答，他原本可能在问什么？

例如 response = “退款通常在 35 个工作日到账，信用卡支付可能需要 710 天。”

judge 反向生成：

- q1：退款多久到账？
- q2：信用卡退款需要多长时间？
- q3：退款到账时间是多少？

#### 2.2 Step 2：算余弦相似度

把 3 个反向问题和原始 user_input 分别做 embedding，算余弦相似度，取平均。

> 以下数值为简化示意，帮助理解分数高低的含义。实际分布取决于所用 embedding 模型，不同模型的相似度区间差异较大。

假设 user_input = 退货怎么寄回去？

| 反向问题 | 与 user_input 的余弦相似度 |
| --- | --- |
| q1：退款多久到账？ | 0.55 |
| q2：信用卡退款需要多长时间？ | 0.50 |
| q3：退款到账时间是多少？ | 0.52 |

```
answer_relevancy = mean(0.55, 0.50, 0.52) = 0.523
```

0.523 偏低。用户问的是退货流程（怎么寄回去），模型回答了退款到账时间——跑题了。反向生成的问题跟原问题不匹配，所以分数低。

再看一个切题的例子对比：

假设 response = 您可以在订单页面点击申请退货，选择上门取件或自行寄回，地址是北京市 xxx。

judge 反向生成：

- q1：退货怎么操作？
- q2：退货的寄送地址是什么？
- q3：退货流程是什么？

| 反向问题 | 与 user_input 的余弦相似度 |
| --- | --- |
| q1：退货怎么操作？ | 0.88 |
| q2：退货的寄送地址是什么？ | 0.82 |
| q3：退货流程是什么？ | 0.86 |

```
answer_relevancy = mean(0.88, 0.82, 0.86) = 0.853
```

对比一下：跑题的回答 0.523，切题的回答 0.853。差距来自反向生成问题跟原问题的语义匹配程度。

> **关键陷阱**：answer_relevancy 只看切题，不看正确性。一个事实完全错误的回答，只要看起来在回答用户的问题，relevancy 就可能很高。比如用户问“保修期多久”，模型回答“保修期 5 年”（实际 1 年），反向生成的问题会是“保修期多久”，跟原问题高度相似 → relevancy 接近 1.0，但答案是错的。所以 answer_relevancy 必须跟 `faithfulness` 和 `answer_correctness` 配套看。

**不达标怎么办**：answer_relevancy 低 → 回答跑题了 → 先查意图识别（自建指标 `intent_top1`）是不是分错了 → 如果意图对，看 prompt 是不是没有聚焦用户问题 → 或者 query rewrite 把用户问题改偏了。

### 3. context_precision：召回里有多少是有用的，排得好不好

context_precision 衡量的是检索质量的精度面——召回的 chunk 里有用的多不多，排得靠不靠前。

#### 3.1 Step 1：逐个 chunk 判有用没用

judge 对 retrieved_contexts 里的每个 chunk，结合 user_input 和 reference 做二值判断：这个 chunk 对正确回答用户问题有没有用。

注意这里的判断逻辑：judge 看的是这个 chunk 的信息是否有助于回答原始问题。具体来说，它参考 reference（标准答案）来判断——如果一个 chunk 包含了标准答案中涉及的信息，就算 useful。

#### 3.2 Step 2：按 Average Precision 计算

context_precision 不是简单算有用 chunk 占比——那样就丢失了**排序信息**。一组 5 个 chunk 里有 2 个有用的，排在第 1、2 位跟排在第 4、5 位，对生成质量的影响差别很大。

Average Precision 的直觉是：**每发现一个有用 chunk，就算一下截止到这个位置，有用的占比是多少，最后对所有有用位置的占比取平均**。有用 chunk 排得越靠前，截止位置的占比就越高，最终分数越高。

具体公式：

```
对每个 useful chunk 在位置 k：precision@k = 截止位置 k 时有用数量 / k
最终：context_precision = 所有 useful chunk 的 precision@k 之和 / useful chunk 总数
```

用一个例子。用户问 AirPods Pro 2 保修期，检索回来 5 个 chunk：

| 位置 | chunk 内容摘要 | 判定 |
| --- | --- | --- |
| 1 | AirPods Pro 2 售后政策 | useful ✓ |
| 2 | 扫地机器人使用手册 | not useful |
| 3 | AirPods Pro 2 产品规格 | useful ✓ |
| 4 | 电视遥控器配对说明 | not useful |
| 5 | 耳机清洁指南 | not useful |

逐步计算：

- 位置 1 是 useful：截止位置 1，有用 1 个，总共看了 1 个 → precision@1 = 1/1 = 1.0
- 位置 2 是 not useful：跳过，不参与计算
- 位置 3 是 useful：截止位置 3，有用 2 个，总共看了 3 个 → precision@3 = 2/3 ≈ 0.667
- 位置 4、5 是 not useful：跳过

```
context_precision = (1.0 + 0.667) / 2 ≈ 0.833
```

**对比感受排序的影响**：如果两个 useful chunk 都排在前两位（位置 1 和 2），分数会变成：

- precision@1 = 1/1 = 1.0
- precision@2 = 2/2 = 1.0
- context_precision = (1.0 + 1.0) / 2 = 1.0

如果两个 useful chunk 排在最后两位（位置 4 和 5）：

- precision@4 = 1/4 = 0.25
- precision@5 = 2/5 = 0.40
- context_precision = (0.25 + 0.40) / 2 = 0.325

同样是 5 个 chunk 里 2 个有用，排前面得 1.0，排后面只有 0.325——差了 3 倍。这就是为什么 context_precision 不只看有没有找到，还看排在哪。

**不达标怎么办**：context_precision 低 → 召回里噪声多或排序差 → 先看噪声是不是排在相关 chunk 前面 → 检查 rerank 是不是没生效，或 metadata 过滤是否太宽。只有在相关 chunk 已稳定排在前列、后面主要是冗余噪声时，才优先考虑缩小 topK；否则盲目缩小 topK 可能会进一步拉低 `context_recall`。

### 4. context_recall：标准答案需要的信息找全没

context_recall 衡量的是检索质量的覆盖面——标准答案所需的信息，contexts 找到了多少。

#### 4.1 Step 1：拆 reference 的 statements

跟 faithfulness 的第一步类似，但拆解对象不同。faithfulness 拆的是 response（实际回答），context_recall 拆的是 reference（标准答案）。

judge 把 reference 拆成一组原子级陈述。

reference = AirPods Pro 2 标准保修 1 年，自购买之日起算，覆盖非人为硬件故障。

拆出：

- statement 1：AirPods Pro 2 标准保修期为 1 年
- statement 2：保修自购买之日起算
- statement 3：保修覆盖非人为硬件故障

#### 4.2 Step 2：判 contexts 能不能覆盖

对每个 statement，judge 判断它能否从 retrieved_contexts 推导出来。

假设 contexts 里有“标准保修期：自购买之日起 1 年”但没有提到保修范围：

- statement 1：**covered**（contexts 明确写了 1 年）
- statement 2：**covered**（contexts 写了自购买之日起）
- statement 3：**not covered**（contexts 里没有保修范围的信息）

```
context_recall = covered / total = 2 / 3 ≈ 0.667
```

0.667 说明标准答案需要 3 个信息点，contexts 只覆盖了 2 个——保修范围这个信息没被召回。

再看两个极端情况帮助理解：

- 如果 contexts 覆盖了全部 3 个 statement → context_recall = 3/3 = 1.0 → 检索没有遗漏
- 如果 contexts 一条都没覆盖 → context_recall = 0/3 = 0.0 → 检索完全没找到相关内容，可能是知识库缺这块信息，或者检索策略对这类 query 有盲区

**与 faithfulness 的对偶关系：**

| 维度 | faithfulness | context_recall |
| --- | --- | --- |
| 拆解对象 | response（实际回答） | reference（标准答案） |
| 拆出的单元 | claims（事实声明） | statements（信息点） |
| 判断依据 | contexts 能不能支撑 | contexts 能不能覆盖 |
| 衡量什么 | 回答有没有编（幻觉） | 需要的信息找全没（遗漏） |
| 方向 | 从回答往 contexts 对 | 从标准答案往 contexts 对 |

两者像一枚硬币的两面：faithfulness 从回答出发查多说了什么，context_recall 从标准答案出发查漏掉了什么。一个管幻觉（说了不该说的），一个管遗漏（该说的没说）。

**不达标怎么办**：context_recall 低 → 标准答案需要的信息没在 contexts 里 → 按以下顺序排查：

- **知识库缺内容**：那条信息压根就没入库——去知识库里搜一下，搜不到就是源头问题
- **chunk 切分太细**：关键信息被切到两个 chunk 里，单个 chunk 不完整导致 judge 判为 not covered
- **检索策略盲区**：某类 query 的表述跟知识库内容的措辞差异大，向量检索找不到——考虑加关键词检索兜底

### 5. answer_correctness：端到端答案对不对

answer_correctness 是一个综合性指标，衡量回答与标准答案的一致程度。算法是加权复合：

```
answer_correctness = 0.75 × claim_F1 + 0.25 × embedding_similarity
```

#### 5.1 claim F1 计算

judge 把 response 和 reference 都拆成 claims，然后做集合比对：

- **TP**（True Positive）：response 里有的 claim，reference 里也有——事实一致
- **FP**（False Positive）：response 里有但 reference 里没有——可能是多余信息，也可能是错误信息，claim F1 不区分二者
- **FN**（False Negative）：reference 里有但 response 里没有——漏说了

```
precision = TP / (TP + FP)    → response 说的里面有多少跟 reference 一致
recall    = TP / (TP + FN)    → reference 要求的里面有多少被 response 覆盖了
F1        = 2 × precision × recall / (precision + recall)
```

**judge 怎么判断两个 claim 相同**：这是实操中的常见困惑。RAGAS 不是做字符串精确匹配，而是让 judge 模型做语义判断。比如：

- response claim：保修期是 12 个月
- reference claim：标准保修 1 年

judge 会判为相同——因为语义一致，只是措辞不同。但如果：

- response claim：保修期是 2 年
- reference claim：标准保修 1 年

judge 会判为不同——数值不一致就是不同的事实。

这个语义判断依赖 judge 模型的能力。大部分情况下 GPT-4 级别的 judge 能正确处理同义改写，但偶尔也会误判——比如中文里“7 天无理由”和“七天无理由退货”可能被判为不同 claim。这类噪声在下一篇会展开讲。

#### 5.2 embedding similarity

把 response 和 reference 整体做 embedding，算余弦相似度。这是一个语义层面的软匹配，不做 claim 拆分，看的是整体意思像不像。

**为什么需要两部分结合**：

- 只用 claim F1：过于严格。同义不同词可能被判为不同 claim，导致分数偏低
- 只用 embedding similarity：过于宽松。两段话大意相近但关键细节错误时，embedding 分数仍然可能很高

0.75 权重给硬指标（逐条事实对齐），0.25 权重给软指标（整体语义相似），既保证事实准确性优先，又容忍合理的表述差异。

#### 5.3 手算一个完整例子

- response = AirPods Pro 2 标准保修 1 年。
- reference = AirPods Pro 2 标准保修 1 年，自购买之日起算，覆盖非人为硬件故障。
- response claims：[保修 1 年]
- reference claims：[保修 1 年，自购买之日起算，覆盖非人为硬件故障]

集合比对：

- TP = 1（保修 1 年两边都有）
- FP = 0（response 没多说任何 reference 里没有的内容）
- FN = 2（自购买之日起算 + 覆盖非人为硬件故障，reference 有但 response 没说）

```
precision = 1 / (1 + 0) = 1.0      → response 说的全是对的（虽然只说了一条）
recall    = 1 / (1 + 2) ≈ 0.333    → reference 要求的只覆盖了 1/3
F1        = 2 × 1.0 × 0.333 / (1.0 + 0.333) = 0.5
```

embedding similarity 假设 = 0.85（两段话大意相同，只是 response 少了细节，整体语义接近）

```
answer_correctness = 0.75 × 0.5 + 0.25 × 0.85 = 0.375 + 0.2125 ≈ 0.588
```

0.588 说明答案对了一部分但不完整——保修期说对了，但漏了两个关键信息点。precision 是 1.0（没说错话），但 recall 只有 0.333（该说的大部分没说）。

**再看一个说多了的例子**：

- response = AirPods Pro 2 标准保修 1 年，自购买之日起算，支持全球联保。
- reference = AirPods Pro 2 标准保修 1 年，自购买之日起算，覆盖非人为硬件故障。
- response claims：[保修 1 年，自购买之日起算，支持全球联保]
- reference claims：[保修 1 年，自购买之日起算，覆盖非人为硬件故障]

集合比对：

- TP = 2（保修 1 年 + 自购买之日起算）
- FP = 1（全球联保——response 说了但 reference 里没有，可能是正确但多余的信息，也可能是错误信息）
- FN = 1（覆盖非人为硬件故障——reference 有但 response 没说）

```
precision = 2 / (2 + 1) ≈ 0.667    → response 说的有 1/3 是 reference 里没要求的
recall    = 2 / (2 + 1) ≈ 0.667    → reference 要求的覆盖了 2/3
F1        = 2 × 0.667 × 0.667 / (0.667 + 0.667) = 0.667
```

假设 embedding similarity = 0.88

```
answer_correctness = 0.75 × 0.667 + 0.25 × 0.88 = 0.5 + 0.22 = 0.72
```

0.72 比上一个 0.588 高——因为覆盖了更多信息点，虽然多说了一条 reference 里没有的。

#### 5.4 与 faithfulness 的本质区别

| 维度 | faithfulness | answer_correctness |
| --- | --- | --- |
| 比较对象 | response vs contexts | response vs reference |
| 衡量什么 | 回答有没有编（忠于检索上下文） | 回答对不对（跟标准答案一致） |
| 能查出什么问题 | 模型编了 contexts 里没有的内容 | 答案跟标准答案有偏差（不管偏差来源） |

一个答案可能 faithful 但不 correct——contexts 里的信息本身过时或不完整，模型忠实地引用了不准确的内容。也可能 correct 但不 faithful——模型靠预训练知识答对了，但答案在 contexts 里找不到依据。

用一个例子说清楚：

> **场景**：知识库里那篇保修文档还是旧版，写着保修期 6 个月。实际政策已经更新为 1 年。
>
>
> - response = 保修期 6 个月（忠实引用了 contexts）
> - reference = 保修期 1 年（标准答案是新政策）
>
>
> 结果：faithfulness 很高（确实是 contexts 里说的），answer_correctness 很低（跟标准答案不一致）。
>
>
>
> 诊断：问题不在模型，也不在 prompt，而在知识库内容过时。

对 RAG 来说，correct 但不 faithful 是另一种风险信号：模型靠预训练知识答对了，这次碰巧对了，但下次预训练知识过时就会出错。RAG 的核心价值就是用检索到的实时信息取代预训练知识，所以即使答对了也应该关注 faithfulness。

**不达标怎么办**：answer_correctness 低 → 先看是哪边拖后腿：

- precision 低（FP 多）→ 回答说了不该说的 → 查 faithfulness 是不是也低（在编）或查 contexts 里是否有误导信息
- recall 低（FN 多）→ 回答漏了该说的 → 查 context_recall 是不是也低（检索没找全）或查 prompt 是不是没引导模型把信息说完整

answer_correctness 是端到端的综合信号，本身不直接告诉你问题出在哪一层。它的价值是“报警”，定位问题要靠其他 4 个指标的组合。

## 三组配对读法：组合看指标才能精准归因

单看一个指标只能知道有问题，但不知道问题出在哪。下面三组配对读法覆盖了日常迭代中最常见的归因场景。

### 配对一：context_recall × faithfulness → 区分没找到和乱编

这组配对回答一个核心问题：**回答不对，是因为检索没找到正确信息，还是因为模型拿到了信息却在编？**

| context_recall | faithfulness | 诊断 |
| --- | --- | --- |
| 高 | 高 | 检索找到了，模型也忠实引用了 → 管道健康 |
| 高 | 低 | 检索找到了，但模型没忠实引用，额外编了内容 → prompt 问题，加强“只基于给定上下文回答”的约束 |
| 低 | 高 | 检索没找全，但模型对已有内容很忠实 → 检索问题，去查 chunk 策略、知识库覆盖 |
| 低 | 低 | 检索没找全，模型还在编 → 两层都有问题，先修检索（因为检索是输入，生成是输出，输入质量不行时调输出意义不大） |

**实操场景**：你发现某类售后问题 answer_correctness 持续偏低。拉出这类样本的 context_recall 和 faithfulness 一看：context_recall 0.4、faithfulness 0.9。诊断清晰——模型没在编，是检索根本没找到售后政策相关的 chunk。去知识库一查，发现售后政策文档用的是 PDF 扫描件，OCR 识别有误，导致关键段落没被正确索引。

### 配对二：context_precision × context_recall → 检索的精度和覆盖怎么平衡

这组配对回答：**检索策略调宽调窄的方向对不对？**

| context_precision | context_recall | 诊断 |
| --- | --- | --- |
| 高 | 高 | 找得全，噪声少，排序好 → 检索层健康 |
| 高 | 低 | 召回里噪声少但覆盖不够 → 策略太窄，检索只找到了一部分相关内容 → 考虑加大 topK、加混合检索、放宽 metadata filter |
| 低 | 高 | 覆盖够但噪声多、排序差 → 策略太宽，什么都召回了但有用的排不到前面 → 加 rerank、收紧 metadata filter、考虑缩小 topK |
| 低 | 低 | 又没找全又噪声多 → 检索策略有系统性问题，向量模型 / chunk 策略 / 知识库结构需要整体审视 |

**实操场景**：某次迭代把 topK 从 5 调到 10，context_recall 从 0.6 升到 0.8（覆盖提升了），但 context_precision 从 0.85 降到 0.55（噪声明显增加了）。下一步不是把 topK 调回来，而是加 rerank——在更大的召回池里通过精排把有用 chunk 排到前面，同时保持覆盖率。

### 配对三：answer_relevancy × answer_correctness → 区分跑题和答错

这组配对回答：**回答分数低，是因为没在回答用户的问题，还是在回答但答得不对？**

| answer_relevancy | answer_correctness | 诊断 |
| --- | --- | --- |
| 高 | 高 | 回答切题且正确 → 生成层健康 |
| 高 | 低 | 看起来在回答用户的问题，但内容跟标准答案对不上 → 答错了，查 faithfulness（是不是在编）或查 contexts（是不是喂了错误信息） |
| 低 | 高 | 内容跟标准答案对得上，但回答方式跑偏了 → 比较少见，可能是回答太啰嗦、包含大量无关展开，导致反向生成的问题偏了 → 精简 prompt |
| 低 | 低 | 既跑题又答错 → 先查意图识别，可能用户问题被分到错误的处理流程 |

**实操场景**：用户问“这款耳机防水吗”，模型回答了一大段耳机音质参数和降噪等级。answer_relevancy 0.35（跑题），answer_correctness 0.30（跟标准答案 IP54 防水等级完全不搭）。查根因：query rewrite 环节把防水误理解为产品规格查询，检索走了规格参数表而不是功能特性表。修 query rewrite 的策略就能解决。

## 最低验收线：先给指标一个解释框架

> 前面两篇提到过，这里再补充下，帮助大家查看。

指标能驱动迭代的前提，是团队知道“多少算低”。RAGAS 的绝对分数不能脱离业务和评估集解释，但可以先给一组工程上的参考线：

| 指标 | 参考解释 |
| --- | --- |
| faithfulness | 低于 0.8 优先查幻觉；高风险场景（售后政策、价格、保修条款）应更严格 |
| answer_relevancy | 低于 0.75 优先查是否答非所问、漏答核心问题或过度展开 |
| answer_correctness | 低于 0.75 说明端到端答案质量需要人工看样本，不应只看均值 |
| context_precision | 低说明 contexts 噪声多或排序差，优先查 TopK / rerank / metadata filter |
| context_recall | 低说明 reference 需要的信息没被召回，优先查 chunk、召回策略、知识库覆盖 |

**关于阈值的说明**：上表的数字不是绝对标准线，是工程上的参考起点。不同业务场景对分数的要求差异很大：

- **严格场景**（保修条款、退款金额、规章制度）：faithfulness 低于 0.9 就应该排查，因为一个错误 claim 可能导致客诉或法律风险
- **宽松场景**（产品推荐、使用建议、闲聊类）：即使分数稍低，可能只是语义上的表述差异（比如建议搭配充电盒使用 vs 推荐配合充电盒一起用），并不影响用户体验

正确的做法是：先用上表作为第一轮筛选的起点，跑完第一批评估后，人工抽检低分样本，看看低分是真问题还是评估噪声，再根据业务实际校准阈值。

## 小结

5 个指标拆完，回到开头的问题：faithfulness 0.85、answer_correctness 0.69，好还是不好？

现在你可以这样读：

- faithfulness 0.85 → 85% 的 claims 在 contexts 中找到了支撑，15% 找不到支撑（可能是真幻觉，也可能是 judge 误判）→ 够不够取决于业务场景，高风险场景不够
- answer_correctness 0.69 → 端到端答案质量偏低 → 不直接告诉你原因，需要配合其他指标定位
- 如果 context_recall 也低 → 问题大概率在检索层
- 如果 context_recall 高但 faithfulness 低 → 问题在生成层，模型拿到了正确信息但没好好用

指标本身不解决问题，但它给你一个精确的坐标——知道问题在哪一层、哪个环节，才能避免感觉哪里不对就到处调参的低效循环。

> Source: https://t.zsxq.com/PaWec
> Resolved: https://articles.zsxq.com/id_vakm5sn6wg2p.html
