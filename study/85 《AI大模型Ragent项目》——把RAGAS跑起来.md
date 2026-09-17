# 《AI大模型Ragent项目》——把RAGAS跑起来

上一篇确定了选型——faithfulness / answer_relevancy / answer_correctness / context_precision / context_recall，5 个指标覆盖幻觉、切题、正确、检索精度、检索覆盖。跟自建指标的分工也讲清楚了：自建当 CI 闸门，RAGAS 当离线深度评估。

但选型是纸上谈兵，得跑起来才算数。

这篇从零开始：装环境、配模型、喂数据、拿分数。跟着做完，你的终端里会出现第一行 RAGAS 评分——一个 0~1 之间的浮点数，代表你的 RAG 系统在某个维度上的表现。

## Part 1：Python 环境

**为什么选 Python 3.11+？**

RAGAS 的核心依赖链（`pydantic`、`langchain-core`、`datasets`、`tokenizers`）在 Python 3.11 上兼容性最稳定。3.10 偶尔遇到 `pydantic-core` 编译问题，选 3.11 不是唯一选择，是稳妥选择。

> 如果安装了更高版本的 3.12 / 3.13 等，遇到问题可回滚至 3.11，目前仅测试该版本稳定无问题。

## Part 2：依赖安装

### 1. 三个核心依赖

| 包 | 作用 |
| --- | --- |
| ragas | RAGAS 评估框架本体，提供指标定义和 evaluate() 函数 |
| langchain-openai | 连接 judge 模型和 embedding 模型（RAGAS 通过 LangChain 接口调 LLM） |
| datasets | HuggingFace 的数据容器，RAGAS 要求输入格式为 Dataset 对象 |

一条命令搞定：

```
pip install ragas langchain-openai datasets
```

### 2. 常见安装坑

| 问题 | 原因 | 解决 |
| --- | --- | --- |
| 下载极慢 / 超时 | PyPI 国内访问慢 | 换镜像：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ragas langchain-openai datasets |
| pydantic 版本冲突报错 | 其他包锁了 pydantic v1 | 虚拟环境隔离；或 pip install "pydantic>=2.0,<3.0" 强制升 |
| macOS M 芯片报 tokenizers 编译错误 | Rust 编译工具缺失 | 先跑 xcode-select --install，或 pip install --prefer-binary tokenizers |
| ImportError: cannot import name 'RunConfig' | ragas 版本太旧 | pip install --upgrade ragas |
| numpy / scipy 编译报错 | 缺少系统级数学库 | Linux 装 apt install libopenblas-dev；macOS 通常不会遇到 |

### 3. 验证安装

创建个 `test.py` 文件测试下输出。

```
import ragas
print(ragas.__version__)  # 应输出 0.4.x

from ragas.metrics._faithfulness import faithfulness
print(faithfulness)  # 不报错就行
```

## Part 3：Judge 模型配置

### 1. 为什么需要两个模型

RAGAS 的指标内部需要做两件事：

- **LLM 推理**：拆 claims、判 supportable、生成反向问题——需要一个 judge LLM
- **Embedding 计算**：`answer_relevancy` 要算反向生成问题与原问题的余弦相似度——需要一个 embedding 模型

两个都得配。只配 LLM 不配 embedding，跑到 `answer_relevancy` 时会报错。只配 embedding 不配 LLM，所有指标都跑不了。

### 2. 环境变量配置

本项目通过 [aihubmix](https://aihubmix.com/?lang=zh-CN)（OpenAI 兼容的 API 聚合平台）来调 judge 模型。配 4 个环境变量：

```
export AIHUBMIX_API_KEY=<your_key>
export AIHUBMIX_BASE_URL=https://aihubmix.com/v1
export JUDGE_MODEL=gpt-5.4-mini
export EMBEDDING_MODEL=text-embedding-3-large
```

最早用了 GPT 5.5 模型跑测评，那个 Token 花的呀，贼离谱。所以，后面选择了 5.4-mini 模型，跑一次大概10-20人民币。如果大家觉得费用偏贵，可以使用我已经列好的测评项目下的 `examples` 目录文件查看。

> 这个变量就是在命令行窗口，执行 Python 命令前执行的，仅在当前窗口有效。如果你开个新窗口，还需要再单独配置。

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| AIHUBMIX_API_KEY | API 认证密钥（必填，缺失直接报错） | 无 |
| AIHUBMIX_BASE_URL | API 端点地址 | https://aihubmix.com/v1 |
| JUDGE_MODEL | judge LLM 模型 ID | gpt-5.4-mini |
| EMBEDDING_MODEL | embedding 模型 ID | text-embedding-3-large |

> 如果你不用 aihubmix 而是直接用 OpenAI，把 `AIHUBMIX_BASE_URL` 改成 `https://api.openai.com/v1`、key 换成 OpenAI 的 key 就行。任何兼容 OpenAI Chat Completions API 的平台都能用——SiliconFlow、Azure OpenAI、各种国内代理都可以。

### 3. 关于 judge 模型的选择

被评系统（Ragent）用 GPT / DeepSeek / Qwen 做问答生成。Judge 模型如果也用同族模型评分，会有同源偏置——同族模型对相似风格的输出天然给更高分。

理想情况是被评和 judge 用不同族模型。本项目当前的妥协是 judge 用 gpt-5.4-mini——成本低、速度快、JSON 输出稳定，Ragent 默认测评用的 Qwen 模型，两者也能进行有效区分。

## Part 4：跑第一个分数

### 1. 输入数据格式

RAGAS 的 `evaluate()` 函数接收一个 HuggingFace `Dataset` 对象，必须包含以下四个字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| user_input | str | 用户的原始问题 |
| response | str | RAG 系统的实际回答 |
| retrieved_contexts | list[str] | 检索到的上下文列表（每个元素是一个 chunk 文本） |
| reference | str | 标准答案（ground truth） |

> 字段命名注意：RAGAS 用 `user_input` 不是 `question`，`response` 不是 `answer`，`retrieved_contexts` 不是 `contexts`。写错了会报 KeyError，而且报错信息不够直观，排查半天才发现是列名问题。

### 2. 最小可运行 demo

下面是一个完整的 demo 脚本。用比特严选的 AirPods Pro 2 保修期场景——跟上一篇幻觉例子呼应：

```
"""RAGAS Hello World：跑通第一个分数。

运行前确保：
1. pip install ragas langchain-openai datasets
2. export AIHUBMIX_API_KEY=<your_key>
"""
import os

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics._answer_correctness import answer_correctness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness

# ── 1. 配置 judge 模型 ─────────────────────────────────────
api_key = os.environ["AIHUBMIX_API_KEY"]
base_url = os.environ.get("AIHUBMIX_BASE_URL", "https://aihubmix.com/v1")
judge_model = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")
emb_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
answer_relevancy.strictness = int(os.environ.get("RAGAS_ANSWER_RELEVANCY_STRICTNESS", "1"))
show_progress = os.environ.get("RAGAS_SHOW_PROGRESS", "1").lower() in {"1", "true", "yes"}

judge = ChatOpenAI(
    model=judge_model,
    api_key=api_key,
    base_url=base_url,
    temperature=0,
    timeout=180,
)
emb = OpenAIEmbeddings(
    model=emb_model,
    api_key=api_key,
    base_url=base_url,
    timeout=180,
)

# ── 2. 准备数据（比特严选：AirPods Pro 2 保修期） ──────────
ds = Dataset.from_dict({
    "user_input": [
        "AirPods Pro 2 的保修期是多久？",
        "扫地机器人充不进电怎么办？",
    ],
    "response": [
        "AirPods Pro 2 的标准保修期为 1 年，自购买之日起算，覆盖非人为硬件故障。",
        "建议按以下步骤排查：1) 检查充电座触点是否有灰尘；2) 确认电源线插紧；3) 在 0-40°C 环境静置 30 分钟后重试。如仍无法充电建议联系售后。",
    ],
    "retrieved_contexts": [
        [
            "【AirPods Pro 2 售后政策】标准保修期：自购买之日起 1 年。"
            "保修范围包括非人为损坏的硬件故障。如需延长保修，可购买 AppleCare+ 服务。"
        ],
        [
            "【扫地机器人故障排查】充电异常处理：1) 将主机放回充电座，观察指示灯；"
            "2) 检查电源线是否插紧、插座是否通电；3) 擦拭主机和充电座触点；"
            "4) 在 0-40°C 环境静置 30 分钟后再充电。如反复出现 E13/E14 错误码，"
            "检查附近是否有强磁物体。持续异常建议转人工售后。"
        ],
    ],
    "reference": [
        "AirPods Pro 2 标准保修 1 年，自购买之日起算，覆盖非人为硬件故障。",
        "先按顺序排查：检查充电座触点、确认电源线和插座、擦拭触点、在合适温度环境静置后重试。如有 E13/E14 错误码检查是否靠近强磁物体。持续异常转人工售后。",
    ],
})

# ── 3. 跑评测 ─────────────────────────────────────────────
result = evaluate(
    dataset=ds,
    metrics=[faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall],
    llm=judge,
    embeddings=emb,
    show_progress=show_progress,
)

# ── 4. 输出结果 ───────────────────────────────────────────
df = result.to_pandas()
print(df[["faithfulness", "answer_relevancy", "answer_correctness", "context_precision", "context_recall"]])
print(f"\n均值：")
for col in ["faithfulness", "answer_relevancy", "answer_correctness", "context_precision", "context_recall"]:
    print(f"  {col:<24s} = {df[col].mean():.3f}")

```

### 3. 跑成功的标志

上面这个文件我放到了 `examples/ragas_quickstart.py` 位置，如果大家跑的话，进入 `ragenteval` 根目录下，执行下述命令：

```
export AIHUBMIX_API_KEY=<your_aihubmix_key>
python examples/ragas_quickstart.py
```

终端应该输出类似：

```
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:50<00:00,  5.06s/it]
   faithfulness  answer_relevancy  answer_correctness  context_precision  context_recall
0           1.0          0.933871            0.992231                1.0             1.0
1           0.8          0.354321            0.667502                1.0             1.0

均值：
  faithfulness             = 0.900
  answer_relevancy         = 0.644
  answer_correctness       = 0.830
  context_precision        = 1.000
  context_recall           = 1.000
```

5 列都有 0~1 之间的浮点数，没有 NaN，没有报错——RAGAS 跑通了。

> 实际分数会跟这里的示意不同（取决于 judge 模型、温度、当时的 API 响应）。重点是看到正常的数值输出，不是看具体数字。

### 4. 跑失败排查

| 报错 | 原因 | 解决 |
| --- | --- | --- |
| AuthenticationError / 401 | API key 错误或未设环境变量 | echo $AIHUBMIX_API_KEY 确认非空且正确 |
| NotFoundError / 404 | 模型名不对 | 确认 JUDGE_MODEL 跟平台支持的 model ID 一致 |
| TimeoutError / 连接超时 | 网络不稳定 | 加大 timeout（已在 demo 里设了 180s），或检查代理 |
| OutputParserException | judge 返回了非 JSON 格式 | 在 ChatOpenAI 里加 model_kwargs={"response_format": {"type": "json_object"}} |
| KeyError: 'user_input' | Dataset 列名写错 | 对照上面的表格检查四个字段名拼写 |
| 结果全是 NaN | judge 每次调用都失败 | 检查 API key、网络、模型可用性 |
| RateLimitError / 429 | 触发平台限流 | 等几秒重试，或降低并发 |

**加 RunConfig 控制超时和重试（推荐）：**

```
from ragas.run_config import RunConfig

result = evaluate(
    dataset=ds,
    metrics=[faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall],
    llm=judge,
    embeddings=emb,
    run_config=RunConfig(max_retries=3, timeout=180),
)
```

`RunConfig` 控制 RAGAS 内部每次 LLM 调用的超时和重试次数。网络不稳定时加上能避免一次偶发超时就整体失败。

## Part 5：从 demo 到项目封装

### 1. 差距对比

教学 demo 能跑通第一个分数，但项目里实际用的 `eval/rag/metrics/ragas_judge.py` 做了更多工程化处理：

| 维度 | 教学 demo | 项目封装（ragas_judge.py） |
| --- | --- | --- |
| 数据来源 | 手写 1 条样本 | 从 runs/*.jsonl 加载完整 EvalRecord 列表 |
| 样本过滤 | 无 | filter_evaluable()：跳过 response 空 / contexts 空 / reference 空 / final_status ≠ success |
| 指标数量 | 硬编码 5 个 | 固定跑 5 个 |
| 方差控制 | 单次跑 | --ragas-n 3 并发跑 3 次，按样本取均值 |
| 错误处理 | 报错即停 | 三层 fallback：batch JSON mode → batch 无 JSON mode → 逐条 eval |
| NaN 处理 | 无 | 全量跑完后对 NaN 样本单独重试（加大 timeout 到 1200s） |
| 结果聚合 | 打印 DataFrame | 按 intent_l1 / intent_l2 分层统计均值，输出 MetricResult |

这些增强不需要现在搞懂——先跑通 demo 理解 RAGAS 的输入输出和基本流程，回头看项目代码时就不会对 `evaluate()`、`Dataset`、`MetricResult` 这些概念陌生了。

### 2. 项目命令行

跑通 demo 之后，日常评测用项目封装好的命令行。运行前需要先配好 ragent 服务端地址和 RAGAS judge 的环境变量（详见项目根目录 `README.md` 的环境变量章节）。最简单的方式是一键全量：

```
python -m eval rag all                      # run → score → report 一条龙
python -m eval rag all --limit 5            # smoke：只跑 5 条
python -m eval rag all -w 5                 # 5 线程并行
python -m eval rag all --skip-ragas         # 跳过 LLM-judge，仅自建指标
python -m eval rag all --ragas-limit 10     # RAGAS 只评前 10 条
python -m eval rag all --ragas-n 3          # RAGAS 独立跑 3 次取均值
```

也可以分步运行，按需组合：

```
# 1. 调 ragent 跑评测
python -m eval rag run --limit 20                        # 全量 20 条
python -m eval rag run --limit 5                         # smoke
python -m eval rag run --filter-intent S1_选购推荐        # 定向调试
python -m eval rag run --debug                           # 保留原始 SSE 字节流

# 2. 算所有指标
python -m eval rag score                                 # 自建 + RAGAS
python -m eval rag score --skip-ragas                    # 跳过 LLM-judge
python -m eval rag score --ragas-limit 10                # RAGAS 只评前 10 条
python -m eval rag score --ragas-n 3                     # RAGAS 3 次取均值
python -m eval rag score runs/v1_xxx.jsonl               # 指定 runs 文件

# 3. 出报告
python -m eval rag report                                # 默认 swiss 主题
python -m eval rag report --theme magazine               # 杂志风
python -m eval rag report --only-slides                  # 只重出 slides.html

# 4. A/B 对比
python -m eval rag diff v1_xxx v1_yyy                    # 终端看对比
python -m eval rag diff run_a run_b -o reports/diff.md   # 同时落 markdown
```

RAGAS 是 score 阶段的第 5 步。即使它因为 API key 没配或网络超时失败了，前 4 步自建指标照样落盘——不会因为 RAGAS 挂了就丢掉已经算好的意图 / 检索 / 行为 / 性能指标。

典型输出：

```
[1/5] intent ...
[2/5] retrieval ...
[3/5] behavior ...
[4/5] latency ...
[5/5] ragas (LLM-as-judge) ...
RAGAS：可评 18 条，跳过 2 条
  - empty retrieved_contexts: 2
  judge=gpt-5.4-mini  embedding=text-embedding-3-large  via https://aihubmix.com/v1
Evaluating: 100%|████████████████████████| 5/5 [02:34<00:00, 30.8s/it]

落盘：eval/reports/v1_20260530_163213/_scores.json

=== 整体指标 ===
  intent_top1              = 85.0%
  hit@5                    = 90.0%
  recall@5                 = 76.7%
  mrr@10                   = 0.678
  refusal_when_required    = 5.0%
  ttft_p50_ms              = 4820
  ttft_mean_ms             = 5130
  faithfulness             = 0.847
  answer_relevancy         = 0.812
  answer_correctness       = 0.693
  context_precision        = 0.778
  context_recall           = 0.756
```

上半段是自建指标（秒出），下半段是 RAGAS（等了两分半）。两套指标一起落到 `_scores.json` 里，后续 report 阶段统一读取、统一呈现。

> 输出里 `跳过 2 条` 是因为这两条样本的 `retrieved_contexts` 为空——被 reject 或走了兜底，没有检索上下文，RAGAS 没法评。跳过是正确行为，不需要处理。

## 小结与下一篇预告

RAGAS 跑起来了。回顾一下整条路径：

- Python 3.11 + 虚拟环境
- `pip install ragas langchain-openai datasets`
- 配好 judge 模型和 embedding 模型的环境变量
- 构造包含 `user_input` / `response` / `retrieved_contexts` / `reference` 的 Dataset
- 调 `evaluate()` 拿到分数

五步走完，终端里出现 0~1 的浮点数，代表 RAG 系统在各个维度上的表现。后续每次改 prompt、换模型、调检索参数，跑一遍就能看到这些数字的变化——变好了还是变差了，一目了然。

RAGAS 能跑了，但 5 个指标各自怎么算的？`faithfulness` 怎么把回答拆成 claims、怎么让 judge 判 supportable？`answer_correctness` 里 0.75 × claim F1 + 0.25 × similarity 的权重是什么意思？`context_recall` 和 `faithfulness` 为什么说是对偶关系？两个指标同时低意味着什么、同时高又意味着什么？

> Source: https://t.zsxq.com/25Iqw
> Resolved: https://articles.zsxq.com/id_86wgu6kg9j5x.html
