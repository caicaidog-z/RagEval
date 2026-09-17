# 《AI大模型Ragent项目》——RAGAS的坑

上一篇把 5 个指标的算法拆透了，配对读法也讲了——RAGAS 不再是黑盒。但拿到分数之后还有一个问题：这些数字本身靠不靠谱？

跑两次同样的数据，faithfulness 一次 0.85 一次 0.80，差了 5 个点——是系统变了还是 RAGAS 自己在抖？judge 模型跟被评模型用同一个 provider，分数会不会偏高？有几条样本总是冒 NaN 怎么办？跑一次完整评测到底要花多少钱？

这篇讲五个实操坑和对策。知道这些坑，才能正确地使用 RAGAS 的分数——不被假信号误导，也不因为噪声而错过真问题。

## 坑一：Judge 同源偏置

### 1. 现象和原因

被评模型和 judge 模型用同一个 provider 的同族模型时（比如被评 gpt-5.4 + judge gpt-5.4-mini，或被评 Qwen-72B + judge Qwen-32B），RAGAS 分数会偏高。

原因不复杂：同族模型在预训练阶段用了相似的数据和对齐策略，对相同风格的输出有共同的偏好。judge 更倾向于认可同族模型的表达方式——同样一个回答，GPT judge 给 GPT 生成的回答打分，会比给 Qwen 生成的回答高几个百分点。

这不是 bug，是 LLM-as-judge 的固有特性，学术上叫 self-preference bias。

**有多严重**：在学术 benchmark 上，同源 vs 异源 judge 的分数差异通常在 3~8%。不是致命的偏差，但足以影响 A/B 测试的结论。如果你改了一版 prompt，分数提了 4%，但其中 5% 是同源偏置带来的，那实际上是退化了——你却以为改进了。

### 2. 本项目的状态

Ragent 默认用 Qwen 系列模型做问答生成，judge 用 gpt-5.4-mini。两者不是同族（Qwen vs GPT），同源偏置风险较低。但如果被评切到 GPT 模型，就会出现同族问题。

### 3. 对策

**最佳做法**：被评和 judge 用不同族模型。GPT 生成的回答用 Qwen 评，Qwen 生成的回答用 GPT 评。

**退而求其次**：固定一个 judge 模型（比如始终用 gpt-5.4-mini），不管被评模型怎么换。绝对分数可能偏，但不同版本之间的相对变化是公平的——因为偏置方向和幅度保持一致。

**底线意识**：知道这个偏差存在就行。不要过度解读 2~3% 的分数差异，尤其是在换被评模型的前后。

> 同源偏置影响的是绝对分数的可信度，不影响同一 judge 下不同版本之间的相对比较。所以固定 judge 是性价比最高的对策——绝对值可能虚高，但趋势是真的。

## 坑二：单跑方差

### 1. 现象和量级

同样的评估集、同样的 runs 文件、同样的 judge 模型，RAGAS 跑两次分数不一样。faithfulness 第一次 0.847，第二次 0.823，差了 2.4 个百分点。

在本项目 18 条可评样本上，单次跑的方差通常在 3~5% 以内。faithfulness 0.85 和 0.80 之间的差异很可能只是噪声，不代表系统真的变好或变差了。

### 2. 原因

RAGAS 内部每次调 judge 都是独立的 LLM 推理。即使设了 `temperature=0`，不同次调用返回的内容也可能有微妙差异——API 侧的 batching、量化精度、甚至请求排队顺序都会影响。

具体到指标层面：拆 claims 时可能多拆一个或少拆一个（“保修 1 年，自购买之日起算”拆成 1 个还是 2 个 claims，judge 两次的判断可能不同），判 supportable 时边界 case 可能这次判 yes 下次判 no。这些微小差异累加起来就是方差。

### 3. 本项目的应对：多轮取均值

```
python -m eval rag score --ragas-n 3
```

三轮独立评测，按样本取均值。每条样本的每个指标有 3 个独立分数，取均值后再汇总。单条样本的随机波动被平均掉了。

关键实现：三轮是并发跑的，不是串行——

```
# ragas_judge.py: compute() 里的并发多轮
if n_runs <= 1:
    dfs = [_run(evaluable, ...).to_pandas()]
else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_runs) as ex:
        futures = [ex.submit(_run, evaluable, ...) for _ in range(n_runs)]
        dfs = [f.result().to_pandas() for f in futures]
```

并发三轮，时间不会翻 3 倍，大约多 50%~80%（受 API 并发限流影响）。

**为什么不用 OpenAI API 的**`n`**参数**：OpenAI Chat API 支持 `n=3` 让一次请求返回 3 个候选回答。但这 3 个候选共享同一次请求的上下文，不算真正独立的采样。`--ragas-n 3` 是发 3 次完全独立的请求，每次从头推理，方差压制效果更好。

**上线门槛建议：**

| 比较方式 | 门槛 | 说明 |
| --- | --- | --- |
| 单次跑，改造前 vs 改造后 | 差 ≥ 5% 才认定有效 | 低于 5% 可能是噪声 |
| --ragas-n 3 取均值后比较 | 差 ≥ 3% 才认定有效 | 均值更稳，门槛可以低一些 |
| 看单条样本变化 | 不建议 | 单条方差太大，没有统计意义 |

## 坑三：中文场景下的 NaN

### 1. 现象

跑完 RAGAS，某几条样本的 `faithfulness` 或 `context_recall` 是 NaN。不是所有样本都 NaN，就偶发几条。其他指标可能正常，就某一两个指标出问题。

### 2. 原因

RAGAS 内部调 judge 拆 claims / statements 时，要求返回结构化的 JSON。但 RAGAS 的默认 prompt 是英文写的，judge 在处理中文内容时偶尔会出三种情况：

- 在 JSON 之外加一段中文解释——"以下是我的分析：{...}"
- 用中文引号（""）替代英文引号（""），导致 JSON 解析失败
- 返回非标准格式——"claims: [...]" 而不是纯 JSON 数组

RAGAS 的解析器碰到这些情况直接返回 NaN。

### 3. 五层防御

本项目在 `ragas_judge.py` 里做了五层 fallback，逐层兜底：

| 层 | 策略 | 何时触发 |
| --- | --- | --- |
| 1 | JSON mode 强制输出合法 JSON | 默认启用 |
| 2 | 关闭 JSON mode，回退到 LLM 原生输出 | 第 1 层整批失败 |
| 3 | 逐条 eval，隔离问题样本 | 第 2 层整批失败 |
| 4 | NaN 样本逐条重试（超时放宽到 1200s） | 全量跑完后仍有 NaN |
| 5 | RunConfig(max_retries=2) 每次 LLM 调用内部重试 | 贯穿始终 |

前三层是 `_run()` 函数内部的 fallback：

```
# ragas_judge.py: _run() 的三层 fallback

# 第一层：batch + JSON mode
try:
    return _do_eval(records, use_json_mode=True)
except Exception as _e:
    batch_exc = _e

# 第二层：batch 无 JSON mode
try:
    return _do_eval(records, use_json_mode=False)
except Exception as _e2:
    ...

# 第三层：逐条 eval，隔离问题样本
for i, r in enumerate(records):
    try:
        dfs.append(_do_eval([r], use_json_mode=False).to_pandas())
    except Exception:
        dfs.append(pd.DataFrame({k: [float("nan")] for k in RAGAS_METRIC_KEYS}))
```

第一层用 `response_format: json_object` 强制 judge 输出合法 JSON，能解决大部分中文引号和格式问题。整批失败才退到第二层。第三层逐条跑的好处是：一条样本有问题不会拖垮整批——问题样本标 NaN，其他样本照常拿分。

第四层在 `compute()` 里：全量跑完后，对仍然是 NaN 的样本单独重试一次，超时从默认 900s 放宽到 1200s——

```
# ragas_judge.py: NaN 样本逐条重试
if failed:
    for qid, k in sorted(failed):
        record = record_map[qid]
        try:
            retry_df = _run([record], ..., timeout=1200).to_pandas()
            fv = float(retry_df.iloc[0].get(k))
            if fv == fv:  # not NaN
                per_metric[k][qid] = fv
        except Exception:
            pass
```

经过五层防御，实测 NaN 率控制在 5% 以下。偶尔剩的 NaN 样本在最终汇总时跳过（不参与均值计算），不会拉偏整体分数。

### 4. 要不要自定义中文 prompt

能不动就别动。RAGAS 的内部 prompt 跟算法深度耦合——改了 prompt，拆 claim 的粒度可能变了，同样一段 response 拆出 3 个 claims 还是 5 个 claims 会直接影响分数。改完之后跟之前版本的分数就不可比了。

上面的 fallback + retry 策略已经把 NaN 率控制在可接受范围。只有当 NaN 率 > 10% 且集中在同一个指标（比如 `faithfulness` 每次都有大量 NaN）时，才值得考虑覆盖默认 prompt——代价是后续 RAGAS 版本升级时需要手动同步你的自定义 prompt。

## 坑四：成本失控

### 1. 成本构成

RAGAS 的每个指标都需要多次 judge 调用。粗算本项目的开销：

| 环节 | 调用次数 |
| --- | --- |
| 5 指标 × 每条约 3 次 judge 调用 | ≈ 15 次 / 条 |
| 20 条评估集（18 条可评） × 15 次 | ≈ 270 次 / 轮 |
| --ragas-n 3 三轮取均值 | ≈ 810 次 |
| NaN 重试（约 5% 样本） | ≈ 30~50 次 |
| 总计 | ≈ 850~900 次 / 完整评测 |

按 gpt-5.4-mini 定价，一次完整评测大约 1020 元人民币（没有开多轮取均值）。听起来不多，但开发阶段一天跑 5 次就是 50100 元，一周下来几百元。

如果后续评估集扩到 150 条 + 3 轮，单次就是 6750+ 次调用，一次评测上百元。

### 2. 控制手段

| 手段 | 命令 | 效果 |
| --- | --- | --- |
| CI 跳过 RAGAS | --skip-ragas | 自建指标零成本，日常提交不烧钱 |
| 限制评估条数 | --ragas-limit 5 | 只评前 5 条，快速验证能不能跑通 |
| 单轮不取均值 | 不加 --ragas-n | 省 2/3 的调用量，开发阶段看趋势够用 |
| 复用 runs 文件 | python -m eval rag score | 不重跑 runner，只重算指标。改 judge 配置后复用同一份 runs，不额外调被评接口 |

### 3. 分层策略

| 场景 | 建议跑法 | 预估成本 |
| --- | --- | --- |
| 每次 PR | --skip-ragas（纯自建指标） | 零 |
| 开发调试 | --ragas-limit 5（5 条快速验证） | < 2 元 |
| 每日定时 | 完整 20 条单轮 | 10~20 元 |
| Release 前 | 20 条 × --ragas-n 3 | 30~50 元 |
| Prompt / 模型大改 | 完整 + 3 轮 + 人工抽查 | 30~50 元 + 人力 |

核心原则：**自建指标当日常闸门（零成本），RAGAS 只在需要语义级评估时按需开启**。不要每次 commit 都跑 RAGAS。

## 坑五：Judge 误判与人工校准

### 1. 现象

RAGAS 跑出来某条样本 `faithfulness` 只有 0.5，你打开一看——回答完全忠实于 contexts，没有任何编造。或者反过来，某条样本 `answer_correctness` 给了 0.9，但人工一看回答明显遗漏了关键信息。

这不是偶发现象。LLM-as-judge 本质上是用一个大模型去评判另一个大模型的输出，判断能力受限于 judge 模型自身的理解力。在中文场景下尤其容易出问题：

- **同义改写误判**：7 天无理由退货和七天内可无理由退，judge 可能判为不同 claim，导致 F1 偏低
- **枚举顺序敏感**：reference 列了 A、B、C 三个条件，response 列了 C、A、B，judge 有时会认为不完全匹配
- **隐含信息遗漏**：response 说保修期内免费维修，隐含了有保修期这个前提，但 judge 可能不把这算作一个独立 claim
- **格式化答案干扰**：response 用了表格或编号列表，reference 是纯文本段落，judge 在拆 claims 时粒度不一致

这些误判在单条样本上可能差 0.2~0.3，直接影响均值——18 条样本里 2 条被误判，整体分数就漂了好几个百分点。

这里给大家拆了两个我手动调校过的 JSON，大家可以复制出来看看，打的分很低，不超过0.5，但是回答内容确没有什么大问题。

```
{"query_id": "S12-01", "user_input": "智能音箱能控制扫地机吗？", "reference": "可以。前提条件：① 小爱音箱（Play 增强版 / Pro 等）和石头扫地机器人已添加到同一个米家家庭；② 两台设备都在线；③ 设备命名清晰，建议改为「房间 + 设备」格式（例如「客厅扫地机」），避免「设备1」「设备2」这类难以语音识别的名称。语音指令示例：「小爱同学，启动客厅扫地机」「小爱同学，让扫地机回充」。也可在米家 APP > 智能 > 自动化中创建场景（如「清洁模式」），用语音一键执行多个动作。控制失败时先在米家 APP 检查目标设备在线状态和名称是否重复。具体按房间清扫依赖扫地机地图已保存且房间已命名。", "reference_doc_ids": ["AUTO_GUIDE_002", "MANUAL_SPK_001", "MANUAL_VAC_001"], "reference_doc_ids_nice": [], "intent_l1": "SUPPORT", "intent_l2": "S12_生态联动", "difficulty": "medium", "requires_rag": true, "response": "可以，前提是先把**小爱音箱**和**扫地机**都添加到同一个**米家家庭**，并给扫地机设置清晰名称。\n\n### 一、怎么设置1. 打开**米家 APP**。\n2.先添加**小爱音箱**和**扫地机**。\n3. 给设备命名为“房间 +设备”，例如“客厅扫地机”。\n4.进入**米家 APP > 智能 > 自动化**，可以创建联动或手动场景。\n5. 场景保存后，可对音箱说“小爱同学，执行清洁模式”。\n\n### 二、怎么语音控制在米家 APP 中先添加灯、插座、净化器、扫地机等设备后，对音箱说“小爱同学”唤醒，再说控制指令。\n\n结合现有说明，扫地机属于可被控制的设备，**石头扫地机器人系列**支持的联动能力包括：**清扫、回充、区域清扫**，但**具体能力以 App 地图和机型为准**。\n\n### 三、控制失败时看这几项- 确认**小爱音箱在线**\n- 检查扫地机是否在线- 检查设备名称是否重复- 如果扫地机无法按房间清扫，通常是**地图未保存或房间未命名**，需要在扫地机 App 中保存地图和房间**如果你**只是想先测试是否能控制，可以先把扫地机命名为“客厅扫地机”，再尝试让音箱执行与扫地机相关的语音控制。", "thinking": null, "latency_ms": 11355, "first_token_ms": 8522, "final_status": "success", "error": null, "conversation_id": "2058198989905367040", "task_id": "2058198989905367041", "retrieved_doc_ids": ["AUTO_GUIDE_002", "MANUAL_SPK_001", "NET_GUIDE_001", "MANUAL_VAC_001", "MANUAL_VAC_003"], "retrieved_doc_ids_raw": ["AUTO_GUIDE_002", "MANUAL_SPK_001", "NET_GUIDE_001", "MANUAL_VAC_001", "MANUAL_VAC_003"], "retrieved_chunk_ids": ["2055926512789331969", "2055926515897311233", "2055926512986464257", "2055926516086054913", "2055926516862001154"], "retrieved_contexts": ["xxx内容较多，省略......"], "retrieved_context_doc_ids": ["AUTO_GUIDE_002", "MANUAL_SPK_001", "NET_GUIDE_001", "MANUAL_VAC_001", "MANUAL_VAC_003"], "intent_pred": "S12_生态联动", "intent_pred_all": ["S12_生态联动"], "has_kb": true, "has_mcp": false, "trace_id": null}
{"query_id": "S8-01", "user_input": "扫地机首次使用，如何开机？", "reference": "长按主机电源键，指示灯亮起后即完成开机。首次使用前建议将主机放回充电座或基站充满电（T7 约 6 小时，T7S Plus 约 3 小时，G10 / G10S Pro 以基站回充显示为准）。", "reference_doc_ids": ["MANUAL_VAC_001"], "reference_doc_ids_nice": [], "intent_l1": "SUPPORT", "intent_l2": "S8_操作指引", "difficulty": "easy", "requires_rag": true, "response": "首次使用时，**长按主机电源键**，指示灯亮起后就完成开机。\n\n如果是第一次用，通常还建议先把主机放回充电座或基站充满电再使用：**T7充满约需6 小时，T7S Plus约需3 小时，G10 / G10S Pro以基站回充显示为准**。", "thinking": null, "latency_ms": 9527, "first_token_ms": 9521, "final_status": "success", "error": null, "conversation_id": "2059655497528754176", "task_id": "2059655497528754177", "retrieved_doc_ids": ["MANUAL_VAC_001", "MANUAL_VAC_002", "MANUAL_VAC_003", "MANUAL_AIR_001", "MANUAL_HEALTH_001"], "retrieved_doc_ids_raw": ["MANUAL_VAC_001", "MANUAL_VAC_002", "MANUAL_VAC_003", "MANUAL_AIR_001", "MANUAL_HEALTH_001"], "retrieved_chunk_ids": ["2058461174980194305", "2058461176070713345", "2058461177186398210", "2058461160698589185", "2058461166193127425"], "retrieved_contexts": ["xxx内容较多，省略......"], "retrieved_context_doc_ids": ["MANUAL_VAC_001", "MANUAL_VAC_002", "MANUAL_VAC_003", "MANUAL_AIR_001", "MANUAL_HEALTH_001"], "intent_pred": "S8_操作指引", "intent_pred_all": ["S8_操作指引"], "has_kb": true, "has_mcp": false, "trace_id": null}
```

### 2. 对策：人工抽查 + `per_sample.csv` 人工列校准

RAGAS 分数不是最终裁决，而是一个自动化的初筛信号。关键发布决策不能只看自动分数，需要人工抽查低分样本并校准。

本项目在 `per_sample.csv` 里为每个 RAGAS 指标自动生成了一个 `*_manual` 空列，专门用于人工校准。

#### 2.1 per_sample.csv 的结构

跑完 `python -m eval rag report` 后，`eval/reports/v1_xxx/per_sample.csv` 长这样：

```
query_id, ..., faithfulness, faithfulness_manual, answer_relevancy, answer_relevancy_manual, ...
F1-01,   ..., 0.5000,      ,                     0.8234,           ,                        ...
S1-03,   ..., 1.0000,      ,                     0.9123,           ,                        ...
```

每个 RAGAS 指标后面紧跟一个 `_manual` 列，初始为空。人工审查后在这一列填入校准分数。

#### 2.2 人工审查流程

**Step 1：定位低分样本**

用 Excel 或任何表格工具打开 `per_sample.csv`，按 `faithfulness` 或 `answer_correctness` 升序排，找到分数最低的 3~5 条。

**Step 2：人工判分**

打开对应样本的 `response`、`retrieved_contexts`、`reference`（在 runs JSONL 文件里），人工判断 RAGAS 的打分对不对。

如果 RAGAS 给 `faithfulness` 打了 0.5 但你判断应该是 0.9（回答确实忠实于上下文），在 `faithfulness_manual` 列填入 `0.9`。

填写格式灵活——支持小数（`0.9`）、百分数（`90%`）、整数百分比（`90`，会自动除以 100）。

**Step 3：重跑 report**

```
python -m eval rag report
```

report 命令会自动检测 `per_sample.csv` 里的人工列。策略是**人工优先，空值回退 RAGAS**——

- `faithfulness_manual` 有值 → 用人工分
- `faithfulness_manual` 为空 → 用原始 RAGAS 分

然后基于校准后的 per_sample 分数重新计算 overall 均值、`by_intent_l1` / `by_intent_l2` 分层均值，重新生成 `report.md` 和 `slides.html`。

#### 2.3 核心代码逻辑

`report/markdown.py` 里的三个关键函数：

```
# 1. 从 per_sample.csv 读取人工列
def load_manual_overrides(per_sample_path: Path) -> dict[str, dict[str, float]]:
    """没有文件或没有人工列时返回空。"""
    ...
    for row in reader:
        for key, col in manual_cols.items():
            score = _parse_manual_score(row.get(col))
            if score is not None:
                overrides[key][qid] = score
    return overrides

# 2. 人工分覆盖 RAGAS 分，重算均值
def apply_manual_overrides(records, metrics, overrides) -> list[MetricResult]:
    """人工列有值用人工，否则用原 RAGAS。重算 overall / by_intent。"""
    per_sample = dict(metric.per_sample)
    for qid, score in manual_scores.items():
        per_sample[qid] = score  # 覆盖
    # 基于覆盖后的 per_sample 重算均值和分层
    ...

# 3. report 命令里的调用
manual_overrides = load_manual_overrides(per_sample_path)
report_metrics = apply_manual_overrides(records, metrics, manual_overrides)
```

整个流程不需要重跑 RAGAS（不额外调 judge API），只是在 report 阶段用人工分替换自动分再重算——零成本，秒出。

### 3. 什么时候需要人工校准

不是每次跑完都要逐条审。建议在以下场景做人工抽查：

| 场景 | 抽查范围 |
| --- | --- |
| Release 前 | 所有 answer_correctness < 0.6 的样本 |
| 分数异常波动 | 跟上次相比下降 > 5% 的指标，看低分样本是真退化还是误判 |
| 高风险意图 | 售后政策、保修条款、价格类样本，不管分数高低都抽查 |
| 首次跑 RAGAS | 全量抽查一遍，建立对 judge 打分准确度的基线认知 |

> 大部分样本的自动评分是靠谱的，只有边界 case 需要人工介入。校准的结果会持久保留在 `per_sample.csv` 里——后续重跑 report 时自动生效，不需要重新审。

## 跑 RAGAS 前的体检清单

每次跑 RAGAS 之前过一遍这张表，能省掉大部分跑了半小时发现 API key 过期的尴尬：

| # | 检查项 | 怎么查 | 不通过的后果 |
| --- | --- | --- | --- |
| 1 | API key 有效 | echo $AIHUBMIX_API_KEY 非空 | 直接报错退出 |
| 2 | judge 模型可用 | 用 curl 或代码发一次 chat completions 请求 | 全部样本 NaN |
| 3 | embedding 模型可用 | 用 curl 或代码发一次 embeddings 请求 | answer_relevancy 报错 |
| 4 | judge 和被评不同族 | 对比 JUDGE_MODEL 和 Ragent 配置的 chat 模型 | 同源偏置（分数偏高） |
| 5 | 评估集有 ground_truth | 查 eval_set_v1.jsonl 的 ground_truth 字段非空 | answer_correctness / context_recall 算不了 |
| 6 | runs 里 contexts 非空 | 检查 JSONL 里 retrieved_contexts 非空率 | 大量样本被 skip |
| 7 | 网络稳定 | ping aihubmix.com 或 curl 测试 | 超时导致 NaN 或中途失败 |
| 8 | 预留足够时间 | 20 条 × 3 轮 ≈ 15~30 分钟 | 中途超时被 kill |

> 第 6 项容易忽略。如果 runner 跑出来的 runs 文件里大量样本 `retrieved_contexts` 为空（被 reject 或走了兜底），RAGAS 会把它们全部跳过。跑完发现可评 5 条，跳过 15 条——不是 RAGAS 的问题，是被评系统的检索链路有问题，回头查 Ragent。

## 小结与下一篇预告

五个坑总结一下：

- **同源偏置**：judge 和被评同族会偏高 3~8%。固定 judge 保证相对比较可信，不过度解读绝对值
- **单跑方差**：3~5% 是正常噪声。用 `--ragas-n 3` 压制，差 ≥ 3% 才认定有效变化
- **中文 NaN**：五层 fallback 把 NaN 率控制在 5% 以下。能不改默认 prompt 就别改
- **成本失控**：分层策略——CI 不跑、开发限 5 条、Release 全量 3 轮
- **Judge 误判**：低分样本人工抽查，在 `per_sample.csv` 的 `*_manual` 列填入校准分，重跑 `python -m eval rag report` 自动生效

RAGAS 不是跑出数字就结束。知道数字的局限性——哪些差异是真的、哪些是噪声、哪些是偏置、哪些是误判——才能正确使用它做决策。

指标都算完了，坑也避了。接下来的问题是：怎么把这些数字呈现出来？一堆 `_scores.json` 和 JSONL 文件没人愿意看。需要分层看板、失败样例归因、可以打开浏览器直接演示的报告。

> Source: https://t.zsxq.com/8YZhT
> Resolved: https://articles.zsxq.com/id_02tlmru5ruml.html
