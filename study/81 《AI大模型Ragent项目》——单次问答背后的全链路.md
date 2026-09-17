# 《AI大模型Ragent项目》——单次问答背后的全链路

上一篇讲了初始化——用三个 Python 脚本把 Ragent 从空白搭到了可评测状态：4 个知识库、115 篇已分块的文档、30 个意图节点。评估集也准备好了，20 条 query 等着跑。

接下来的问题是：怎么把这 20 条 query 逐条喂进 Ragent，把检索结果、生成答案、性能数据**一次性收齐**？

拿评估集里的一条样本来说——“预算 3000 左右，有没有拍照比较好的手机推荐？”。跑完之后，我需要知道四件事：召回了哪几篇文档（算 Hit@K）、命中了哪个意图叶子（算意图准确率）、模型回了什么（算 faithfulness、answer_correctness 等）、用户等了多久才看到第一个字（算 TTFT P95）。

问题在于，Ragent 的生产接口 `/rag/v3/chat` 是 SSE 流式输出，一个字一个字蹦出来——拿得到答案和首字耗时，但拿不到中间产物。召回了哪些 chunk、分到了哪个意图，SSE 流里没有这些数据。

一个接口不够用，runner 的核心设计就是**一条 query 跑两个接口**。

## 为什么一个接口不够用

先看评测需要哪些数据，以及生产 SSE 接口能不能给：

| 评测需要的数据 | 用来算什么指标 | 生产 SSE 能给吗 |
| --- | --- | --- |
| 召回了哪些 docId / chunkId | Hit@K / Recall@K / MRR | ❌ 不暴露中间产物 |
| 意图分类结果 | 意图 Top-1 准确率 | ❌ |
| chunk 文本内容 | RAGAS faithfulness / context_recall | ❌ |
| 模型完整回复 | faithfulness / answer_correctness | ✅ 拼接 delta 即可 |
| 首字到达时间 | TTFT P95 | ✅ 需要自己打点 |
| 总耗时 | 性能基线 | ✅ |

六项里有三项拿不到。所以需要一个旁路接口来补齐检索证据，跟 SSE 接口配合，一次性收齐所有数据。

## Part 1：被测侧——评测旁路接口 `/rag/eval`

### 1. 只跑检索，不跑 LLM

`EvalController` 是专门为评测写的旁路接口，核心思路很简单：**复用生产链路的检索能力，但跳过 LLM 生成**。

```
@RestController
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "app.eval", name = "enabled", havingValue = "true")
public class EvalController {

    private final QueryRewriteService queryRewriteService;
    private final IntentResolver intentResolver;
    private final RetrievalEngine retrievalEngine;

    @GetMapping("/rag/eval")
    public Result<EvalResponse> chat(@RequestParam String question) {
        long start = System.currentTimeMillis();
        RewriteResult rewriteResult = queryRewriteService.rewriteWithSplit(question, List.of());
        List<SubQuestionIntent> subIntents = intentResolver.resolve(rewriteResult);
        RetrievalContext rc = retrievalEngine.retrieve(subIntents, searchProperties.getDefaultTopK());
        return Results.success(buildResponse(rc, subIntents, System.currentTimeMillis() - start));
    }
}
```

三行核心调用——改写拆分 → 意图识别 → 检索引擎——跟生产链路用的是同一套 Service。区别在于拿到检索结果就打包返回了，不调大模型、不走 SSE。

`@ConditionalOnProperty` 是开关控制：配置 `app.eval.enabled=true` 才注册这个 Bean，生产环境不配就不存在，零开销。

返回的 `EvalResponse` 包含评测需要的全部检索证据：

| 字段 | 说明 |
| --- | --- |
| retrievedDocIds | 召回的业务文档 ID（已去重，按首次出现顺序） |
| retrievedChunkIds | 召回的 chunk 主键（已去重） |
| retrievedContexts | 召回的 chunk 文本（与 chunkIds 顺序对应） |
| retrievedContextDocIds | chunk 维度的业务 docId（与 contexts 一一对应，保留 null，不去重） |
| intentLeafIds | 每个子问题 top-1 的意图叶子节点 ID |
| hasKb / hasMcp | 是否走了 KB 检索 / MCP 工具 |
| latencyMs | 接口耗时 |

### 2. `chunkId → docId` 两跳映射

评估集里的 `expected_doc_ids` 用的是业务码（如 `FAQ_VAC_001`），但向量库检索出来的 chunk 只有 ragent 内部的雪花 ID。从 chunk 到业务码要跳两次数据库：

```
chunkId
  → t_knowledge_chunk.doc_id    （第一跳：chunk 对应哪个文档的内部 ID）
  → t_knowledge_document.doc_name → 剥 .md 后缀
  → 业务码（如 FAQ_VAC_001）     （第二跳：内部 ID 对应哪个业务码）
```

代码里分三步批量查询：先拿 chunkId 查到内部 docId，再拿内部 docId 查到 docName，最后 `stripExtension()` 剥后缀。两次数据库查都是批量 `selectByIds`，不会逐条查。

这里有两个维度的 docId 列表，用途不同。用一个具体例子来说明——假设一条 query 检索回来 5 个 chunk，来自 3 篇文档：

```
chunk[0]  ← 来自 PROD_PHONE_001
chunk[1]  ← 来自 PROD_PHONE_001（同一篇文档的另一段）
chunk[2]  ← 来自 POLICY_RETURN_002
chunk[3]  ← 来自 PROD_PHONE_003
chunk[4]  ← 映射失败，查不到对应文档
```

两个列表分别长这样：

| 字段 | 值 | 长度 |
| --- | --- | --- |
| retrievedContextDocIds | [PROD_PHONE_001, PROD_PHONE_001, POLICY_RETURN_002, PROD_PHONE_003, null] | 5（跟 chunk 数一样） |
| retrievedDocIds | [PROD_PHONE_001, POLICY_RETURN_002, PROD_PHONE_003] | 3（去重 + 去 null） |

`retrievedDocIds` 给 Hit@K / Recall@K 这类文档级指标用。这些指标回答的问题是：期望文档有没有被召回？只要 `PROD_PHONE_001` 出现在列表里就算命中，不管它贡献了 1 个 chunk 还是 5 个 chunk。所以去重成 3 个文档 ID 就够了。

`retrievedContextDocIds` 给 RAGAS 的 context_precision 这类 chunk 级指标用。这个指标回答的是另一个问题：召回的每一段文本（chunk）对回答有没有帮助？它会逐个 chunk 评判，而且排在前面的 chunk 权重更高。chunk[0] 和 chunk[1] 虽然都来自 `PROD_PHONE_001`，但一个可能讲的是屏幕参数，另一个讲的是价格——对不同问题的帮助完全不同，必须分开评判。所以这个列表要跟 `retrievedContexts` 严格一一对应：5 个 chunk 就是 5 个元素，不去重、不合并。

### 3. 旁路的代价：漂移风险

需要坦诚交代一个设计妥协：`/rag/eval` 和 `/rag/v3/chat` 走的是不同代码路径。

`/rag/eval` 直接组合了三个 Service（改写、意图、检索），而生产链路在 `RAGChatService.streamChat()` 里可能有额外的后处理（当前是没有的）——比如跨通道的结果合并、基于分数的二次过滤。这些后处理不会反映在旁路结果里。

这是有意识的权衡：完全复刻生产链路的 shadow 模式维护成本太高，每次生产代码改动都要同步一份到旁路。当前的做法是足够近似——改写、意图、检索主体逻辑完全一致，并且没有污染主流程，算是较优的解决方案。

## Part 2：评测侧——Runner 的双接口聚合

### 1. 一条 query 的完整旅程

一条 query 从评估集到 `EvalRecord`，经过以下完整链路：

![图片.png](assets/81-01.png)

Runner 对每条 query 依次调两个接口，把结果合并成一条 `EvalRecord` 写进 JSONL 文件。后续 score 阶段读这个文件算指标，不需要再调任何接口。

### 2. 自己写 SSE 解析器

Runner 里自己实现了 `parse_sse_stream()`，没用 `requests` 库自带的 `iter_lines()`。原因是：`iter_lines()` 在事件分隔符 `\n\n` 跨 HTTP chunk 边界时，会把空行吞掉，导致中间事件丢失。

自写解析器的核心逻辑：

```
def parse_sse_stream(byte_iter: Iterator[bytes]) -> Iterator[tuple[str, str]]:
    buffer = ""
    for chunk in byte_iter:
        buffer += chunk.decode("utf-8", errors="replace")
        while True:
            # 找 \r\n\r\n 或 \n\n，兼容两种换行风格
            idx_crlf = buffer.find("\r\n\r\n")
            idx_lf = buffer.find("\n\n")
            if idx_crlf == -1 and idx_lf == -1:
                break
            # 取更靠前的分隔符，切出一个完整事件块
            event_block = buffer[:idx]
            buffer = buffer[idx + sep_len:]
            yield from _parse_event_block(event_block)
    # 流结束后冲刷 buffer 残留
    if buffer.strip():
        yield from _parse_event_block(buffer)
```

思路很直接：把收到的字节流拼进 buffer，在 buffer 里找 `\n\n` 分隔符，找到就切出一个完整事件块交给 `_parse_event_block()` 解析。事件块里的 `event:` 字段设事件名，`data:` 字段是载荷。

Ragent SSE 流推送的事件类型及 runner 的处理：

| 事件名 | 含义 | Runner 处理 |
| --- | --- | --- |
| meta | 连接元数据 | 取 conversationId、taskId |
| message（type=response） | 答案增量 delta | 拼接 response 文本 + TTFT 打点 |
| message（type=think） | 思考链增量 delta | 拼接 thinking 文本 |
| finish | 正常结束 | final_status 记为 success |
| reject | 拒绝回答 | final_status 记为 refused，记录原因 |
| cancel | 用户取消 | final_status 记为 cancelled |
| done | 流终止标记 | break 退出解析循环 |

### 3. TTFT 打点：只算答案首字

TTFT（Time To First Token）是评测性能数据里口径最精确的一个指标。看关键四行：

```
start = time.time()
# ... SSE 流处理循环 ...
if delta_type == "response":
    if state["first_token_ms"] is None and content:
        state["first_token_ms"] = int((time.time() - start) * 1000)
    state["response"] += content
```

三个条件缺一不可：

- `delta_type` 等于 `response`——只算 `type=response` 的 delta，`think` 类型不算。用户在意的是看到答案的时间，不是看到思考过程的时间
- `content` 非空——空字符串 delta 不算。有些模型会先发一个空 delta 占位
- `first_token_ms` 还没记录——只记第一次。后续 delta 不更新

举个实际场景：模型开了深度思考模式，先输出一段 thinking（“让我分析一下你的需求...”），然后才开始输出 response（“推荐您考虑...”）。thinking 首字可能在 200ms 就到了，但 response 首字要等到 1500ms。TTFT 记的是 1500ms——这才是用户体感上等答案等了多久。

如果整个流都没有 `response` 类型的 delta（比如被 reject 了，或者出了异常），`first_token_ms` 是 `None`，后续算性能指标时会跳过这条样本。

### 4. `build_record`：三段合一

两个接口的数据拿齐之后，`build_record()` 把它们跟评估集的静态字段合并成一条 `EvalRecord`。这个 dataclass 分三段：

**第一段——从评估集原样复制**：

| 字段 | 说明 |
| --- | --- |
| query_id / user_input | 样本标识和原始问题 |
| reference | 标准答案（ground_truth） |
| reference_doc_ids / reference_doc_ids_nice | 必须命中的文档 / 可选扩展文档 |
| intent_l1 / intent_l2 | 标注的意图分类 |
| difficulty / requires_rag | 难度和是否需要 RAG |

第二段——从 `/rag/v3/chat` SSE 流拿到：

| 字段 | 说明 |
| --- | --- |
| response | 模型完整回复（拼接所有 response delta） |
| thinking | 思考链文本（如有） |
| first_token_ms | TTFT，答案首字到达时间 |
| latency_ms | 总耗时（从发请求到流结束） |
| final_status | success / refused / error / cancelled / unknown |
| conversation_id / task_id | 会话和任务标识 |

**第三段——从 `/rag/eval` JSON 接口拿到**：

| 字段 | 说明 |
| --- | --- |
| retrieved_doc_ids | 召回文档的业务码（已映射、已去重） |
| retrieved_doc_ids_raw | 召回文档的 ragent 内部 ID |
| retrieved_chunk_ids | 召回的 chunk ID |
| retrieved_contexts | 召回的 chunk 文本 |
| retrieved_context_doc_ids | chunk 维度 docId（保留 null） |
| intent_pred | 预测的意图叶子（取第一个非空） |
| has_kb / has_mcp | 是否走了 KB / MCP |

合并过程中有两个关键转换：

- **ragent doc_id → 业务码**：`/rag/eval` 返回的是 ragent 内部 ID，需要通过 `doc_id_map.json` 的反向映射（ragent_doc_id → 业务码）转成评估集能对比的业务码
- `intent_pred` 取值：多子问题场景下可能有多个意图结果，取第一个非空叶子——`next((c for c in intent_codes if c), None)`

最终一条 `EvalRecord` 序列化成一行 JSON，写入 `runs/v1_<timestamp>.jsonl`。20 条 query 就是 20 行。

### 5. 并发控制与运行

运行 runner 的常见场景：

```
# 环境变量配好
export RAGENT_BASE_URL=http://localhost:9090/api/ragent
export RAGENT_USERNAME=admin
export RAGENT_PASSWORD=admin

# 跑前 5 条冒烟测试
python -m eval rag run --limit 5

# 跑全部 20 条
python -m eval rag run --limit 20

# 只跑某个意图的样本（排查特定场景）
python -m eval rag run --limit 20 --filter-intent S1_选购推荐

# 4 线程并行跑（快，但结果顺序不保证）
python -m eval rag run --limit 20 --workers 4
```

实际运行输出：

```
登录 http://localhost:9090/api/ragent ...
OK

将跑 5 条，落 eval/runs/v1_20260531_143022.jsonl

  [ 1/ 5] S1-01    '预算3000元左右，有没有拍照比较好的手机推荐？'  ->   success  docs=3  ctx=5  resp=  486  ttft=  152ms  total=  1853ms
  [ 2/ 5] S2-01    'iPhone 16 Pro 的屏幕刷新率是多少？'           ->   success  docs=1  ctx=3  resp=  230  ttft=  128ms  total=  1204ms
  [ 3/ 5] S14-01   '买了两周的耳机能退吗？'                      ->   success  docs=2  ctx=4  resp=  352  ttft=  163ms  total=  1567ms
  [ 4/ 5] C1-01    '你好'                                        ->   success  docs=0  ctx=0  resp=   89  ttft=  201ms  total=   634ms
  [ 5/ 5] F1-01    '扫地机器人E003错误码是什么意思？'              ->   success  docs=1  ctx=3  resp=  415  ttft=  145ms  total=  1723ms

完成。统计：{'success': 5, 'refused': 0, 'error': 0, 'cancelled': 0, 'unknown': 0}
产物：eval/runs/v1_20260531_143022.jsonl
```

每行日志能看到关键信息：样本 ID、query 预览、状态、召回文档数、chunk 数、response 长度、TTFT、总耗时。跑出问题的样本一眼就能定位。

并发模式的区别：

| 模式 | 行为 | 适用场景 |
| --- | --- | --- |
| 串行（默认，workers=1） | 逐条跑，每条之间 sleep 0.3s | 日常评测，避开 ragent 并发限流 |
| 多线程（--workers N） | ThreadPoolExecutor 并行调度，写文件用锁保护 | 赶时间跑全量，需要确认 ragent 能扛住并发 |

超时常量也值得了解，特别是排查 runner 卡住的时候：

| 常量 | 值 | 说明 |
| --- | --- | --- |
| SSE_CONNECT_TIMEOUT | 15s | SSE 连接建立超时 |
| SSE_READ_TIMEOUT | 300s | SSE 流读取超时（长答案生成可能要几十秒） |
| EVAL_CONNECT_TIMEOUT | 15s | 评测旁路连接超时 |
| EVAL_READ_TIMEOUT | 180s | 评测旁路读取超时（多通道并行检索可能较慢） |

## 绕不开的两个妥协

写到这里，有必要把两个设计妥协放在一起讲清楚，因为它们会影响你理解评测结果的精度。

**妥协一：旁路漂移。** `/rag/eval` 和 `/rag/v3/chat` 走不同代码路径。旁路直接组合三个 Service 拿检索结果，生产链路还有后处理步骤。如果生产链路加了一段 rerank 逻辑但旁路没跟上，评测拿到的检索证据跟模型实际看到的 chunk 就有偏差。

**妥协二：双检索不一致。** 同一条 query 分别调两个接口，是两次独立检索。Query 改写有随机性、向量近似搜索有浮动、缓存状态可能不同——两次召回的结果不保证完全一致。

这不是 bug，是有意识的工程权衡。如果要消除这两个问题，就得改造生产链路——在 SSE 流里夹带检索中间产物，或者在生产代码里加一层旁路日志落库。代价是侵入生产代码、增加维护负担。当前做法的好处是**生产链路完全不改**，TTFT 和 response 是真实数据，检索证据足够近似。

> 评测不是要完美复现生产环境的每一个细节，而是在可接受的精度范围内，用最低的侵入成本拿到够用的数据。知道边界在哪，比追求完美更重要。

## 小结与下一篇预告

Runner 的核心是一条 query 跑两个接口：

- SSE `/rag/v3/chat`：走真实生产链路，拿 response、thinking、TTFT、总耗时
- JSON `/rag/eval`：走检索旁路，拿 docIds、chunkIds、contexts、intentLeafIds

两个接口的产物加上评估集的静态字段，合并成一条 `EvalRecord`，落 `runs/*.jsonl`。20 条 query 跑完就是 20 行 JSON，后续 score 阶段读这个文件反复算指标，不需要再调任何接口。

数据收齐了，该算指标了。意图分类是整个链路的最上游闸门——意图错了路由就错，路由错了检索目标就错，后面全盘皆输。检索指标（Hit@K / Recall@K / MRR）是纯集合运算，秒级出结果，可以挂 CI。

> Source: https://t.zsxq.com/jCALF
> Resolved: https://articles.zsxq.com/id_so86no0fmlaz.html
