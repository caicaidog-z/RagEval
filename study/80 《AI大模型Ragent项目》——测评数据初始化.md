# 《AI大模型Ragent项目》——测评数据初始化

上一篇讲了评估集的 schema——150 条样本，12 个字段，每个字段为什么存在、被谁消费。评估集设计好了，但 Ragent 还是空的。

评估集里标了 S1-01 应该召回 `GUIDE_PHONE_002`，但如果 Ragent 的知识库里压根没有这篇文档，评测就是空转——检索指标算出来全是 0，不是系统差，是根本没东西可检索。意图树也一样，评估集标了 `S1_选购推荐`，但 Ragent 没建这个意图节点，意图识别就无从谈起。

这篇讲的是评测的基建：怎么用三个 Python 脚本把 Ragent 从空白状态搭到可评测状态——有知识库、有文档、有意图树。

## 初始化总览：从空白到可评测

执行初始化之前，需要先安装 Python 环境，版本不得低于 3.11，参考 [廖雪峰老师 Python 安装教程](https://liaoxuefeng.com/books/python/install/index.html)。

### 1. 三步数据流

初始化分三步，每步依赖上一步的产物，必须按顺序执行：

![plantuml-8.png](assets/80-01.png)

三步跑完后，Ragent 就有了 4 个知识库、115 篇已分块的文档、30 个意图节点，处于可评测状态。

### 2. 运行前配置

跑之前需要配三个环境变量：

```
export RAGENT_BASE_URL=http://localhost:9090/api/ragent
export RAGENT_USERNAME=admin
export RAGENT_PASSWORD=admin
```

所有脚本通过 `POST /auth/login` 获取 token，后续请求带 `Authorization` header 鉴权。

## 初始化第一步：建 4 个知识库

### 1. 为什么按文档类型建库

第一个问题：为什么是 4 个知识库而不是 22 个？

如果按意图切，每个二级意图一个知识库，就要建 22 个 KB。管理成本高不说，很多意图其实共享同一批文档——S14 售后政策和 S15 退换货查的都是政策类文档，S8 操作指引和 S9 配网连接查的都是使用手册。按意图切会导致大量重复灌入。

所以按文档类型切成 4 个：

| KB Key | 名称 | 文档数 | 覆盖场景 |
| --- | --- | --- | --- |
| product | 比特严选-商品库 | 65 | 商品详情、选购指南、对比评测 |
| manual | 比特严选-使用手册库 | 25 | 操作手册、配网指南、APP 指南 |
| policy | 比特严选-政策库 | 15 | 保修、退换货、物流、发票会员政策 |
| faq | 比特严选-FAQ库 | 10 | 故障排查、错误码手册 |

一个意图通过意图树关联到对应的 KB（下一步讲），简洁且灵活。

### 2. 运行脚本与产物

运行命令：

```
python eval/rag/init/create_kbs.py
```

脚本调用 `POST /knowledge-base` 接口逐个创建，产出 `kb_ids.json`——记录每个 KB 的 ragent snowflake ID：

```
{
  "product": {
    "kb_id": "2058461026313089024",
    "name": "比特严选-商品库",
    "collection_name": "kb-product",
    "embedding_model": "qwen-emb-8b"
  }
}
```

> 注意：这个脚本没有幂等性——跑两次会建两套重复的 KB。这是有意为之，知识库创建是一次性操作，不值得为幂等增加复杂度。如果需要重建，先用 `reset_kbs.py` 清理再跑。
>
>
>
> 可以看到 `embedding_model` 的值是 qwen-emb-8b，这个也是 Ragent 项目默认的向量模型。如果调整了 Ragent，记得这里也要调整。

## 初始化第二步：灌入 115 篇文档

### 1. 文档怎么组织

115 篇 Markdown 文档按知识库类型分目录存放：

```
knowledge_base/
├── 01_product/     → product KB（65 篇）
│   ├── detail/       45 篇商品详情
│   └── guide/        20 篇选购指南
├── 02_manual/      → manual KB（25 篇）
│   ├── app/          5 篇 APP 指南
│   ├── network/      5 篇配网指南
│   └── product/      15 篇产品手册
├── 03_policy/      → policy KB（15 篇）
│   ├── warranty/     5 篇保修政策
│   ├── return/       4 篇退换货政策
│   ├── logistics/    3 篇物流政策
│   └── invoice_vip/  3 篇发票会员政策
└── 04_faq/         → faq KB（10 篇）
    ├── error_code/   6 篇错误码手册
    └── trouble/      4 篇故障排查
```

目录名前缀（`01_`、`02_` 等）决定了文档归属哪个知识库。业务码就是文件名去掉 `.md` 后缀——`PROD_PHONE_001.md` 的业务码是 `PROD_PHONE_001`，跟评估集里 `expected_doc_ids` 用的是同一套编码。

### 2. 上传 + 异步分块

每篇文档的上传分两步走：

- **上传文件**：`POST /knowledge-base/{kb-id}/docs/upload`（multipart form），传 `.md` 文件，返回 ragent 的 `doc_id`
- **触发分块**：`POST /knowledge-base/docs/{doc-id}/chunk`，立即返回，分块在后台异步执行

分块参数：

```
{
    "targetChars": 1400,   # 目标 chunk 大小
    "maxChars": 1800,      # 单个 chunk 上限
    "minChars": 600,       # 单个 chunk 下限
    "overlapChars": 0      # chunk 间无重叠
}
```

为什么不重叠？因为这些文档是结构化 Markdown，按标题层级切分已经有足够的语义边界，加重叠反而引入冗余。

运行命令：

```
# 先 dry-run 看看文件分布，确认没问题
python eval/rag/init/upload_docs.py --dry-run

# 正式上传，每篇之间等 1 秒避免触发限制
python eval/rag/init/upload_docs.py --sleep 1
```

### 3. 断点续传：`doc_id_map.json`

这是 `upload_docs.py` 最重要的设计。每上传成功一篇，立即把业务码 → ragent doc_id 的映射追加写入 `doc_id_map.json`：

```
{
  "PROD_PHONE_001": {
    "ragent_doc_id": "2058461148245700608",
    "kb_key": "product",
    "kb_id": "2058461026313089024",
    "rel_path": "knowledge_base/01_product/detail/PROD_PHONE_001.md"
  }
}
```

下次启动时读取这个文件，已上传的跳过。115 篇上传到第 60 篇网络断了？重跑脚本自动从第 61 篇继续，不会重复上传前 60 篇。

这个映射文件还有另一个关键作用：runner 跑评测时，ragent 返回的是内部 doc_id，需要通过 `doc_id_map.json` 反查出业务码，才能跟评估集的 `expected_doc_ids` 做比对算 Hit@K。

### 4. 等待异步分块完成

这里有个容易踩的坑：`/chunk` 接口只是触发分块任务，不等完成就返回了。115 篇全部上传完后，后台可能还在分块——文档上传了但 chunk 还没生成，检索什么都搜不到。

怎么确认分块完成？调 `GET /knowledge-base/{kb-id}/docs` 看每篇文档的 `chunkCount`，大于 0 才算就绪。实际操作中上传完等几分钟就行，也可以去 Ragent 管理后台的文档列表页直接看状态。

## 初始化第三步：构建意图树

### 1. 意图树长什么样

意图树是三层结构，共 30 个节点：3 个 DOMAIN → 5 个 CATEGORY → 22 个 TOPIC。

```
SUPPORT (DOMAIN)
├── SUPPORT_PRESALE (CATEGORY)
│   ├── S1_选购推荐  → product KB
│   ├── S2_参数咨询  → product KB
│   ├── S3_对比选购  → product KB
│   ├── S4_价格活动  → policy KB
│   ├── S5_库存到货  → product KB
│   ├── S6_配件兼容  → product KB
│   └── S7_适用场景  → product KB
├── SUPPORT_USAGE (CATEGORY)
│   ├── S8_操作指引  → manual KB
│   ├── S9_配网连接  → manual KB
│   ├── S10_APP功能  → manual KB
│   ├── S11_固件升级 → manual KB
│   ├── S12_生态联动 → manual KB
│   └── S13_保养维护 → manual KB
└── SUPPORT_AFTERSALES (CATEGORY)
    ├── S14_售后政策 → policy KB
    ├── S15_退换货   → policy KB
    ├── S16_物流配送 → policy KB
    └── S17_发票会员 → policy KB

FEEDBACK (DOMAIN)
└── FEEDBACK_ALL (CATEGORY)
    ├── F1_故障报告  → faq KB
    ├── F2_功能建议  → SYSTEM ⭐
    └── F3_投诉吐槽  → SYSTEM ⭐

CHAT (DOMAIN)
└── CHAT_ALL (CATEGORY)
    ├── C1_寒暄问候  → SYSTEM ⭐
    └── C2_越界提问  → SYSTEM ⭐
```

22 个 TOPIC 叶子分两种：

- **KB-kind**（18 个）：意图命中后，去关联的知识库做 RAG 检索
- **SYSTEM-kind**（4 个，带 ⭐）：不走 RAG，直接返回系统预设回复

呼应上一篇讲的 `requires_rag` 字段：评估集里 `requires_rag=false` 的 18 条样本，大部分就落在这 4 个 SYSTEM-kind 叶子上。

### 2. KB 归属怎么决定

每个 KB-kind 叶子关联哪个知识库，不是手工指定的，而是从评估集数据里投票算出来的。

算法很直观：

- 读评估集 150 条样本
- 对每条样本，取 `intent_l2` 和 `expected_doc_ids`
- 通过 `doc_id_map.json` 查出每个文档属于哪个 KB
- 按多数投票决定该意图关联哪个 KB

举个例子：S8-操作指引 的 8 条样本，`expected_doc_ids` 里的文档 90% 在 manual KB，投票结果 → S8 关联 manual。S14-售后政策 的 8 条样本，文档 85% 在 policy KB，投票结果 → S14 关联 policy。

好处：不用手工维护意图和 KB 的映射关系。改了评估集里的 `expected_doc_ids`，重建意图树时 KB 归属自动调整。

### 3. 易混淆意图的边界描述

创建 TOPIC 节点时，除了 `intentCode` 和 `kbId`，还有一个 `description` 字段。脚本里硬编码了 4 对容易混淆的意图，在描述里加了边界区分：

| 容易混淆的两个意图 | 区分方式 |
| --- | --- |
| S4 价格活动 vs S14 售后政策 | 价格优惠 / 促销 → S4；保修维修 / 售后服务 → S14 |
| S5 库存到货 vs S16 物流配送 | 下单前问什么时候能到 → S5；下单后问快递到哪了 → S16 |

这种边界描述帮助 Ragent 的意图识别模型区分相近意图。比如用户问“买了三天了还没发货”，没有边界描述可能误分到 S5 库存到货，有了描述就能正确识别为 S16 物流配送。

另外还有一个 `examples` 字段——从评估集中取该意图的最多 5 条 query 作为示例。示例越贴近真实用户表达，意图识别越准。

运行命令：

```
python eval/rag/init/build_intent_tree.py
```

脚本按 DOMAIN → CATEGORY → TOPIC 的顺序逐层创建节点（子节点依赖父节点的 `parentCode`，顺序不能乱），产出 `intent_ids.json` 记录所有节点的 ragent ID。

## 重置与本地产物

### 1. 重置：从头再来

什么时候需要重置？换了 embedding 模型需要重新灌库、文档内容大面积更新、评估集改了需要重建意图树。

`reset_kbs.py` 做的事情：遍历所有知识库 → 逐个删除其下所有文档 → 删除知识库本身 → 清理本地映射文件。

安全设计上做了几层保护：

```
# 默认 dry-run，只打印将要删除的内容，不实际执行
python eval/rag/init/reset_kbs.py

# 加 --yes 才会真正删除，删前还有 3 秒倒计时可 Ctrl-C 取消
python eval/rag/init/reset_kbs.py --yes
```

> 重置后三个映射文件全部失效，需要从第一步开始重跑完整的初始化流程。需要注意，意图识别树 `t_intent_node` 表和 RustFS 中的 Bucket 需要登录控制台手动删除。

### 2. 三个映射文件为什么不进 git

初始化过程中产出三个映射文件，它们都在 `.gitignore` 里：

| 文件 | 内容 | 为什么不进 git |
| --- | --- | --- |
| kb_ids.json | KB 名 → ragent snowflake ID | 每次建库 ID 都不同 |
| doc_id_map.json | 业务码 → ragent doc ID | 换机器 / 换数据库 ID 全变 |
| intent_ids.json | intentCode → ragent 节点 ID | 同上 |

这三个文件是本地运行产物，不是源码。每个开发者拉完代码后跑一遍初始化自己生成。如果有人把自己的 `doc_id_map.json` 提交到 git，另一个开发者 pull 下来直接用——但他的 ragent 实例里的 doc_id 跟你的完全不同，所有检索都会 miss。

## 初始化踩过的坑

### 1. 分块没等完就跑评测

115 篇上传完，兴冲冲跑 runner，结果检索指标惨不忍睹——Hit@5 只有 0.3。排查了半天检索逻辑，最后发现是后台还在分块，很多文档的 `chunkCount` 还是 0。向量库里没有数据，检索当然什么都搜不到。

教训：上传完等几分钟，或者调 API 确认所有文档的 `chunkCount > 0` 再跑评测。

### 2. 意图树节点创建顺序错了

改了脚本逻辑后，TOPIC 节点在 CATEGORY 之前创建，API 报错找不到 `parentCode`。意图树是树形结构，子节点依赖父节点存在，必须按 DOMAIN → CATEGORY → TOPIC 的顺序逐层创建。

教训：初始化脚本的执行顺序很重要，不只是三步之间有依赖，同一步内部也有层级依赖。

### 3. 文档更新后忘了重建意图树

加了几篇新文档到 policy KB，重跑了 `upload_docs.py`。检索没问题，但意图识别出了偏差——因为意图树的 KB 归属是从评估集投票算的，新文档如果改变了投票结果，需要重建意图树才能生效。

教训：文档变更后，不只要重跑第二步（上传），还要看看第三步（意图树）是否需要更新。三步是有依赖链的。

## 小结与下一篇预告

初始化 = 三步走：

- `create_kbs.py`：建 4 个知识库，产出 `kb_ids.json`
- `upload_docs.py`：灌 115 篇文档 + 异步分块，产出 `doc_id_map.json`（增量保存，断点续传）
- `build_intent_tree.py`：构建 30 个意图节点（3 层结构，18 个 KB-kind + 4 个 SYSTEM-kind），KB 归属数据驱动投票，产出 `intent_ids.json`

三个映射文件串联三步，都不进 git。跑完之后 Ragent 处于可评测状态。

系统搭好了，接下来怎么把评估集里的 query 逐条喂进去拿数据？runner 要同时调两个接口——SSE 流拿真实答案和首字耗时，JSON 旁路拿检索证据。两个接口的数据怎么聚合？TTFT 的打点口径是什么？见第 4 篇：《一条 query 怎么跑：评测旁路接口 + SSE 聚合 + TTFT 打点》。

> Source: https://t.zsxq.com/1OfHs
> Resolved: https://articles.zsxq.com/id_h0gjwioc5fuw.html
