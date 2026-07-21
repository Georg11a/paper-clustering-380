# Path 1 Theory-Move Typology：改动、收益与局限

## 1. 目的与范围

Path 1 是加在聚类结果之后的**规则式解释层**。它不改变论文如何被嵌入、降维或分到 cluster，也不提高或降低 silhouette score。它改变的是：系统如何为已经形成的 cluster 生成可比较、可审计的概念标签。

改动前，网站使用通用 action frames：

- defines
- organizes
- applies
- evaluates
- synthesizes

这些动词能概括文本动作，但很难回答更重要的方法学问题：一个 cluster 在理论上究竟是在建构新理论、借用外部理论、检验理论，还是反思设计理论本身？

改动后，每个非 noise cluster 都先接受一次自动 theory-move coding：

1. `Building New Theory`
2. `Borrowing Theory from Other Fields`
3. `Testing Theory Empirically`
4. `Meta-Theoretical Reflection on Design`
5. `Unclear Theory Move — Requires Human Review`

第五项不是一种实质性 theory move，而是证据不足或结果含混时的保守状态。

## 2. 文献基础应如何表述

这四种实质类别是**综合多条文献脉络后形成的 analytical codes**，不是从 Gregor (2006) 或 Gregor and Jones (2007) 原样复制的四分类 taxonomy。

- [Gregor (2006)](https://doi.org/10.2307/25148742) 按理论功能讨论多种 theory types，为区分不同理论作用提供总体基础，但其五类并不等于本项目的四类。
- [Gregor and Jones (2007)](https://doi.org/10.17705/1jais.00129) 讨论 design theory 的组成，包括 purpose and scope、constructs、testable propositions 和 justificatory knowledge，可支持同时呈现 theory move 与 application domain。
- [Colquitt and Zapata-Phelan (2007)](https://doi.org/10.5465/amj.2007.28165855) 将 theory building 与 theory testing 作为两个理论贡献维度，为 building 和 testing codes 提供直接依据。
- [Truex, Holmström, and Keil (2006)](https://aisel.aisnet.org/jais/vol7/iss1/33/) 讨论从其他学科借用并适配理论时需要处理的问题。
- [Moeini et al. (2020)](https://doi.org/10.1177/0268396220912745) 进一步讨论 borrowed theory 的 recontextualization，为 borrowing code 提供更具体的依据。
- [Love (2000)](https://doi.org/10.1016/S0142-694X(99)00012-5) 对 design theory 进行 meta-theoretical analysis，为 meta-theoretical reflection code 提供直接依据。
- [Hsieh and Shannon (2005)](https://doi.org/10.1177/1049732305276687) 的 directed content analysis 支持从既有理论和研究中预先定义 codes，再用于文本编码。
- [Grimmer and Stewart (2013)](https://doi.org/10.1093/pan/mps028) 强调自动文本分析必须接受针对具体问题的人工验证，不能代替 close reading。

推荐的方法学表述是：

> Drawing on directed content analysis, we operationalized four theory-move categories from several strands of prior research: theory building, theory testing, theory borrowing and adaptation, and meta-theoretical reflection on design. These categories are literature-informed analytical codes rather than a taxonomy reproduced verbatim from Gregor (2006) or Gregor and Jones (2007). The resulting labels are treated as automatic first-pass coding and are subject to human validation.

## 3. 系统具体改了什么

### 3.1 新增独立分类器

[`scripts/theory_typology.py`](../scripts/theory_typology.py) 包含四组确定性 regex patterns、权重、cluster-level aggregation 和 `Unclear` fallback。输入相同，输出必然相同。

分类器只读取：

- title
- abstract
- discussion summary
- discussion excerpt

它不读取整段 `extracted_context`，因为其中可能包含 related-work 叙述；如果直接使用，系统可能把被引用论文的理论动作误认成当前论文自己的贡献。

### 3.2 从 paper-level evidence 聚合到 cluster

规则先对每篇论文分别计分，再汇总到 cluster。这样可以记录：

- 哪些 pattern 命中；
- 有多少篇论文支持该 move；
- cluster 的 weighted score；
- 第一名与第二名是否过于接近。

对于 8 篇及以上的 cluster，至少需要 2 篇且不少于约 10% 的论文支持同一 move。证据太少或类别过于接近时，系统输出 `Unclear`，而不是默认成 building。

### 3.3 新标签由三个独立部分构成

标签现在使用：

`Design-knowledge form: Theory move | Domain: Application domain`

例如：

`Design Knowledge: Borrowing Theory from Other Fields | Domain: Tool`

如果是：

`Design Knowledge: Unclear Theory Move — Requires Human Review | Domain: Industrial Design`

它表示**theory move 需要人工判断**，而 `Industrial Design` 只是系统推断的应用领域。它不表示“必须由工业设计师审核”。使用 `| Domain:` 是为了避免旧写法 `Requires Human Review in Industrial Design` 造成这种歧义。

### 3.4 网站增加审计信息

每个 paper detail panel 现在显示：

- Form
- Theory move
- Support
- Matched patterns

CSV 和网页 payload 新增：

- `theory_move_key`
- `theory_move`
- `theory_move_patterns`
- `theory_move_support`

[`scripts/cluster_papers.py`](../scripts/cluster_papers.py) 负责未来完整重跑时生成这些字段；[`scripts/update_cluster_claims.py`](../scripts/update_cluster_claims.py) 可在不改变 cluster assignments 和 UMAP coordinates 的情况下刷新现有结果。

## 4. 哪些地方得到改善

### 4.1 标签从表面动作变成理论贡献判断

旧标签中的 defines、organizes 或 synthesizes 描述的是写作或知识处理动作。Path 1 试图描述理论贡献类型，因此不同 cluster 之间更容易进行方法学比较。

### 4.2 标签具有统一比较框架

所有关键词、text views 和 clustering methods 使用同一组 codes。研究者可以横向比较某个 keyword 下 building、borrowing、testing 和 meta-reflection 的分布，而不必先统一大量开放式标签。

### 4.3 输出可重复、可追踪

规则、权重和阈值固定；同样的 corpus 会得到同样的结果。界面同时暴露支持论文数和 matched patterns，因此结果可以回溯到具体判断依据。

### 4.4 失败方式更诚实

旧方案曾考虑 unmatched cases 自动回退到 building，但 building 本身是实质性判断。现在证据不足会进入 `Unclear Theory Move — Requires Human Review`，避免为了覆盖率制造虚假的确定性。

### 4.5 theory move 与 domain 不再混为一谈

一个 cluster “研究什么领域”和“做了什么理论动作”是两个维度。明确分开后，`Borrowing Theory | Domain: Tool` 不会被误解成 “Tool” 是一种理论动作。

## 5. 主要好处

- **Deterministic**：没有随机生成或 prompt drift。
- **Reproducible**：同一输入、规则版本和阈值产生同一输出。
- **Auditable**：能显示 support count、score 和 matched patterns。
- **Comparable**：固定 codes 便于跨 cluster 和跨 keyword 统计。
- **Low cost**：不需要为每个 cluster 调用 LLM。
- **Conservative**：不确定时保留 `Unclear`，不强行赋予理论意义。
- **Compatible with Path 2**：未来可让 LLM 只处理 `Unclear` 或高价值 clusters，并把 Path 1 作为 fallback 和比较基线。

## 6. 局限与风险

### 6.1 当前 `Unclear` 比例很高

当前网站共有 182 个实际 clusters：

| Theory move | Cluster 数 |
|---|---:|
| Building New Theory | 13 |
| Borrowing Theory from Other Fields | 24 |
| Testing Theory Empirically | 1 |
| Meta-Theoretical Reflection on Design | 13 |
| Unclear Theory Move — Requires Human Review | 131 |

约 72% 的 clusters 为 `Unclear`。这说明规则目前很保守，也说明全站许多 keyword clusters 并不以明确的 theory move 为中心。它不应被描述为已经完成了高覆盖率自动编码。

### 6.2 四类并非天然互斥

一篇论文可以先借用理论，再扩展理论，并进行经验检验。当前系统为了生成单一主标签，只选择 dominant move；这会压缩多重理论贡献。未来可增加 secondary move，而不是把分类当作完全互斥的事实。

### 6.3 regex 无法稳定理解否定、转述和隐含贡献

例如 “prior work tested the theory” 与 “we tested the theory” 在词汇上很接近。限制文本来源和使用较具体 patterns 能降低风险，但不能消除语义误判。

### 6.4 domain 也是启发式推断

`Industrial Design`、`Tool` 或 `Game` 来自已有 facet patterns，不是人工确认的研究领域。theory move 为 `Unclear` 并不意味着 domain 也不确定，反之亦然；两项都应允许人工修改。

### 6.5 固定 taxonomy 限制表达能力

Path 1 适合全 corpus 的一致编码，但标签不如 Path 2 的开放式 LLM label 具体。它可能无法表达诸如“通过参与式保存实践重构文化遗产理论”这样的细粒度概念。

### 6.6 目前没有 gold-standard validation set

规则自测只能证明代码按预期运行，不能证明 coding 具有研究效度。在人工标注样本上报告 precision、recall 或 coder agreement 之前，这些输出只能称为 automatic first-pass coding。

## 7. 建议的人工验证流程

1. 优先使用 `Design theory → KMEANS → Context` 作为 pilot，因为它包含多个 cluster，且每个 cluster 有可用 PDF。
2. 每个 cluster 选择 2 篇代表论文，共约 10 篇。代表性可按 centroid distance/representative rank 选择，而不是随机抽取。
3. 两位研究者独立阅读 title、abstract 和相关 source evidence，为每篇论文及 cluster 编码 theory move。
4. 比较人工编码与 Path 1 输出，并记录 false positive、false negative 和多标签案例。
5. 对分歧案例讨论后修订 codebook 和 patterns；不要只为提高覆盖率而降低 `Unclear` 阈值。
6. 在锁定规则版本后，再报告覆盖率、每类 precision/recall，以及人工一致性指标。

## 8. 可以得出的结论与不能得出的结论

可以说：

> The website now applies a deterministic, literature-informed theory-move typology as an auditable first-pass coding layer. Unsupported or ambiguous clusters are explicitly flagged for human review.

不能说：

- 这四类就是 Gregor (2006) 原文提出的 taxonomy；
- Path 1 已经证明所有 clusters 的理论类型；
- `Unclear` 表示论文没有理论贡献；
- 新标签改善了 clustering quality；
- 自动规则可以取代人工阅读和验证。

Path 1 的真正价值是建立一个一致、透明、可复核的起点，而不是把规则输出包装成最终解释。
