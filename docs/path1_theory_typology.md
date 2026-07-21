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

改动后，每个非 noise cluster 先接受 research-contribution coding。全站上层使用以下七类：

1. `Empirical Contribution`
2. `Algorithmic Contribution`
3. `Artifact Contribution`
4. `Methodological Contribution`
5. `Theoretical Contribution`
6. `Dataset Contribution`
7. `Survey/Synthesis Contribution`

只有 primary 或真正并列的 secondary contribution 为 Theoretical 时，系统才继续执行 Path 1：

1. `Building New Theory`
2. `Borrowing and Adapting Existing Theory`
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

[`scripts/research_typology.py`](../scripts/research_typology.py) 负责七种 contribution types 和 13 application domains；[`scripts/theory_typology.py`](../scripts/theory_typology.py) 只负责 Theoretical contribution 下的四种 Path 1 moves。两个模块都包含确定性 patterns、权重、cluster-level aggregation 和保守 fallback；输入相同，输出必然相同。

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

一般标签现在使用：

`Design-knowledge form: Primary contribution | Domain: application domain(s)`

例如：

`Design Guidelines: Methodological Contribution | Domain: Healthcare, Medicine, Surgery`

这里的 Domain 使用 13 类 application-domain codebook。`Interface`、`Tool` 和 `Dashboard` 是 artifact/system types，不是 domain；`Designers`、`Patients` 是 stakeholder/population；`Interview`、`Case Study` 是 method。不同 facet 不应互相填补。

如果 cluster 的 artifact 是 Tool 但没有可靠的 application context/domain，则应分别呈现：

`Design Knowledge: Artifact Contribution | Domain: Generic, Abstract, Domain-Agnostic`

`Artifact/System: Tool`

没有具体 application-domain evidence 时使用 `Generic, Abstract, Domain-Agnostic`，不能用 artifact 代替。

如果是：

`Design Theory: Theoretical Contribution — Unclear Theory Move — Requires Human Review | Domain: Design, Creativity, Architecture`

它表示 primary contribution 已被编码为 Theoretical，但细分 theory move 仍需人工判断；Domain 是独立的 application setting。非理论贡献不会显示 Path 1，也不会被标成 theory unclear。

### 3.4 网站增加审计信息

每个 paper detail panel 现在显示：

- Form
- Primary/secondary contribution
- Application domain(s)
- Contribution/domain support and matched patterns
- Path 1 theory move（仅在 Theoretical contribution 下）

CSV 和网页 payload 新增：

- `theory_move_key`
- `theory_move`
- `theory_move_patterns`
- `theory_move_support`

并新增 `contribution_type*` 与 `application_domain*` 审计字段。

[`scripts/cluster_papers.py`](../scripts/cluster_papers.py) 负责未来完整重跑时生成这些字段；[`scripts/update_cluster_claims.py`](../scripts/update_cluster_claims.py) 可在不改变 cluster assignments 和 UMAP coordinates 的情况下刷新现有结果。

## 4. 哪些地方得到改善

### 4.1 标签从表面动作变成研究贡献判断

旧标签中的 defines、organizes 或 synthesizes 描述的是写作或知识处理动作。新上层先判断 Empirical、Artifact、Methodological、Theoretical 等主要研究贡献；只有理论贡献再判断 Path 1 move。

### 4.2 标签具有统一比较框架

所有关键词、text views 和 clustering methods 使用同一组 contribution/domain codes；Theoretical 子集再横向比较 building、borrowing、testing 和 meta-reflection。

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

### 6.1 路由后仍有 `Unclear`，但含义更准确

当前 182 个 cluster-view results 的 primary contributions 为 Methodological 114、Theoretical 52、Survey/Synthesis 8、Artifact 4、Unclear 4。52 个 Theoretical results 中，22 个 theory moves 仍为 Unclear。当前没有 cluster 以 Empirical、Algorithmic 或 Dataset 为 primary，尽管 paper-level coding 中存在 Empirical 和 Algorithmic papers；这表示它们没有在任何 cluster 内形成占主导地位且达到支持阈值的贡献类型。这里的 182 包含同一 corpus 在不同 keyword、view 和 algorithm 下的重复分析，不能解释为 182 个独立主题。

### 6.2 四类并非天然互斥

一篇论文可以先借用理论，再扩展理论，并进行经验检验。当前系统为了生成单一主标签，只选择 dominant move；这会压缩多重理论贡献。未来可增加 secondary move，而不是把分类当作完全互斥的事实。

### 6.3 regex 无法稳定理解否定、转述和隐含贡献

例如 “prior work tested the theory” 与 “we tested the theory” 在词汇上很接近。限制文本来源和使用较具体 patterns 能降低风险，但不能消除语义误判。

### 6.4 domain 也是启发式推断

13 类 domains 来自自动 patterns，不是人工确认的研究领域。Tool 和 Interface 已被排除，但跨领域词仍可能产生过宽或错误的多标签结果；contribution、domain 和 theory move 都应允许人工修改。

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

> The website first applies a deterministic contribution-type and application-domain coding layer. Path 1 theory-move coding is then applied only to clusters with a theoretical primary or equivalent secondary contribution. All outputs are auditable first-pass codes subject to human validation.

不能说：

- 这四类就是 Gregor (2006) 原文提出的 taxonomy；
- 所有 clusters 都具有可解释的 theory move；
- `Unclear` 表示论文没有理论贡献；
- 新标签改善了 clustering quality；
- 自动规则可以取代人工阅读和验证。

Path 1 的真正价值是建立一个一致、透明、可复核的起点，而不是把规则输出包装成最终解释。

## 9. 全站上层分类：Contribution Type × Application Domain

Path 1 不再作为所有 clusters 的唯一主分类。系统先编码 contribution type 和 application domain，再只对 theoretical contributions 使用 Path 1。

Contribution type 使用 Empirical、Algorithmic、Artifact、Methodological、Theoretical、Dataset 和 `Survey/Synthesis` 七类。默认选择一个 primary contribution type，只有两个贡献真正等价时才增加 secondary type。完整定义、需要抽取的 evidence fields 和 summary templates 见 [`research_contribution_domain_codebook.md`](research_contribution_domain_codebook.md)。

Application domain 使用以下 13 类；每类的英文单句定义见 [`research_contribution_domain_codebook.md`](research_contribution_domain_codebook.md)：

1. Healthcare, medicine, surgery
2. Finance, business, economy
3. Transportation, mobility, planning
4. Law, democracy, governance
5. Everyday, employment, public service
6. Education, teaching, research
7. Manufacturing, industry, automation
8. Media, communication, entertainment
9. Environment, resource, energy
10. Software, system, cybersecurity
11. Defense, military, emergency
12. Design, creativity, architecture
13. Generic, abstract, domain-agnostic

Domain 可以多标签，因为同一论文可能横跨多个 application settings。Domain 应优先由 abstract 明示内容判断；未明示时才参考任务和数据集。Interface、Tool、Dashboard 和 Game 仍是 artifact/system types，不能作为 domain。

推荐的层级输出是：

`Form | Primary contribution | Application domain(s)`

并按 contribution type 增加专属字段：

- Theoretical contribution → Path 1 theory move
- Artifact contribution → Artifact/system type
- Empirical contribution → Empirical purpose and method
- Methodological contribution → Method/guideline/framework type
- Algorithmic contribution → Algorithm/model contribution
- Dataset contribution → Dataset/benchmark type
- Survey/Synthesis → Review/synthesis type

例如：

`Design Guidelines | Methodological contribution | Domain: Software, system, cybersecurity`

`Artifact/System: Interface`

这里 Interface 描述产出或研究对象的形态，不描述 application domain。
