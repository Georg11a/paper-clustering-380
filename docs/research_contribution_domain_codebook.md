# Research Contribution and Application Domain Codebook

## Purpose

The explorer codes each cluster along two independent dimensions: its primary research contribution and its application domain or domains. These labels are deterministic first-pass codes supported by matched textual evidence and must be spot-checked by human reviewers.

Artifact forms such as Interface, Tool, Dashboard, Prototype, and System are artifact subtypes rather than application domains. A method such as Interview or Case Study is also not a contribution type by itself: the primary contribution is determined by the main knowledge output claimed by the papers.

## Contribution-Specific Cluster Summaries

| Primary contribution | Evidence fields to extract | Cluster-summary template |
|---|---|---|
| **Empirical** | Study purpose; population or data; phenomenon; method; supported finding | “Empirically examines **[phenomenon]** among **[population]** in **[domain]**, using **[method]**. The studies consistently find **[supported finding]**.” |
| **Algorithmic** | Computational task; algorithm or model; objective; evaluation metric or benchmark | “Develops **[algorithm/model]** for **[task]** in **[domain]**, evaluated against **[metric/benchmark]**.” |
| **Artifact** | Artifact type; intended users; supported activity; evaluation or deployment | “Introduces **[tool/interface/system]** for **[users]** to support **[activity]** in **[domain]**, evaluated through **[method]**.” |
| **Methodological** | Method subtype; target activity; intended user; desired outcome; validation | “Proposes **[guidelines/framework/method]** for **[design or research activity]**, intended to improve **[outcome]** in **[domain]**.” |
| **Theoretical** | Theory move; focal concept; named theory; source discipline; scope | “Adapts **[named theory]** from **[discipline]** to explain **[design phenomenon]** in **[domain]**.” |
| **Dataset** | Dataset content; unit; population; scale; intended use; availability | “Contributes a dataset of **[content/unit]** covering **[scope]**, intended for **[task]** in **[domain]**.” |
| **Survey/Synthesis** | Review scope; corpus boundary; synthesis method; output; research gap | “Reviews **[scope]** across **[corpus]**, synthesizing **[taxonomy/trends/gaps]** and identifying **[key gap]**.” |

Templates define the required evidence slots; the system must not fill a slot by guessing. Unsupported slots remain unstated or trigger human review.

## Application Domains

| Domain | One-sentence definition |
|---|---|
| **Healthcare, Medicine, Surgery** | Covers clinical diagnosis, treatment, surgery, patient care, healthcare delivery, and health policy. |
| **Finance, Business, Economy** | Covers finance, investment, credit, markets, business management, and economic decision-making. |
| **Transportation, Mobility, Planning** | Covers vehicles, traffic, mobility, routing, logistics, and urban or infrastructure planning. |
| **Law, Democracy, Governance** | Covers legal and judicial decisions, democracy, elections, public policy, regulation, and governance. |
| **Everyday, Employment, Public Service** | Covers everyday personal decisions, employment and workforce issues, welfare, and public services. |
| **Education, Teaching, Research** | Covers learning, teaching, assessment, academic research, and educational administration. |
| **Manufacturing, Industry, Automation** | Covers manufacturing, production, industrial processes, automation, robotics, and supply chains. |
| **Media, Communication, Entertainment** | Covers news, social media, communication, content production, entertainment, games, and sports. |
| **Environment, Resource, Energy** | Covers the environment, climate, agriculture, water, energy, natural resources, and sustainability. |
| **Software, System, Cybersecurity** | Covers software development, systems engineering, IT operations, privacy, and cybersecurity. |
| **Defense, Military, Emergency** | Covers defense, military operations, public safety, emergency response, and disaster management. |
| **Design, Creativity, Architecture** | Covers UI/UX, product, graphic, and industrial design, creative practice, craft, and architecture. |
| **Generic, Abstract, Domain-Agnostic** | Covers general theories, methods, algorithms, or frameworks that do not target a specific application domain. |

Application domains may be multi-label when multiple settings are explicitly supported. `Generic, Abstract, Domain-Agnostic` is mutually exclusive with specific domains and is used only when no specific application setting has reliable evidence.

## Theoretical Contribution Subtypes

The theory-move layer is a contribution-specific subtype, not the only type of cluster interpretation. It distinguishes Building New Theory, Borrowing and Adapting Existing Theory, Testing Theory Empirically, and Meta-Theoretical Reflection on Design.

Borrowing and adaptation should be reported as `named theory × source discipline × target phenomenon`, for example: “Adapting Activity Theory from Psychology to explain collaborative design practice.” If the theory or source discipline is not explicit, the system must say that it is not stated rather than infer it.

## Methodological Sources

- [Wobbrock and Kientz (2016)](https://doi.org/10.1145/2907069) describe research-contribution types and how their evaluation criteria differ.
- [Truex, Holmström, and Keil (2006)](https://aisel.aisnet.org/jais/vol7/iss1/33/) discuss how theories borrowed from another discipline should be adapted to a new phenomenon and context.
- [Gebru et al. (2021)](https://doi.org/10.1145/3458723) motivate structured documentation of dataset composition, collection, uses, and limitations.
- [Page et al. (2021)](https://doi.org/10.1136/bmj.n71) specify transparent reporting of systematic-review objectives, selection, synthesis, results, and limitations.
