#!/usr/bin/env python3
"""Refine Page 3 cluster labels without changing cluster membership.

The refinement is deterministic and post hoc. It combines document coverage
with cross-cluster exclusivity, prefers multiword phrases, expands common
abbreviations, removes lexical duplicates, and writes evidence-led summaries.
No LLM is used and no clustering fields or coordinates are modified.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


PAYLOAD_RE = re.compile(
    r'(<script id="payload" type="application/json">)(.*?)(</script>)',
    re.DOTALL,
)
TOKEN_RE = re.compile(r"[a-z0-9]+")

ABBREVIATIONS = (
    (re.compile(r"\bdr\b", re.I), "design rationale"),
    (re.compile(r"\bdsr\b", re.I), "design science research"),
    (re.compile(r"\bhci\b", re.I), "human computer interaction"),
    (re.compile(r"\bhr[iI]\b"), "human robot interaction"),
    (re.compile(r"\bai\b", re.I), "artificial intelligence"),
    (re.compile(r"\boo\b", re.I), "object oriented"),
)

DISPLAY_ABBREVIATIONS = {
    "human computer interaction": "HCI",
    "human robot interaction": "Human–Robot Interaction",
    "artificial intelligence": "AI",
    "design science research": "Design Science Research",
    "object oriented": "Object-Oriented",
}

CUSTOM_STOP_WORDS = {
    "abstract", "analysis", "approach", "approaches", "article", "articles",
    "author", "authors", "based", "chapter", "conclusion", "design", "designs",
    "different", "discussion", "findings", "framework", "frameworks", "future",
    "information", "introduction", "knowledge", "method", "methods", "model",
    "models", "new", "paper", "papers", "process", "processes", "project",
    "projects", "research", "result", "results", "section", "study", "studies",
    "system", "systems", "theory", "use", "used", "uses", "using", "work",
    "dhs", "dhsfx", "edps", "eis", "ll", "sep", "soni", "cation",
}

STOP_WORDS = set(ENGLISH_STOP_WORDS) | CUSTOM_STOP_WORDS


# Human-readable profiles for previously reviewed topic families. The current
# partition can split or merge these families when the corpus expands, so the
# profiles are matched by lexical evidence rather than by cluster number.
REVIEWED_PROFILES = {
    0: {
        "label": "Design Heuristics · Extraction, Formalization, and Evaluation",
        "focus": "the development and use of design heuristics as reusable guidance for ideation and evaluation",
        "distinction": "The papers emphasize how heuristics are extracted, classified, formalized, and adapted to domains such as products, dashboards, usability, and risk communication.",
        "evidence": ["design heuristics", "heuristic extraction", "knowledge formalization", "usability heuristics"],
    },
    1: {
        "label": "HCI Design Knowledge · Models, Principles, Rules, and Heuristics",
        "focus": "the forms in which HCI design knowledge is represented and made actionable",
        "distinction": "It concentrates on models, methods, principles, rules, and engineering guidance for human–computer interaction rather than on a single application domain.",
        "evidence": ["HCI design knowledge", "engineering principles", "models and methods", "rules and heuristics"],
    },
    2: {
        "label": "Game Design Knowledge · Mechanics, Patterns, and Player Experience",
        "focus": "theories and reusable structures for designing games, gamification, and interactive narratives",
        "distinction": "Recurring concerns include game mechanics, design patterns, player experience, serious games, achievements, and design education.",
        "evidence": ["game design", "game mechanics", "game design patterns", "player experience", "serious games"],
    },
    3: {
        "label": "Visualization Design Knowledge · Visual Inquiry, Sonification, and Design Studies",
        "focus": "knowledge and methods for designing visual and auditory representations of information",
        "distinction": "The cluster connects visualization design spaces and patterns with sonification, visual inquiry tools, design-study methodology, aesthetics, and narrative visualization.",
        "evidence": ["visualization design", "sonification", "visual inquiry", "design study methodology", "visualization patterns"],
    },
    4: {
        "label": "Design Methods · Description, Evaluation, and Ethical Practice",
        "focus": "how design methods are described, assessed, corroborated, and used in practice",
        "distinction": "Unlike clusters centered on a substantive design domain, these papers examine method structure, effectiveness, rationale, situated use, and ethics-focused methods.",
        "evidence": ["design methods", "method evaluation", "method rationale", "ethics-focused methods", "design practice"],
    },
    5: {
        "label": "Learning Design · Educational Patterns, Teacher Knowledge, and Blended Learning",
        "focus": "reusable design knowledge for teaching, learning environments, and technology-enhanced education",
        "distinction": "The papers address educational design patterns, teacher competencies, blended and online learning, learning analytics, scaffolding, and collaborative pattern-language development.",
        "evidence": ["learning design", "educational design patterns", "teacher design knowledge", "blended learning", "learning environments"],
    },
    6: {
        "label": "Design Rationale · Argumentation, Documentation, and Reuse",
        "focus": "the capture and use of reasons, alternatives, and decisions produced during design",
        "distinction": "The cluster links argumentation structures with rationale documentation, design intent, communication, retrieval, and reuse; the abbreviation DR is normalized to design rationale.",
        "evidence": ["design rationale", "argumentation", "rationale documentation", "design intent", "rationale reuse"],
    },
    7: {
        "label": "Design Principles · Formulation, Development, and Utilization",
        "focus": "the construction and practical use of design principles as prescriptive design knowledge",
        "distinction": "The papers examine principle anatomy, formulation, development from evidence, reporting quality, and transfer into organizational, visual, and educational settings.",
        "evidence": ["design principles", "principle formulation", "principle development", "research-based principles", "principle utilization"],
    },
    8: {
        "label": "Formal Design Theories · C–K Theory, Mathematical Models, and Creative Preservation",
        "focus": "formal and mathematical accounts of design reasoning and innovation",
        "distinction": "C–K theory, general design theory, artifact and process models, and applications to creative preservation distinguish this cluster from broader discussions of theory construction.",
        "evidence": ["C-K theory", "mathematical theory of design", "general design theory", "creative preservation", "design process models"],
    },
    9: {
        "label": "Human-Centered and Embodied Design · Agency, Empathy, and Reflective Practice",
        "focus": "human experience, embodiment, agency, and reflection in interaction and technology design",
        "distinction": "The papers range from design-oriented HCI and reflective practice to empathy, embodied co-design, posthuman critiques, sensory communication, and human-centeredness assessment.",
        "evidence": ["human-centered design", "embodied design", "reflective practice", "empathy", "design agency"],
    },
    10: {
        "label": "Public-Health and Participatory Design · Co-Design, Equity, and Health Systems",
        "focus": "participatory and value-sensitive design in public-health and health-related settings",
        "distinction": "The cluster emphasizes co-design, equitable access, health informatics, chronic-disease self-management, impact assessment, and the translation of design knowledge across disciplines.",
        "evidence": ["public health", "participatory design", "co-design", "health informatics", "value-sensitive design"],
    },
    11: {
        "label": "Interaction Patterns · Human–Robot, Human–AI, and Pervasive Systems",
        "focus": "reusable interaction patterns and design spaces for intelligent and pervasive systems",
        "distinction": "Human–robot sociality, human–AI interaction primitives, pervasive interaction, and system interaction patterns provide the cluster's shared context.",
        "evidence": ["interaction patterns", "human-robot interaction", "human-AI interaction", "pervasive systems", "design space"],
    },
    12: {
        "label": "Engineering Product Design · Complex Products, Manufacturing, and Prototyping",
        "focus": "systematic methods and theories for engineering products and production-oriented innovation",
        "distinction": "The papers cover complex and mechanical products, additive manufacturing, hybrid components and prototyping, robust design, proof of concept, and product recovery.",
        "evidence": ["engineering product design", "complex products", "additive manufacturing", "hybrid prototyping", "systematic design procedures"],
    },
    13: {
        "label": "Information-Systems Design Theory · Digital, Community, and Socio-Technical Systems",
        "focus": "the construction and application of design theories for information systems",
        "distinction": "The cluster includes web-based education, communities, platforms, information infrastructures, privacy, security laboratories, integration management, and explainable systems.",
        "evidence": ["information systems design theory", "socio-technical systems", "digital platforms", "community information systems", "information infrastructures"],
    },
    14: {
        "label": "Design Theory Construction · Criteria, Epistemology, and Research through Design",
        "focus": "how design theories and knowledge claims are constructed, evaluated, and justified",
        "distinction": "It foregrounds theory criteria, epistemology, definitions, research through design, designerly research methods, and relations between theory, artifacts, and practice.",
        "evidence": ["design theory construction", "research through design", "theory criteria", "design epistemology", "designerly research methods"],
    },
    15: {
        "label": "Design Science Research Knowledge · Codification, Accumulation, and Reuse",
        "focus": "the production and management of design knowledge contributions in design science research",
        "distinction": "The papers emphasize knowledge typologies, contribution reporting, codification, accumulation mechanisms, reusable principles, ontologies, problem and solution spaces, and meta-studies.",
        "evidence": ["design science research", "design knowledge codification", "knowledge accumulation", "knowledge reuse", "design knowledge contributions"],
    },
    16: {
        "label": "Forms of Design Knowledge · Product, Industrial, and Tacit Knowledge",
        "focus": "the forms, levels, representations, and transfer of design knowledge",
        "distinction": "Industrial and product design, tacit and intermediate knowledge, prototyping, education, cognition, and knowledge representation anchor this cluster.",
        "evidence": ["design knowledge", "industrial design knowledge", "product design", "tacit knowledge", "knowledge representation"],
    },
    17: {
        "label": "Designer Expertise · Practice Knowledge, Philosophy, and Communities",
        "focus": "what designers know and how expertise is formed, researched, and legitimated",
        "distinction": "The cluster brings together design expertise, sociology and philosophy of design knowledge, practice-based research, instructional-design expertise, and practitioner communities.",
        "evidence": ["design expertise", "designer knowledge", "practice-based research", "design philosophy", "practitioner communities"],
    },
    18: {
        "label": "Organizational and AI-Enabled Systems · Modularity, Data, and Design Rules",
        "focus": "frameworks and design rules for organizational, enterprise, data-intensive, and AI-enabled systems",
        "distinction": "Modularity, platforms, business and software-process design, data ecosystems, AI governance, cognitive engineering, system integration, and organizational behavior make this a broad systems-oriented cluster.",
        "evidence": ["organizational design", "AI-enabled systems", "modularity", "data ecosystems", "design rules"],
    },
    19: {
        "label": "Software Architecture and Interfaces · Design Spaces, Rules, and Abstraction",
        "focus": "architectural knowledge for designing software systems and user interfaces",
        "distinction": "The shared emphasis is on design spaces, architectural rules and styles, abstraction, interface structure, implementation guidance, and ecological interface foundations.",
        "evidence": ["software architecture", "user interface", "design spaces", "design rules", "architectural abstraction"],
    },
    20: {
        "label": "Emerging Software and AI Patterns · Languages, Agents, and Domain Applications",
        "focus": "pattern-based guidance for emerging software, AI, and domain-specific systems",
        "distinction": "Compared with the other pattern clusters, this group is more heterogeneous and application-led, spanning AI agents, domain-specific languages, retrieval-augmented generation, health algorithms, biological systems, and fluent interfaces.",
        "evidence": ["software design patterns", "AI design patterns", "domain-specific languages", "agent design patterns", "domain applications"],
    },
    21: {
        "label": "Architectural Patterns · Reference Architectures, Services, and Components",
        "focus": "patterns for organizing software architectures and reusable system structures",
        "distinction": "Reference architectures, layers and styles, component-oriented development, service engineering, Tropos, and blockchain interoperability distinguish it from code-level and interaction-oriented pattern clusters.",
        "evidence": ["architectural patterns", "reference architectures", "service engineering", "component-oriented software", "blockchain interoperability"],
    },
    22: {
        "label": "Software Design Patterns · Object-Oriented Reuse, Composition, and Formalization",
        "focus": "the representation and application of reusable design solutions in software engineering",
        "distinction": "Object-oriented architecture, composition, formal semantics, refactoring, security taxonomies, and industrial pattern use distinguish this cluster from pattern languages centered on human practice.",
        "evidence": ["software design patterns", "object-oriented design", "pattern reuse", "pattern composition", "formal semantics"],
    },
    23: {
        "label": "Pattern Languages · Interaction, Learning, and Collaborative Practice",
        "focus": "pattern languages as mechanisms for capturing and communicating reusable design knowledge",
        "distinction": "Unlike the software-centered pattern cluster, these papers apply patterns to interaction design, learning, project communication, participatory futures, collaborative knowledge, and service co-creation.",
        "evidence": ["pattern language", "interaction design", "learning patterns", "collaborative knowledge", "service co-creation"],
    },
}


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    for pattern, replacement in ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\bet\s+al\.?\b", " ", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9-]+", " ", text)
    return " ".join(text.casefold().split())


def canonical_tokens(phrase: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(phrase.casefold()):
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 5 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tuple(tokens)


def phrase_is_usable(phrase: str) -> bool:
    tokens = phrase.split()
    if not 2 <= len(tokens) <= 4:
        return False
    if tokens[0] in STOP_WORDS or tokens[-1] in STOP_WORDS:
        return False
    if any(len(token) < 2 for token in tokens):
        return False
    return len(set(canonical_tokens(phrase))) >= 2


def display_phrase(phrase: str, title_case: bool = False) -> str:
    phrase = " ".join(phrase.split())
    if phrase in DISPLAY_ABBREVIATIONS:
        return DISPLAY_ABBREVIATIONS[phrase]
    output = phrase.title() if title_case else phrase
    replacements = {
        "Hci": "HCI",
        "Ai": "AI",
        "Dsr": "DSR",
        "Object Oriented": "Object-Oriented",
        "Human Robot": "Human–Robot",
    }
    for source, target in replacements.items():
        output = output.replace(source, target)
    return output


def rank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return (order + 1) / max(len(values), 1)


def lexical_overlap(left: str, right: str) -> float:
    a, b = set(canonical_tokens(left)), set(canonical_tokens(right))
    return len(a & b) / max(len(a | b), 1)


def select_nonredundant(
    candidates: list[str],
    limit: int,
    blocked: list[str] | None = None,
) -> list[str]:
    selected: list[str] = []
    comparison = list(blocked or [])
    for candidate in candidates:
        if any(
            canonical_tokens(candidate) == canonical_tokens(other)
            or lexical_overlap(candidate, other) >= 0.67
            for other in comparison + selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def profile_match_score(
    profile: dict[str, object],
    statistical_phrases: list[str],
    member_titles: list[str],
) -> float:
    """Score a reviewed profile against current statistical and title evidence."""
    phrase_tokens = set()
    for phrase in statistical_phrases:
        phrase_tokens.update(canonical_tokens(phrase))
    title_tokens = set(canonical_tokens(" ".join(member_titles)))
    evidence_phrases = [str(value) for value in profile["evidence"]]
    evidence_tokens = set()
    exact_phrase_hits = 0
    normalized_titles = [normalize_text(title) for title in member_titles]
    for phrase in evidence_phrases:
        tokens = set(canonical_tokens(normalize_text(phrase)))
        evidence_tokens.update(tokens)
        normalized_phrase = normalize_text(phrase)
        if any(normalized_phrase in title for title in normalized_titles):
            exact_phrase_hits += 1
    statistical_overlap = len(evidence_tokens & phrase_tokens) / max(
        len(evidence_tokens), 1
    )
    title_overlap = len(evidence_tokens & title_tokens) / max(len(evidence_tokens), 1)
    return 3.0 * exact_phrase_hits + 2.0 * statistical_overlap + title_overlap


def choose_profile(
    ranking: dict[str, list[str]], members: list[dict[str, object]]
) -> dict[str, object] | None:
    statistical_phrases = ranking["core"] + ranking["distinctive"]
    titles = [str(member.get("title", "")) for member in members]
    scored = sorted(
        [
        (profile_match_score(profile, statistical_phrases, titles), profile)
        for profile in REVIEWED_PROFILES.values()
        ],
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, profile = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    return profile if best_score >= 3.0 and best_score - second_score >= 0.75 else None


def statistical_profile(
    ranking: dict[str, list[str]], members: list[dict[str, object]]
) -> dict[str, object]:
    """Create a deterministic fallback when no reviewed profile fits."""
    phrases = select_nonredundant(
        ranking["core"] + ranking["distinctive"], 5
    )
    if not phrases:
        phrases = ["mixed design research"]
    label_terms = [display_phrase(value, title_case=True) for value in phrases[:3]]
    label = " · ".join(label_terms)
    evidence_text = ", ".join(display_phrase(value) for value in phrases)
    return {
        "label": label,
        "focus": f"a shared lexical emphasis on {evidence_text}",
        "distinction": (
            "This is a deterministic statistical label generated from current "
            "title coverage and cross-cluster exclusivity; it remains a candidate "
            "for human review."
        ),
        "evidence": phrases,
    }


def extract_rankings(papers: list[dict[str, object]]) -> dict[int, dict[str, list[str]]]:
    frame = pd.DataFrame(papers)
    clustered = frame[frame["cluster"].astype(int).ge(0)].copy()
    title_text = clustered["title"].map(normalize_text)
    texts = title_text.tolist()

    vectorizer = CountVectorizer(
        ngram_range=(2, 3),
        min_df=2,
        max_df=0.92,
        stop_words=sorted(STOP_WORDS),
        binary=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9-]+\b",
    )
    matrix = vectorizer.fit_transform(texts)
    vocabulary = np.asarray(vectorizer.get_feature_names_out())
    usable = np.asarray([phrase_is_usable(term) for term in vocabulary])
    global_df = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    cluster_ids = sorted(clustered["cluster"].astype(int).unique())
    cluster_presence = np.zeros((len(cluster_ids), len(vocabulary)), dtype=float)
    id_to_row = {cluster_id: row for row, cluster_id in enumerate(cluster_ids)}

    for cluster_id in cluster_ids:
        row_indices = np.flatnonzero(clustered["cluster"].astype(int).to_numpy() == cluster_id)
        cluster_presence[id_to_row[cluster_id]] = np.asarray(
            matrix[row_indices].sum(axis=0)
        ).ravel()

    output: dict[int, dict[str, list[str]]] = {}
    for cluster_id in cluster_ids:
        row = id_to_row[cluster_id]
        presence = cluster_presence[row]
        size = int((clustered["cluster"].astype(int) == cluster_id).sum())
        coverage = presence / max(size, 1)
        exclusivity = (presence + 0.25) / (global_df + 0.25)
        frequency_rank = rank_percentile(coverage)
        exclusivity_rank = rank_percentile(exclusivity)
        frex = 1.0 / (
            0.60 / np.maximum(frequency_rank, 1e-9)
            + 0.40 / np.maximum(exclusivity_rank, 1e-9)
        )
        phrase_length_bonus = np.asarray(
            [1.0 + 0.08 * (len(term.split()) - 2) for term in vocabulary]
        )
        distinctive_score = frex * phrase_length_bonus
        core_score = coverage * np.log1p(global_df.max() / np.maximum(global_df, 1.0))
        supported = presence >= 2
        distinctive_score[~usable | ~supported] = -1
        core_score[~usable | ~supported] = -1

        core_ranked = vocabulary[np.flatnonzero(core_score >= 0)[np.argsort(-core_score[core_score >= 0])]].tolist()
        distinctive_ranked = vocabulary[
            np.flatnonzero(distinctive_score >= 0)[np.argsort(-distinctive_score[distinctive_score >= 0])]
        ].tolist()
        core = select_nonredundant(core_ranked, 1)
        distinctive = select_nonredundant(distinctive_ranked, 8, blocked=core)
        output[cluster_id] = {
            "core": core,
            "distinctive": distinctive,
        }
    return output


def refine_payload(payload: dict[str, object]) -> dict[str, object]:
    papers = payload["papers"]
    for paper in papers:
        paper["design_knowledge_form"] = "UMAP 10D + HDBSCAN"
    rankings = extract_rankings(papers)
    cluster_records = {
        int(record["cluster"]): record for record in payload["clusters"]
    }

    for cluster_id, ranking in rankings.items():
        members = [paper for paper in papers if int(paper["cluster"]) == cluster_id]
        members.sort(key=lambda paper: int(paper.get("representative_rank", 999999)))
        profile = choose_profile(ranking, members) or statistical_profile(
            ranking, members
        )
        statistical_phrases = select_nonredundant(
            ranking["core"] + ranking["distinctive"], 4, blocked=profile["evidence"]
        )
        evidence = profile["evidence"] + statistical_phrases
        label = profile["label"]
        summary = (
            f"This {len(members)}-paper cluster focuses on {profile['focus']}. "
            f"{profile['distinction']}"
        )

        for paper in members:
            paper["cluster_theme_terms"] = " | ".join(evidence)
            paper["cluster_label_candidate"] = label
            paper["distinguishing_evidence_terms"] = ", ".join(evidence)
            paper["cluster_summary_candidate"] = summary
            paper["lda_topic_words"] = " | ".join(evidence)

        cluster = cluster_records[cluster_id]
        cluster["theme"] = " | ".join(evidence)
        cluster["label"] = label
        cluster["summary"] = summary
        cluster.pop("representatives", None)

    for cluster in payload["clusters"]:
        cluster.pop("representatives", None)

    payload["title"] = "UMAP 10D + HDBSCAN · hdbscan_mcs8_ms1"

    return payload


def update_csv(path: Path, payload: dict[str, object]) -> None:
    frame = pd.read_csv(path).fillna("")
    by_id = {str(paper["paper_id"]): paper for paper in payload["papers"]}
    fields = [
        "cluster_label_candidate",
        "distinguishing_evidence_terms",
        "cluster_summary_candidate",
    ]
    for field in fields:
        frame[field] = frame["paper_id"].astype(str).map(
            lambda paper_id: by_id[paper_id][field]
        )
    frame.to_csv(path, index=False)


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# UMAP 10D + HDBSCAN",
        "",
        "Cluster membership is unchanged. Labels use deterministic phrase extraction, frequency–exclusivity reranking, abbreviation expansion, and lexical deduplication; no LLM is used.",
        "",
    ]
    for cluster in sorted(payload["clusters"], key=lambda item: int(item["cluster"])):
        lines.extend(
            [
                f"## {cluster['label']} ({cluster['count']} papers)",
                "",
                str(cluster["summary"]),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--explorer", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    explorer_path = Path(args.explorer)
    page = explorer_path.read_text(encoding="utf-8")
    match = PAYLOAD_RE.search(page)
    if not match:
        raise ValueError(f"No explorer payload found in {explorer_path}")
    payload = refine_payload(json.loads(match.group(2)))

    for cluster in sorted(payload["clusters"], key=lambda item: int(item["cluster"])):
        if int(cluster["cluster"]) >= 0:
            print(f"Cluster {int(cluster['cluster']) + 1}: {cluster['label']}")

    if args.dry_run:
        return

    updated_page = page[: match.start(2)] + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ) + page[match.end(2) :]
    updated_page = updated_page.replace(
        "Zhicheng workflow · UMAP 10D + HDBSCAN · hdbscan_mcs8_ms1",
        "UMAP 10D + HDBSCAN",
    )
    updated_page = updated_page.replace(
        '            <span class="pill">Rep rank ${p.representative_rank}</span>\n', ""
    )
    updated_page = updated_page.replace(
        '            <span class="pill">Medoid rank ${p.medoid_rank}</span>\n', ""
    )
    updated_page = updated_page.replace(
        '<span class="pill">Form: ${escapeHtml(p.design_knowledge_form || \'n/a\')}</span>',
        '<span class="pill">Method: UMAP 10D + HDBSCAN</span>',
    )
    explorer_path.write_text(updated_page, encoding="utf-8")
    update_csv(Path(args.csv), payload)
    write_markdown(Path(args.summary), payload)


if __name__ == "__main__":
    main()
