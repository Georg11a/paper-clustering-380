#!/usr/bin/env python3
"""Refresh generated explorer labels with design-knowledge contribution claims."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from research_typology import (
    classify_contribution,
    classify_contribution_subtype,
    classify_domains,
    contribution_definition,
    contribution_summary_fields,
    contribution_summary_template,
)
from theory_typology import classify_theory_move


FORM_PATTERNS = {
    "tacit design knowledge": ["tacit knowledge", "tacit design knowledge", "craft knowledge", "expertise"],
    "design rationale": ["design rationale", "rationale", "decision rationale", "traceability"],
    "design theory": ["design theory", "theory", "theoretical framework", "mid-range theory"],
    "design patterns": ["design pattern", "design patterns", "pattern language", "reusable solution"],
    "design guidelines": ["design guideline", "design guidelines", "guideline", "guidelines"],
    "design heuristics": ["design heuristic", "design heuristics", "heuristic", "heuristics"],
    "design principles": ["design principle", "design principles", "principle", "principles"],
    "design frameworks": ["design framework", "design frameworks", "framework", "frameworks"],
    "design methods": ["design method", "design methods", "methodology", "methodologies"],
    "design rules": ["design rule", "design rules", "rule", "rules"],
    "design expertise": ["design expertise", "expertise", "expert knowledge"],
    "design procedures": ["design procedure", "design procedures", "procedure", "procedures"],
    "design knowledge": ["design knowledge", "knowledge base", "knowledge system"],
}

ACTION_PATTERNS = {
    "defines": ["define", "defines", "defined", "definition", "conceptualize", "conceptualizes", "conceptualization"],
    "organizes": ["organize", "organizes", "organized", "taxonomy", "typology", "classification", "categorize", "catalog"],
    "translates": ["translate", "translates", "operationalize", "operationalizes", "actionable", "apply", "application"],
    "represents": ["represent", "represents", "representation", "document", "documentation", "trace", "traceable"],
    "captures": ["capture", "captures", "codify", "codifies", "externalize", "transfer", "share", "communicate"],
    "adapts": ["adapt", "adapts", "adapted", "tailor", "domain-specific", "context-specific"],
    "evaluates": ["evaluate", "evaluates", "validated", "validation", "empirical", "evidence", "assessment"],
    "synthesizes": ["synthesize", "synthesizes", "synthesis", "review", "literature review", "systematic review"],
}

ACTION_DISPLAY = {
    "defines": "defines and conceptualizes",
    "organizes": "organizes and classifies",
    "translates": "translates into actionable guidance",
    "represents": "represents and documents",
    "captures": "captures and transfers",
    "adapts": "adapts to a specific context",
    "evaluates": "evaluates with evidence",
    "synthesizes": "synthesizes prior work on",
}

ACTION_NOUNS = {
    "defines": "definition and conceptualization",
    "organizes": "organization and classification",
    "translates": "translation into actionable guidance",
    "represents": "representation and documentation",
    "captures": "capture and transfer",
    "adapts": "context-specific adaptation",
    "evaluates": "evidence-based evaluation",
    "synthesizes": "synthesis of prior work",
}

ACTION_LABELS = {
    "defines": "Defines {form} Through Conceptual Framing",
    "organizes": "Organizes {form} into Taxonomies or Frameworks",
    "translates": "Translates {form} into Actionable Design Guidance",
    "represents": "Represents {form} as Traceable Decision Knowledge",
    "captures": "Frames {form} as Expertise to Capture and Transfer",
    "adapts": "Adapts {form} for Domain-Specific Use",
    "evaluates": "Evaluates {form} Through Empirical Evidence",
    "synthesizes": "Synthesizes {form} into Shared Design Constructs",
}

CONTEXT_ROLE_RULES = [
    {
        "needles": ["usability", "guidelines", "heuristics", "visual design", "communication", "severe weather"],
        "label": "Interface guidelines and heuristics as evaluative design knowledge",
        "form": "Design Guidelines",
        "action": "evaluates with evidence",
        "contribution": "frames interface guidelines and heuristics as evaluative design knowledge",
        "summary": (
            "This context cluster treats design knowledge as guidelines, heuristics, or evaluative criteria for interface and communication design. "
            "The shared role of knowledge is to diagnose design quality and support better design decisions in concrete interface settings."
        ),
    },
    {
        "needles": ["visualization", "sonification", "aesthetics", "audio visual", "hierarchical", "atlases"],
        "label": "Visual and sensory representations as communicable design knowledge",
        "form": "Design Knowledge",
        "action": "represents and documents",
        "contribution": "represents visual and sensory design knowledge in communicable forms",
        "summary": (
            "This context cluster frames design knowledge through visual, sensory, and hierarchical representation. "
            "Its papers ask how knowledge can be made perceptible, communicable, and inspectable through visualization, sonification, or structured models."
        ),
    },
    {
        "needles": ["hci", "human-computer", "educational access", "neurodiverse", "task", "interface principles"],
        "label": "HCI principles as task and access guidance",
        "form": "Design Principles",
        "action": "translates into actionable guidance",
        "contribution": "translates HCI principles into task and access guidance",
        "summary": (
            "This context cluster treats design knowledge as principles for human-computer interaction, task support, and access. "
            "The cluster links abstract guidance to concrete interface, software, or educational design situations."
        ),
    },
    {
        "needles": ["dark patterns", "design patterns", "pattern language", "hri", "human-robot", "formal", "software patterns"],
        "label": "Pattern languages as reusable knowledge for software and HRI design",
        "form": "Design Patterns",
        "action": "organizes and classifies",
        "contribution": "organizes software and HRI design knowledge as reusable pattern languages",
        "summary": (
            "This context cluster treats design knowledge as pattern language: reusable forms that name, compare, and transfer recurring design situations. "
            "The cluster includes both constructive patterns and critical pattern work such as dark-pattern analysis."
        ),
    },
    {
        "needles": ["landscape architecture", "fashion", "relational", "pragmatism", "thriving", "human actors", "human crafters"],
        "label": "Design theory as situated practice across domains",
        "form": "Design Theory",
        "action": "adapts to a specific context",
        "contribution": "situates design theory across domains such as fashion, landscape, and work systems",
        "summary": (
            "This context cluster uses domain-specific settings to show design theory as situated practice rather than a universal rule set. "
            "Its papers use fields such as fashion, landscape architecture, and work systems to test how design knowledge changes across contexts."
        ),
    },
    {
        "needles": ["ethics", "game", "immersive", "interactive narrative", "trajectories", "practice", "journeys"],
        "label": "Practice-based evidence for interactive experience design",
        "form": "Design Knowledge",
        "action": "evaluates with evidence",
        "contribution": "builds practice-based evidence for interactive experience design",
        "summary": (
            "This context cluster treats situated design practice as evidence for broader claims about interactive experience design. "
            "The cluster links games, immersive practice, journeys, and ethics-focused cases to transferable design knowledge."
        ),
    },
    {
        "needles": ["ai", "artificial", "transparent", "auditable", "identity", "collaboration industry learning"],
        "label": "AI design patterns as auditable and collaborative knowledge",
        "form": "Design Patterns",
        "action": "translates into actionable guidance",
        "contribution": "translates AI-related design patterns into auditable and collaborative knowledge",
        "summary": (
            "This context cluster treats AI-related design knowledge as patterns, identity frameworks, and guidance for collaboration or auditability. "
            "Its role is to turn emerging AI design situations into reusable and inspectable knowledge."
        ),
    },
    {
        "needles": ["tacit", "designerly", "graphic", "situated effects", "inexperienced designers", "digital"],
        "label": "Tacit designerly knowledge as situated expertise",
        "form": "Tacit Design Knowledge",
        "action": "captures and transfers",
        "contribution": "captures tacit designerly knowledge as situated expertise",
        "summary": (
            "This context cluster frames design knowledge as tacit, situated expertise embedded in designerly practice. "
            "The shared concern is how such knowledge can be characterized, surfaced, and transferred without losing its context."
        ),
    },
    {
        "needles": ["cpm", "dfx", "product", "causal", "representation", "near-field", "industrial"],
        "label": "Product-development knowledge as traceable design rationale",
        "form": "Design Rationale",
        "action": "represents and documents",
        "contribution": "represents product-development knowledge as traceable design rationale",
        "summary": (
            "This context cluster treats design knowledge as product-development and decision knowledge: something that can be represented, traced, and reused across engineering or industrial design work. "
            "Its papers emphasize rationale, causal links, product features, and decision support rather than design knowledge as a general topic."
        ),
    },
    {
        "needles": ["principle", "prescriptive", "boundary", "formulation", "dsr", "design science", "codification"],
        "label": "Design principles as prescriptive knowledge for transfer",
        "form": "Design Principles",
        "action": "translates into actionable guidance",
        "contribution": "translates design principles into prescriptive knowledge for transfer",
        "summary": (
            "This context cluster frames design knowledge as prescriptive principles that can travel from research into design practice. "
            "The shared concern is how principles are formulated, bounded, justified, and made usable in design science or practice settings."
        ),
    },
    {
        "needles": ["object oriented", "ontology", "heuristics best practices", "reference", "paradigm", "reusable"],
        "label": "Object-oriented design knowledge as ontology and reusable heuristics",
        "form": "Design Heuristics",
        "action": "organizes and classifies",
        "contribution": "organizes object-oriented design knowledge as ontology and reusable heuristics",
        "summary": (
            "This context cluster understands design knowledge as a formal ontology or reusable set of heuristics and best practices. "
            "Its contribution is to classify and stabilize design knowledge so it can be accumulated, measured, and reused."
        ),
    },
    {
        "needles": ["studio", "teacher", "education", "learning", "competency", "curriculum", "pedagogical"],
        "label": "Studio and education contexts as sites for developing design knowledge",
        "form": "Design Knowledge",
        "action": "adapts to a specific context",
        "contribution": "situates design knowledge in studio, teaching, and learning contexts",
        "summary": (
            "This context cluster treats education, studio learning, and teacher practice as sites where design knowledge is formed, adapted, and transferred. "
            "The important point is not education as a domain label, but how learning environments make design knowledge visible and teachable."
        ),
    },
    {
        "needles": ["game", "health", "public", "dark patterns", "interactive", "narrative", "evidence", "phi"],
        "label": "Evidence-based design knowledge for interactive and public-facing systems",
        "form": "Design Knowledge",
        "action": "evaluates with evidence",
        "contribution": "builds evidence-based design knowledge for interactive and public-facing systems",
        "summary": (
            "This context cluster uses applied settings such as games, health, public systems, and dark-pattern analysis to build evidence-based design knowledge. "
            "The cluster's role is to show how situated cases become transferable claims about design practice."
        ),
    },
    {
        "needles": ["platform", "digital", "multi sided", "taxonomy", "ontology"],
        "label": "Platform knowledge as taxonomy and ontology",
        "form": "Design Frameworks",
        "action": "organizes and classifies",
        "contribution": "organizes digital-platform design knowledge as taxonomy and ontology",
        "summary": (
            "This small context cluster frames design knowledge as a taxonomy or ontology for digital platforms. "
            "Because the group is very small, it should be read as a specific organizing construct rather than a robust broad theme."
        ),
    },
    {
        "needles": ["hri", "designerly", "intermediate", "competence", "social drones", "creation"],
        "label": "Designerly knowledge as intermediate-level construct",
        "form": "Design Knowledge",
        "action": "defines and conceptualizes",
        "contribution": "conceptualizes designerly knowledge as an intermediate-level construct",
        "summary": (
            "This context cluster focuses on designerly or intermediate-level knowledge: knowledge that sits between specific artifacts and general theory. "
            "Its papers ask how design work produces conceptual contributions that can be communicated beyond a single case."
        ),
    },
    {
        "needles": ["strategic", "ecosystems", "toulmin", "inquiry", "data quality", "value"],
        "label": "Strategic design knowledge as reasoning for inquiry and value",
        "form": "Design Knowledge",
        "action": "defines and conceptualizes",
        "contribution": "frames strategic design knowledge as reasoning for inquiry, value, and system-level design",
        "summary": (
            "This context cluster is a looser conceptual group around strategic reasoning, inquiry tools, value, and ecosystem-level design. "
            "It is best read as a provisional theme about how design knowledge supports reasoning across complex design situations."
        ),
    },
]


def normalize(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def title_case(phrase: str) -> str:
    special = {"ai": "AI", "hci": "HCI", "human-ai": "Human-AI", "dsr": "DSR"}
    small = {"and", "or", "of", "for", "in", "with", "to", "as", "into"}
    words = []
    for word in phrase.split():
        low = word.lower()
        if low in special:
            words.append(special[low])
        elif low in small:
            words.append(low)
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)


def score_patterns(text: str, patterns: dict[str, list[str]]) -> list[tuple[str, int]]:
    scored = []
    for label, phrases in patterns.items():
        score = 0
        for phrase in phrases:
            pattern = re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b")
            hits = len(pattern.findall(text))
            score += hits * (3 if " " in phrase else 1)
        if score:
            scored.append((label, score))
    scored.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    return scored


def group_text(group: pd.DataFrame) -> str:
    columns = [
        "title",
        "abstract",
        "primary_reason",
        "val_reason",
        "cluster_theme_terms",
        "facet_method_or_lens",
        "facet_contribution_or_outcome",
    ]
    return normalize(" ".join(" ".join(group[col].fillna("").astype(str)) for col in columns if col in group))


def fallback_form(group: pd.DataFrame) -> str:
    if "keyword" not in group:
        return "design knowledge"
    keyword = normalize(group["keyword"].iloc[0])
    if keyword in FORM_PATTERNS:
        return keyword
    return "design knowledge"


def claim_for_group(group: pd.DataFrame) -> dict[str, str]:
    text = group_text(group)
    forms = score_patterns(text, FORM_PATTERNS)
    actions = score_patterns(text, ACTION_PATTERNS)
    form = forms[0][0] if forms else fallback_form(group)
    action = actions[0][0] if actions else "synthesizes"
    other_actions = [name for name, _ in actions[1:3]]
    other_forms = [name for name, _ in forms[1:3] if name != form]

    form_label = title_case(form)
    action_text = ACTION_DISPLAY.get(action, action)
    label = ACTION_LABELS[action].format(form=form_label)

    contributions = []
    if "facet_contribution_or_outcome" in group and str(group["facet_contribution_or_outcome"].iloc[0]).strip():
        contributions.append(str(group["facet_contribution_or_outcome"].iloc[0]).strip())
    methods = ""
    if "facet_method_or_lens" in group and str(group["facet_method_or_lens"].iloc[0]).strip():
        methods = str(group["facet_method_or_lens"].iloc[0]).strip()
    contexts = ""
    if "facet_population_or_context" in group and str(group["facet_population_or_context"].iloc[0]).strip():
        contexts = str(group["facet_population_or_context"].iloc[0]).strip()

    ranked = group.sort_values([c for c in ["representative_rank", "medoid_rank"] if c in group.columns])
    titles = [str(value) for value in ranked.get("title", pd.Series(dtype=str)).head(2) if str(value).strip()]

    sentences = [
        (
            f"This cluster {action_text} {form}, emphasizing how a design-knowledge construct is made explicit, "
            "organized, or put to work rather than only where it is applied."
        )
    ]
    if other_forms:
        sentences.append("Related knowledge forms include " + ", ".join(title_case(item) for item in other_forms) + ".")
    if other_actions:
        sentences.append("The shared move also involves " + " and ".join(ACTION_NOUNS.get(item, item) for item in other_actions) + ".")
    if contributions:
        sentences.append("Its contribution pattern is coded as " + contributions[0] + ".")
    if methods or contexts:
        facet_bits = []
        if methods:
            facet_bits.append("methods or lenses such as " + methods)
        if contexts:
            facet_bits.append("contexts such as " + contexts)
        sentences.append("Application and method terms are supporting facets, especially " + " and ".join(facet_bits) + ".")
    if titles:
        sentences.append("Representative papers include " + "; ".join(titles) + ".")

    return {
        "cluster_label_candidate": label,
        "cluster_summary_candidate": " ".join(sentences),
        "design_knowledge_form": form_label,
        "design_knowledge_action": action_text,
        "design_knowledge_contribution": f"{action_text} {form_label}",
    }


def context_role_claim(group: pd.DataFrame) -> dict[str, str] | None:
    text = group_text(group)
    best = None
    best_score = 0
    for rule in CONTEXT_ROLE_RULES:
        score = sum(1 for needle in rule["needles"] if needle in text)
        if score > best_score:
            best = rule
            best_score = score
    if not best or best_score == 0:
        return None

    ranked = group.sort_values([c for c in ["representative_rank", "medoid_rank"] if c in group.columns])
    titles = [str(value) for value in ranked.get("title", pd.Series(dtype=str)).head(2) if str(value).strip()]
    representative = ""
    if titles:
        representative = " Representative papers include " + "; ".join(titles) + "."
    return {
        "cluster_label_candidate": best["label"],
        "cluster_summary_candidate": best["summary"] + representative,
        "design_knowledge_form": best["form"],
        "design_knowledge_action": best["action"],
        "design_knowledge_contribution": best["contribution"],
    }


def first_facet_value(group: pd.DataFrame, columns: list[str]) -> str:
    for column in columns:
        if column not in group.columns:
            continue
        value = str(group.iloc[0].get(column, "") or "").strip()
        if value:
            return value.split(",")[0].strip()
    return ""


def typology_claim_for_group(group: pd.DataFrame) -> dict[str, str]:
    records = group.to_dict("records")
    contribution = classify_contribution(records)
    subtype = classify_contribution_subtype(records, contribution.primary_key)
    domains = classify_domains(records)
    form = title_case(fallback_form(group))
    label = f"{form}: {contribution.primary_label}"
    if contribution.secondary_label:
        label += f" + {contribution.secondary_label}"
    if subtype.label and contribution.primary_key != "theoretical":
        label += f" — {subtype.label}"
    theory_applicable = "theoretical" in {contribution.primary_key, contribution.secondary_key}
    theory = classify_theory_move(records) if theory_applicable else None
    if theory is not None:
        label += f" — {theory.label}"
    label += f" | Primary Domain: {domains.primary_label}"

    ranked = group.sort_values([c for c in ["representative_rank", "medoid_rank"] if c in group.columns])
    titles = [str(value) for value in ranked.get("title", pd.Series(dtype=str)).head(2) if str(value).strip()]
    if contribution.primary_key == "unclear":
        summary = (
            "The deterministic first-pass typology did not find enough evidence to assign a primary "
            "research contribution type. It remains flagged for human review."
        )
    elif contribution.primary_key == "mixed":
        summary = (
            "This cluster contains multiple contribution types without enough shared evidence to identify "
            f"one dominant type. Candidate signals are {contribution.patterns_text}."
        )
    else:
        contribution_phrase = contribution.primary_label.lower()
        if subtype.label:
            contribution_phrase += f", specifically {subtype.label.lower()}"
        summary = f"This cluster primarily makes a {contribution_phrase}."
    if contribution.secondary_label:
        summary += f" A genuinely equivalent secondary type is {contribution.secondary_label.lower()}."
    summary += f" Its strongest supported application domain is {domains.primary_label}."
    if domains.additional_labels_text:
        summary += f" Additional supported settings are {domains.additional_labels_text}."
    if theory is not None:
        if theory.key == "unclear":
            summary += " Path 1 found the theoretical contribution relevant but could not assign an unambiguous theory move."
        else:
            summary += f" Within that theoretical contribution, Path 1 codes the move as {theory.label.lower()}."
    if titles:
        summary += " Representative papers include " + "; ".join(titles) + "."
    theory_key = theory.key if theory is not None else "not_applicable"
    theory_label = theory.label if theory is not None else "Not Applicable — Contribution Is Not Theoretical"
    return {
        "cluster_label_candidate": label,
        "cluster_summary_candidate": summary,
        "design_knowledge_form": form,
        "design_knowledge_action": contribution.primary_label,
        "design_knowledge_role": "Automatic first-pass coding",
        "design_knowledge_contribution": contribution.primary_label,
        "contribution_type_key": contribution.primary_key,
        "contribution_type": contribution.primary_label,
        "contribution_type_secondary": contribution.secondary_label,
        "contribution_type_patterns": contribution.patterns_text,
        "contribution_type_support": contribution.support_text,
        "contribution_subtype_key": subtype.key,
        "contribution_subtype": subtype.label,
        "contribution_subtype_patterns": subtype.patterns_text,
        "contribution_subtype_support": subtype.support_text,
        "contribution_type_definition": contribution_definition(contribution.primary_key),
        "contribution_summary_fields": contribution_summary_fields(contribution.primary_key),
        "contribution_summary_template": contribution_summary_template(contribution.primary_key),
        "application_domains": domains.labels_text,
        "primary_application_domain": domains.primary_label,
        "additional_application_domains": domains.additional_labels_text,
        "application_domain_patterns": domains.patterns_text,
        "application_domain_support": domains.support_text,
        "application_domain_definitions": domains.definitions_text,
        "theory_move_key": theory_key,
        "theory_move": theory_label,
        "theory_move_patterns": theory.patterns_text if theory is not None else "Not applicable",
        "theory_move_support": theory.support_text if theory is not None else "Not applicable",
    }


def refresh_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    if "cluster" not in df.columns:
        return df
    for col in [
        "design_knowledge_form",
        "design_knowledge_action",
        "design_knowledge_role",
        "design_knowledge_contribution",
        "contribution_type_key",
        "contribution_type",
        "contribution_type_secondary",
        "contribution_type_patterns",
        "contribution_type_support",
        "contribution_subtype_key",
        "contribution_subtype",
        "contribution_subtype_patterns",
        "contribution_subtype_support",
        "contribution_type_definition",
        "contribution_summary_fields",
        "contribution_summary_template",
        "application_domains",
        "primary_application_domain",
        "additional_application_domains",
        "application_domain_patterns",
        "application_domain_support",
        "application_domain_definitions",
        "theory_move_key",
        "theory_move",
        "theory_move_patterns",
        "theory_move_support",
    ]:
        if col not in df.columns:
            df[col] = ""
    for cluster, group in df.groupby("cluster"):
        mask = df["cluster"].astype(str) == str(cluster)
        if str(cluster) == "-1":
            claim = {
                "cluster_label_candidate": "Unclustered papers",
                "cluster_summary_candidate": (
                    "HDBSCAN did not assign these papers to a dense cluster. "
                    "They are retained for inspection but should not be interpreted as a thematic cluster."
                ),
                "design_knowledge_form": "n/a",
                "design_knowledge_action": "n/a",
                "design_knowledge_contribution": "Not interpreted as a cluster theme.",
                "contribution_type_key": "n/a",
                "contribution_type": "n/a",
                "contribution_type_secondary": "",
                "contribution_type_patterns": "n/a",
                "contribution_type_support": "n/a",
                "contribution_subtype_key": "n/a",
                "contribution_subtype": "n/a",
                "contribution_subtype_patterns": "n/a",
                "contribution_subtype_support": "n/a",
                "contribution_type_definition": "n/a",
                "contribution_summary_fields": "n/a",
                "contribution_summary_template": "n/a",
                "application_domains": "n/a",
                "primary_application_domain": "n/a",
                "additional_application_domains": "",
                "application_domain_patterns": "n/a",
                "application_domain_support": "n/a",
                "application_domain_definitions": "n/a",
                "theory_move_key": "n/a",
                "theory_move": "n/a",
                "theory_move_patterns": "n/a",
                "theory_move_support": "n/a",
            }
        else:
            claim = typology_claim_for_group(group)
        for key, value in claim.items():
            df.loc[mask, key] = value
    df.to_csv(path, index=False)
    return df


def write_summary(df: pd.DataFrame, path: Path) -> None:
    lines = ["# Clustering Summary", "", "## Cluster Themes", ""]
    for cluster in sorted(df["cluster"].unique(), key=lambda value: int(value) if str(value).lstrip("-").isdigit() else 999):
        group = df[df["cluster"].astype(str) == str(cluster)]
        heading = "Unclustered papers" if str(cluster) == "-1" else f"Cluster {cluster}"
        lines.append(f"### {heading} ({len(group)} papers)")
        first = group.iloc[0]
        lines.append(f"Label candidate: {first.get('cluster_label_candidate', '')}")
        lines.append(f"Primary contribution: {first.get('contribution_type', '')}")
        lines.append(f"Secondary contribution: {first.get('contribution_type_secondary', '') or 'n/a'}")
        lines.append(f"Contribution support: {first.get('contribution_type_support', '')}")
        lines.append(f"Contribution subtype: {first.get('contribution_subtype', '') or 'n/a'}")
        lines.append(f"Contribution-subtype support: {first.get('contribution_subtype_support', '')}")
        lines.append(f"Contribution definition: {first.get('contribution_type_definition', '')}")
        lines.append(f"Contribution patterns: {first.get('contribution_type_patterns', '')}")
        lines.append(f"Contribution summary fields: {first.get('contribution_summary_fields', '')}")
        lines.append(f"Contribution summary template: {first.get('contribution_summary_template', '')}")
        lines.append(f"Application domains: {first.get('application_domains', '')}")
        lines.append(f"Primary application domain: {first.get('primary_application_domain', '')}")
        lines.append(f"Additional application domains: {first.get('additional_application_domains', '') or 'n/a'}")
        lines.append(f"Domain support: {first.get('application_domain_support', '')}")
        lines.append(f"Domain definitions: {first.get('application_domain_definitions', '')}")
        lines.append(f"Theory move: {first.get('theory_move', '')}")
        lines.append(f"Theory-move support: {first.get('theory_move_support', '')}")
        lines.append(f"Matched patterns: {first.get('theory_move_patterns', '')}")
        lines.append(f"Summary candidate: {first.get('cluster_summary_candidate', '')}")
        if first.get("design_knowledge_contribution", ""):
            lines.append(f"Design-knowledge contribution: {first.get('design_knowledge_contribution', '')}")
        lines.append(f"Theme words: {first.get('cluster_theme_terms', '')}")
        lines.append("")
        lines.append("Representative papers:")
        ranked = group.sort_values([c for c in ["representative_rank", "medoid_rank"] if c in group.columns])
        for _, row in ranked.head(3).iterrows():
            year = str(row.get("year", "")).replace(".0", "")
            lines.append(f"- {year}: {row.get('title', '')}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def refresh_html(path: Path, df: pd.DataFrame) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'(<script id="payload" type="application/json">)(.*?)(</script>)', text, re.S)
    if not match:
        return
    payload = json.loads(match.group(2))
    by_paper = {
        str(row["paper_id"]): row.to_dict()
        for _, row in df.iterrows()
        if "paper_id" in df.columns
    }
    for paper in payload.get("papers", []):
        row = by_paper.get(str(paper.get("paper_id", "")))
        if not row:
            continue
        for key in [
            "cluster_label_candidate",
            "cluster_summary_candidate",
            "design_knowledge_form",
            "design_knowledge_action",
            "design_knowledge_role",
            "design_knowledge_contribution",
            "contribution_type_key",
            "contribution_type",
            "contribution_type_secondary",
            "contribution_type_patterns",
            "contribution_type_support",
            "contribution_subtype_key",
            "contribution_subtype",
            "contribution_subtype_patterns",
            "contribution_subtype_support",
            "contribution_type_definition",
            "contribution_summary_fields",
            "contribution_summary_template",
            "application_domains",
            "primary_application_domain",
            "additional_application_domains",
            "application_domain_patterns",
            "application_domain_support",
            "application_domain_definitions",
            "theory_move_key",
            "theory_move",
            "theory_move_patterns",
            "theory_move_support",
        ]:
            paper[key] = str(row.get(key, ""))

    by_cluster = {}
    for cluster, group in df.groupby("cluster"):
        first = group.iloc[0]
        by_cluster[str(cluster)] = first
    for cluster in payload.get("clusters", []):
        first = by_cluster.get(str(cluster.get("cluster", "")))
        if first is None:
            continue
        cluster["label"] = str(first.get("cluster_label_candidate", ""))
        cluster["summary"] = str(first.get("cluster_summary_candidate", ""))
        cluster["contribution_type"] = str(first.get("contribution_type", ""))
        cluster["contribution_type_secondary"] = str(first.get("contribution_type_secondary", ""))
        cluster["contribution_type_patterns"] = str(first.get("contribution_type_patterns", ""))
        cluster["contribution_type_support"] = str(first.get("contribution_type_support", ""))
        cluster["contribution_subtype"] = str(first.get("contribution_subtype", ""))
        cluster["contribution_subtype_patterns"] = str(first.get("contribution_subtype_patterns", ""))
        cluster["contribution_subtype_support"] = str(first.get("contribution_subtype_support", ""))
        cluster["contribution_type_definition"] = str(first.get("contribution_type_definition", ""))
        cluster["contribution_summary_fields"] = str(first.get("contribution_summary_fields", ""))
        cluster["contribution_summary_template"] = str(first.get("contribution_summary_template", ""))
        cluster["application_domains"] = str(first.get("application_domains", ""))
        cluster["primary_application_domain"] = str(first.get("primary_application_domain", ""))
        cluster["additional_application_domains"] = str(first.get("additional_application_domains", ""))
        cluster["application_domain_patterns"] = str(first.get("application_domain_patterns", ""))
        cluster["application_domain_support"] = str(first.get("application_domain_support", ""))
        cluster["application_domain_definitions"] = str(first.get("application_domain_definitions", ""))
        cluster["theory_move"] = str(first.get("theory_move", ""))
        cluster["theory_move_patterns"] = str(first.get("theory_move_patterns", ""))
        cluster["theory_move_support"] = str(first.get("theory_move_support", ""))

    new_payload = json.dumps(payload, ensure_ascii=False)
    text = text[: match.start(2)] + new_payload + text[match.end(2) :]

    old = """        <div class="section-title">Cluster Summary Candidate</div>
        <div class="abstract">${escapeHtml(p.cluster_summary_candidate)}</div>
        <div class="section-title">Paper-Oriented Facets</div>"""
    new = """        <div class="section-title">Cluster Summary Candidate</div>
        <div class="abstract">${escapeHtml(p.cluster_summary_candidate)}</div>
        <div class="section-title">Design-Knowledge Contribution</div>
        <div>
          <span class="pill">Form: ${escapeHtml(p.design_knowledge_form || 'n/a')}</span>
          <span class="pill">Action: ${escapeHtml(p.design_knowledge_action || 'n/a')}</span>
        </div>
        <div class="abstract">${escapeHtml(p.design_knowledge_contribution || '')}</div>
        <div class="section-title">Paper-Oriented Facets</div>"""
    if "Design-Knowledge Contribution" not in text:
        text = text.replace(old, new)
    old_typology = """        <div class="section-title">Design-Knowledge Contribution</div>
        <div>
          <span class="pill">Form: ${escapeHtml(p.design_knowledge_form || 'n/a')}</span>
          <span class="pill">Action: ${escapeHtml(p.design_knowledge_action || 'n/a')}</span>
          <span class="pill">Auto role: ${escapeHtml(p.design_knowledge_role || 'n/a')}</span>
        </div>
        <div class="abstract">${escapeHtml(p.design_knowledge_contribution || '')}</div>"""
    new_typology = """        <div class="section-title">Rule-Based Theory Typology</div>
        <div>
          <span class="pill">Form: ${escapeHtml(p.design_knowledge_form || 'n/a')}</span>
          <span class="pill">Theory move: ${escapeHtml(p.theory_move || 'n/a')}</span>
          <span class="pill">Support: ${escapeHtml(p.theory_move_support || 'n/a')}</span>
        </div>
        <div class="abstract">${escapeHtml(p.design_knowledge_contribution || '')}</div>
        <div class="meta">Matched patterns: ${escapeHtml(p.theory_move_patterns || 'n/a')}</div>"""
    text = text.replace(old_typology, new_typology)
    old_typology_card = """        <div class="insight-card">
          <div class="section-title">Design-Knowledge Contribution</div>
          <div class="pill-row">
            <span class="pill">Form: ${escapeHtml(p.design_knowledge_form || 'n/a')}</span>
            <span class="pill">Action: ${escapeHtml(p.design_knowledge_action || 'n/a')}</span>
            <span class="pill">Auto role: ${escapeHtml(p.design_knowledge_role || 'n/a')}</span>
          </div>
          <div class="abstract" style="margin-top:8px">${escapeHtml(p.design_knowledge_contribution || '')}</div>
        </div>"""
    new_typology_card = """        <div class="insight-card">
          <div class="section-title">Rule-Based Theory Typology</div>
          <div class="pill-row">
            <span class="pill">Form: ${escapeHtml(p.design_knowledge_form || 'n/a')}</span>
            <span class="pill">Theory move: ${escapeHtml(p.theory_move || 'n/a')}</span>
            <span class="pill">Support: ${escapeHtml(p.theory_move_support || 'n/a')}</span>
          </div>
          <div class="abstract" style="margin-top:8px">${escapeHtml(p.design_knowledge_contribution || '')}</div>
          <div class="meta" style="margin-top:8px">Matched patterns: ${escapeHtml(p.theory_move_patterns || 'n/a')}</div>
        </div>"""
    text = text.replace(old_typology_card, new_typology_card)
    research_typology = """        <div class="section-title">Research Contribution &amp; Domain</div>
        <div>
          <span class="pill">Form: ${escapeHtml(p.design_knowledge_form || 'n/a')}</span>
          <span class="pill">Primary: ${escapeHtml(p.contribution_type || 'n/a')}</span>
          ${p.contribution_subtype ? `<span class="pill">Subtype: ${escapeHtml(p.contribution_subtype)}</span>` : ''}
          ${p.contribution_type_secondary ? `<span class="pill">Secondary: ${escapeHtml(p.contribution_type_secondary)}</span>` : ''}
          <span class="pill">Primary domain: ${escapeHtml(p.primary_application_domain || p.application_domains || 'n/a')}</span>
          ${p.additional_application_domains ? `<span class="pill">Additional domains: ${escapeHtml(p.additional_application_domains)}</span>` : ''}
        </div>
        <div class="meta">Contribution support: ${escapeHtml(p.contribution_type_support || 'n/a')}</div>
        <div class="meta">Contribution definition: ${escapeHtml(p.contribution_type_definition || 'n/a')}</div>
        <div class="meta">Contribution patterns: ${escapeHtml(p.contribution_type_patterns || 'n/a')}</div>
        <div class="meta">Domain support: ${escapeHtml(p.application_domain_support || 'n/a')}</div>
        <div class="meta">Domain definition: ${escapeHtml(p.application_domain_definitions || 'n/a')}</div>
        ${p.theory_move_key !== 'not_applicable' && p.theory_move_key !== 'n/a' ? `
        <div class="section-title">Path 1 Theory Move</div>
        <div>
          <span class="pill">Theory move: ${escapeHtml(p.theory_move || 'n/a')}</span>
          <span class="pill">Support: ${escapeHtml(p.theory_move_support || 'n/a')}</span>
        </div>
        <div class="meta">Matched patterns: ${escapeHtml(p.theory_move_patterns || 'n/a')}</div>` : ''}"""
    research_typology_card = """        <div class="insight-card">
          <div class="section-title">Research Contribution &amp; Domain</div>
          <div class="pill-row">
            <span class="pill">Form: ${escapeHtml(p.design_knowledge_form || 'n/a')}</span>
            <span class="pill">Primary: ${escapeHtml(p.contribution_type || 'n/a')}</span>
            ${p.contribution_subtype ? `<span class="pill">Subtype: ${escapeHtml(p.contribution_subtype)}</span>` : ''}
            ${p.contribution_type_secondary ? `<span class="pill">Secondary: ${escapeHtml(p.contribution_type_secondary)}</span>` : ''}
            <span class="pill">Primary domain: ${escapeHtml(p.primary_application_domain || p.application_domains || 'n/a')}</span>
            ${p.additional_application_domains ? `<span class="pill">Additional domains: ${escapeHtml(p.additional_application_domains)}</span>` : ''}
          </div>
          <div class="meta" style="margin-top:8px">Contribution support: ${escapeHtml(p.contribution_type_support || 'n/a')}</div>
          <div class="meta">Contribution definition: ${escapeHtml(p.contribution_type_definition || 'n/a')}</div>
          <div class="meta">Subtype support: ${escapeHtml(p.contribution_subtype_support || 'n/a')}</div>
          <div class="meta">Contribution patterns: ${escapeHtml(p.contribution_type_patterns || 'n/a')}</div>
          <div class="meta">Domain support: ${escapeHtml(p.application_domain_support || 'n/a')}</div>
          <div class="meta">Domain definition: ${escapeHtml(p.application_domain_definitions || 'n/a')}</div>
          ${p.theory_move_key !== 'not_applicable' && p.theory_move_key !== 'n/a' ? `
          <div class="section-title" style="margin-top:14px">Path 1 Theory Move</div>
          <div class="pill-row">
            <span class="pill">Theory move: ${escapeHtml(p.theory_move || 'n/a')}</span>
            <span class="pill">Support: ${escapeHtml(p.theory_move_support || 'n/a')}</span>
          </div>
          <div class="meta" style="margin-top:8px">Matched patterns: ${escapeHtml(p.theory_move_patterns || 'n/a')}</div>` : ''}
        </div>"""
    text = text.replace(new_typology, research_typology)
    text = text.replace(new_typology_card, research_typology_card)
    old_research_metrics = """          <div class="meta" style="margin-top:8px">Contribution support: ${escapeHtml(p.contribution_type_support || 'n/a')}</div>
          <div class="meta">Contribution patterns: ${escapeHtml(p.contribution_type_patterns || 'n/a')}</div>
          <div class="meta">Domain support: ${escapeHtml(p.application_domain_support || 'n/a')}</div>"""
    new_research_metrics = """          <div class="meta" style="margin-top:8px">Contribution support: ${escapeHtml(p.contribution_type_support || 'n/a')}</div>
          <div class="meta">Contribution definition: ${escapeHtml(p.contribution_type_definition || 'n/a')}</div>
          <div class="meta">Contribution patterns: ${escapeHtml(p.contribution_type_patterns || 'n/a')}</div>
          <div class="meta">Domain support: ${escapeHtml(p.application_domain_support || 'n/a')}</div>
          <div class="meta">Domain definition: ${escapeHtml(p.application_domain_definitions || 'n/a')}</div>"""
    text = text.replace(old_research_metrics, new_research_metrics)
    old_flat_metrics = old_research_metrics.replace(' class="meta" style="margin-top:8px"', ' class="meta"', 1)
    new_flat_metrics = new_research_metrics.replace(' class="meta" style="margin-top:8px"', ' class="meta"', 1)
    text = text.replace(old_flat_metrics, new_flat_metrics)
    if "Contribution definition:" not in text:
        support_line = "<div class=\"meta\">Contribution support: ${escapeHtml(p.contribution_type_support || 'n/a')}</div>"
        text = text.replace(
            support_line,
            support_line + "\n        <div class=\"meta\">Contribution definition: ${escapeHtml(p.contribution_type_definition || 'n/a')}</div>",
        )
    if "Domain definition:" not in text:
        domain_line = "<div class=\"meta\">Domain support: ${escapeHtml(p.application_domain_support || 'n/a')}</div>"
        text = text.replace(
            domain_line,
            domain_line + "\n        <div class=\"meta\">Domain definition: ${escapeHtml(p.application_domain_definitions || 'n/a')}</div>",
        )
    if "Subtype:" not in text:
        primary_line = "<span class=\"pill\">Primary: ${escapeHtml(p.contribution_type || 'n/a')}</span>"
        text = text.replace(
            primary_line,
            primary_line + "\n          ${p.contribution_subtype ? `<span class=\"pill\">Subtype: ${escapeHtml(p.contribution_subtype)}</span>` : ''}",
        )
    text = text.replace(
        "<span class=\"pill\">Domain: ${escapeHtml(p.application_domains || 'n/a')}</span>",
        "<span class=\"pill\">Primary domain: ${escapeHtml(p.primary_application_domain || p.application_domains || 'n/a')}</span>\n"
        "          ${p.additional_application_domains ? `<span class=\"pill\">Additional domains: ${escapeHtml(p.additional_application_domains)}</span>` : ''}",
    )
    if "Subtype support:" not in text:
        definition_line = "<div class=\"meta\">Contribution definition: ${escapeHtml(p.contribution_type_definition || 'n/a')}</div>"
        text = text.replace(
            definition_line,
            definition_line + "\n        <div class=\"meta\">Subtype support: ${escapeHtml(p.contribution_subtype_support || 'n/a')}</div>",
        )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    root = Path("docs/explorer/final")
    count = 0
    for csv_path in root.rglob("clustered_papers.csv"):
        df = refresh_csv(csv_path)
        write_summary(df, csv_path.with_name("cluster_summary.md"))
        html_path = csv_path.with_name("paper_explorer.html")
        if html_path.exists():
            refresh_html(html_path, df)
        count += 1
    print(f"Updated {count} clustering result folders")


if __name__ == "__main__":
    main()
