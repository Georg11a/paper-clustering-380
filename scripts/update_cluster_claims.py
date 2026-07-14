#!/usr/bin/env python3
"""Refresh generated explorer labels with design-knowledge contribution claims."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


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


def refresh_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    if "cluster" not in df.columns:
        return df
    for col in ["design_knowledge_form", "design_knowledge_action", "design_knowledge_contribution"]:
        if col not in df.columns:
            df[col] = ""
    for cluster, group in df.groupby("cluster"):
        if str(cluster) == "-1":
            continue
        claim = claim_for_group(group)
        mask = df["cluster"].astype(str) == str(cluster)
        for key, value in claim.items():
            df.loc[mask, key] = value
    df.to_csv(path, index=False)
    return df


def write_summary(df: pd.DataFrame, path: Path) -> None:
    lines = ["# Clustering Summary", "", "## Cluster Themes", ""]
    for cluster in sorted(df["cluster"].unique(), key=lambda value: int(value) if str(value).lstrip("-").isdigit() else 999):
        group = df[df["cluster"].astype(str) == str(cluster)]
        lines.append(f"### Cluster {cluster} ({len(group)} papers)")
        first = group.iloc[0]
        lines.append(f"Label candidate: {first.get('cluster_label_candidate', '')}")
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
            "design_knowledge_contribution",
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
