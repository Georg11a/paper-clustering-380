#!/usr/bin/env python3
"""Build the static GitHub Pages explorer for Path 1 cluster interpretations."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def clean(value: object) -> str:
    return " ".join(str(value if value is not None else "").split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="docs/path1.html")
    args = parser.parse_args()

    source = Path(args.input)
    rows = list(csv.DictReader(source.open(encoding="utf-8-sig")))
    if not rows:
        raise ValueError("Path 1 cluster interpretation CSV is empty.")

    clusters = []
    for row in rows:
        clusters.append(
            {
                "keyword": clean(row["keyword"]),
                "clusterId": clean(row["cluster_id"]),
                "paperCount": int(row["paper_count"]),
                "assignmentStatus": clean(row["assignment_status"]),
                "status": (
                    "frozen"
                    if clean(row["interpretation_status"]) == "frozen"
                    else "provisional"
                ),
                "descriptor": clean(
                    row["statistical_descriptor_not_final_label"]
                ),
                "terms": [
                    clean(term)
                    for term in row["top_terms"].split("|")
                    if clean(term)
                ],
                "paperIds": [
                    clean(value)
                    for value in row["representative_paper_ids"].split("|")
                    if clean(value)
                ],
                "titles": [
                    clean(value)
                    for value in row["representative_titles"].split("|")
                    if clean(value)
                ],
                "passageIds": [
                    clean(value)
                    for value in row["representative_passage_ids"].split("|")
                    if clean(value)
                ],
                "siblingOverlap": float(
                    row["maximum_sibling_top10_jaccard"] or 0
                ),
            }
        )

    payload = json.dumps(clusters, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Path 1 · Statistical Cluster Interpretation</title>
  <style>
    :root {{
      --ink: #18313a;
      --muted: #63777d;
      --line: #dce7e4;
      --surface: #ffffff;
      --wash: #f3f7f5;
      --teal: #147b77;
      --teal-soft: #dff2ef;
      --coral: #d96f55;
      --coral-soft: #fbe9e4;
      --blue: #426c9c;
      --shadow: 0 12px 34px rgba(24, 49, 58, .08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--ink); background: var(--wash); }}
    a {{ color: var(--teal); font-weight: 700; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .hero {{
      padding: 36px clamp(20px, 5vw, 72px) 28px;
      color: #fff;
      background:
        radial-gradient(circle at 84% 14%, rgba(255,255,255,.18), transparent 28%),
        linear-gradient(125deg, #183f48 0%, #176d6a 62%, #318c83 100%);
    }}
    .eyebrow {{
      margin-bottom: 10px; color: #bce8e1; font-size: 12px;
      font-weight: 800; letter-spacing: .12em; text-transform: uppercase;
    }}
    h1 {{ max-width: 940px; margin: 0; font-size: clamp(30px, 4.2vw, 54px); line-height: 1.05; }}
    .hero p {{ max-width: 820px; margin: 17px 0 0; color: #e2f2ef; font-size: 16px; line-height: 1.6; }}
    .hero-links {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }}
    .hero-links a {{
      padding: 9px 13px; border: 1px solid rgba(255,255,255,.35);
      border-radius: 999px; color: #fff; background: rgba(255,255,255,.08);
    }}
    .layout {{ width: min(1500px, 100%); margin: 0 auto; padding: 22px clamp(16px, 4vw, 52px) 60px; }}
    .notice {{
      display: grid; grid-template-columns: auto 1fr; gap: 12px;
      align-items: start; padding: 15px 17px; border: 1px solid #f0cfc7;
      border-radius: 13px; background: var(--coral-soft); color: #713a2d;
    }}
    .notice strong {{ display: block; margin-bottom: 3px; }}
    .notice-icon {{
      display: grid; width: 28px; height: 28px; place-items: center;
      border-radius: 50%; color: #fff; background: var(--coral); font-weight: 900;
    }}
    .metrics {{
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px; margin: 18px 0;
    }}
    .metric {{
      min-height: 104px; padding: 17px; border: 1px solid var(--line);
      border-radius: 14px; background: var(--surface); box-shadow: var(--shadow);
    }}
    .metric-value {{ font-size: 29px; font-weight: 820; }}
    .metric-label {{ margin-top: 5px; color: var(--muted); font-size: 13px; line-height: 1.4; }}
    .controls {{
      position: sticky; top: 0; z-index: 5; display: grid;
      grid-template-columns: 1fr 1fr minmax(230px, 2fr); gap: 10px;
      margin: 18px 0; padding: 12px; border: 1px solid var(--line);
      border-radius: 14px; background: rgba(255,255,255,.94);
      box-shadow: 0 8px 22px rgba(24, 49, 58, .07); backdrop-filter: blur(10px);
    }}
    select, input {{
      width: 100%; min-height: 42px; padding: 0 12px; border: 1px solid #cbdad6;
      border-radius: 9px; color: var(--ink); background: #fff; font: inherit;
    }}
    .result-line {{ margin: 5px 1px 13px; color: var(--muted); font-size: 13px; }}
    .groups {{ display: grid; gap: 24px; }}
    .group-title {{
      display: flex; align-items: baseline; justify-content: space-between;
      gap: 12px; margin-bottom: 10px;
    }}
    .group-title h2 {{ margin: 0; font-size: 22px; }}
    .group-meta {{ color: var(--muted); font-size: 13px; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .card {{
      display: flex; min-width: 0; flex-direction: column; padding: 17px;
      border: 1px solid var(--line); border-radius: 15px; background: var(--surface);
      box-shadow: var(--shadow);
    }}
    .card-top {{ display: flex; justify-content: space-between; gap: 10px; align-items: flex-start; }}
    .cluster-id {{ color: var(--blue); font-size: 13px; font-weight: 850; letter-spacing: .03em; }}
    .count {{ margin-top: 3px; color: var(--muted); font-size: 12px; }}
    .badge {{
      flex: 0 0 auto; padding: 5px 8px; border-radius: 999px;
      font-size: 11px; font-weight: 850; text-transform: uppercase;
    }}
    .badge.frozen {{ color: #0d615d; background: var(--teal-soft); }}
    .badge.provisional {{ color: #8a4634; background: var(--coral-soft); }}
    .descriptor {{ margin: 16px 0 11px; font-size: 19px; font-weight: 790; line-height: 1.3; }}
    .descriptor-note {{ color: var(--muted); font-size: 11px; font-weight: 600; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 11px; }}
    .chip {{ padding: 5px 8px; border-radius: 7px; color: #315c5a; background: #edf6f4; font-size: 12px; }}
    .section {{ margin-top: 17px; }}
    .section-label {{
      margin-bottom: 7px; color: var(--muted); font-size: 10px;
      font-weight: 850; letter-spacing: .09em; text-transform: uppercase;
    }}
    .papers {{ display: grid; gap: 7px; margin: 0; padding: 0; list-style: none; }}
    .papers li {{ padding-top: 7px; border-top: 1px solid #edf2f1; font-size: 13px; line-height: 1.4; }}
    .paper-id {{ color: #8a999d; font: 10px ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .overlap {{ margin-top: auto; padding-top: 17px; }}
    .overlap-line {{ display: flex; justify-content: space-between; color: var(--muted); font-size: 11px; }}
    .bar {{ height: 5px; margin-top: 6px; overflow: hidden; border-radius: 99px; background: #edf1f0; }}
    .bar span {{ display: block; height: 100%; background: var(--coral); }}
    .empty {{ padding: 44px 20px; text-align: center; color: var(--muted); }}
    @media (max-width: 1100px) {{ .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 760px) {{
      .metrics, .cards {{ grid-template-columns: 1fr; }}
      .controls {{ position: static; grid-template-columns: 1fr; }}
      .hero {{ padding-top: 28px; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="eyebrow">Design-knowledge survey · Path 1</div>
    <h1>Statistical cluster interpretation</h1>
    <p>Cluster membership comes from keyword-conditioned BGE-M3 embeddings and Spectral clustering. Path 1 applies adapted class-based TF-IDF to explain those assignments; it does not regroup papers.</p>
    <div class="hero-links">
      <a href="index.html">← Main explorer</a>
      <a href="https://github.com/Georg11a/paper-clustering-380/tree/main/outputs/path1/statistical_topics_282_20260728">Source outputs</a>
      <a href="https://github.com/Georg11a/paper-clustering-380/blob/main/docs/human_cluster_confirmation_guide.md">Review guide</a>
    </div>
  </header>
  <main class="layout">
    <section class="notice">
      <div class="notice-icon">!</div>
      <div><strong>Interpretation status matters.</strong>Design Theory is frozen. Every other keyword group is provisional until a human reviewer confirms coherence, granularity, and boundary-paper membership. Statistical descriptors are evidence for review, not final editorial labels.</div>
    </section>
    <section class="metrics" aria-label="Run summary">
      <div class="metric"><div class="metric-value">282</div><div class="metric-label">Retained publications after publication-family review</div></div>
      <div class="metric"><div class="metric-value">27</div><div class="metric-label">Keyword-conditioned clusters and one unsplit group</div></div>
      <div class="metric"><div class="metric-value">3</div><div class="metric-label">Frozen Design Theory interpretations</div></div>
      <div class="metric"><div class="metric-value">24</div><div class="metric-label">Provisional interpretations requiring human confirmation</div></div>
    </section>
    <section class="controls" aria-label="Result filters">
      <select id="keywordFilter" aria-label="Filter by keyword"></select>
      <select id="statusFilter" aria-label="Filter by status">
        <option value="all">All statuses</option>
        <option value="frozen">Frozen</option>
        <option value="provisional">Provisional</option>
      </select>
      <input id="searchInput" type="search" placeholder="Search descriptors, terms, or representative papers" aria-label="Search results">
    </section>
    <div class="result-line" id="resultLine"></div>
    <div class="groups" id="groups"></div>
  </main>
  <script>
    const clusters = {payload};
    const keywordFilter = document.getElementById('keywordFilter');
    const statusFilter = document.getElementById('statusFilter');
    const searchInput = document.getElementById('searchInput');
    const groups = document.getElementById('groups');
    const resultLine = document.getElementById('resultLine');
    const escapeHtml = value => String(value).replace(/[&<>"']/g, character => ({{
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
    }})[character]);
    const keywords = [...new Set(clusters.map(cluster => cluster.keyword))]
      .sort((left, right) => left.localeCompare(right));
    keywordFilter.innerHTML = '<option value="all">All keyword groups</option>' +
      keywords.map(keyword => `<option value="${{escapeHtml(keyword)}}">${{escapeHtml(keyword)}}</option>`).join('');

    function card(cluster) {{
      const papers = cluster.titles.map((title, index) =>
        `<li>${{escapeHtml(title)}}<div class="paper-id">${{escapeHtml(cluster.paperIds[index] || '')}}</div></li>`
      ).join('');
      const terms = cluster.terms.slice(0, 10)
        .map(term => `<span class="chip">${{escapeHtml(term)}}</span>`).join('');
      const overlap = Math.round(cluster.siblingOverlap * 100);
      return `<article class="card">
        <div class="card-top">
          <div><div class="cluster-id">${{escapeHtml(cluster.clusterId)}}</div><div class="count">${{cluster.paperCount}} papers</div></div>
          <span class="badge ${{cluster.status}}">${{cluster.status}}</span>
        </div>
        <div class="descriptor">${{escapeHtml(cluster.descriptor || 'Descriptor pending')}}</div>
        <div class="descriptor-note">Statistical descriptor · not a final topic label</div>
        <div class="chips">${{terms}}</div>
        <div class="section"><div class="section-label">Representative papers</div><ul class="papers">${{papers}}</ul></div>
        <div class="overlap"><div class="overlap-line"><span>Maximum sibling term overlap</span><span>${{overlap}}%</span></div><div class="bar"><span style="width:${{overlap}}%"></span></div></div>
      </article>`;
    }}

    function render() {{
      const keyword = keywordFilter.value;
      const status = statusFilter.value;
      const query = searchInput.value.trim().toLocaleLowerCase();
      const filtered = clusters.filter(cluster => {{
        const matchesKeyword = keyword === 'all' || cluster.keyword === keyword;
        const matchesStatus = status === 'all' || cluster.status === status;
        const searchable = [
          cluster.keyword, cluster.clusterId, cluster.descriptor,
          ...cluster.terms, ...cluster.titles
        ].join(' ').toLocaleLowerCase();
        return matchesKeyword && matchesStatus && (!query || searchable.includes(query));
      }});
      resultLine.textContent = `${{filtered.length}} of ${{clusters.length}} cluster interpretations shown`;
      if (!filtered.length) {{
        groups.innerHTML = '<div class="empty">No cluster interpretations match these filters.</div>';
        return;
      }}
      const grouped = Object.groupBy
        ? Object.groupBy(filtered, cluster => cluster.keyword)
        : filtered.reduce((result, cluster) => {{
            (result[cluster.keyword] ||= []).push(cluster);
            return result;
          }}, {{}});
      groups.innerHTML = Object.entries(grouped).map(([keywordName, items]) =>
        `<section><div class="group-title"><h2>${{escapeHtml(keywordName)}}</h2><div class="group-meta">${{items.reduce((sum, item) => sum + item.paperCount, 0)}} papers · ${{items.length}} interpretations</div></div><div class="cards">${{items.map(card).join('')}}</div></section>`
      ).join('');
    }}
    [keywordFilter, statusFilter].forEach(control => control.addEventListener('change', render));
    searchInput.addEventListener('input', render);
    render();
  </script>
</body>
</html>
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(f"Wrote {len(clusters)} Path 1 cluster cards to {output}")


if __name__ == "__main__":
    main()
