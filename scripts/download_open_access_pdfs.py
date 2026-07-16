#!/usr/bin/env python3
"""Download openly available PDFs for papers listed in a checklist CSV.

The script is intentionally conservative: it only saves a file after confirming
that the response is a PDF, and it does not try to bypass publisher logins.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urlparse

import requests


PDF_MAGIC = b"%PDF"


@dataclass
class Candidate:
    url: str
    source: str


def clean_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "paper.pdf"


def arxiv_pdf_from_doi(doi: str) -> str | None:
    m = re.search(r"10\.48550/arXiv\.([^/]+)$", doi, flags=re.I)
    if not m:
        return None
    arxiv_id = m.group(1)
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def openalex_candidates(session: requests.Session, doi: str) -> list[Candidate]:
    if not doi:
        return []
    url = "https://api.openalex.org/works/" + quote(f"https://doi.org/{doi}", safe="")
    try:
        response = session.get(url, timeout=20)
    except requests.RequestException:
        return []
    if response.status_code != 200:
        return []
    try:
        data = response.json()
    except json.JSONDecodeError:
        return []

    urls: list[Candidate] = []
    seen: set[str] = set()

    def add(value: str | None, source: str) -> None:
        if value and value.startswith("http") and value not in seen:
            seen.add(value)
            urls.append(Candidate(value, source))

    primary = data.get("primary_location") or {}
    add(primary.get("pdf_url"), "OpenAlex primary pdf_url")

    open_access = data.get("open_access") or {}
    add(open_access.get("oa_url"), "OpenAlex oa_url")

    for loc in data.get("locations") or []:
        add((loc or {}).get("pdf_url"), "OpenAlex location pdf_url")
        add((loc or {}).get("landing_page_url"), "OpenAlex location landing_page")

    return urls


def semantic_scholar_candidates(session: requests.Session, doi: str, url: str) -> list[Candidate]:
    paper_id = None
    if doi:
        paper_id = "DOI:" + doi
    elif "semanticscholar.org/paper/" in url:
        paper_id = url.rstrip("/").split("/")[-1]
    if not paper_id:
        return []
    api_url = "https://api.semanticscholar.org/graph/v1/paper/" + quote(paper_id, safe=":")
    try:
        response = session.get(api_url, params={"fields": "openAccessPdf,url"}, timeout=20)
    except requests.RequestException:
        return []
    if response.status_code != 200:
        return []
    try:
        data = response.json()
    except json.JSONDecodeError:
        return []
    pdf = data.get("openAccessPdf") or {}
    pdf_url = pdf.get("url")
    return [Candidate(pdf_url, "Semantic Scholar openAccessPdf")] if pdf_url else []


def direct_candidates(doi: str, url: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    arxiv = arxiv_pdf_from_doi(doi)
    if arxiv:
        candidates.append(Candidate(arxiv, "arXiv DOI"))
    if url and url.lower().split("?")[0].endswith(".pdf"):
        candidates.append(Candidate(url, "direct PDF URL"))
    return candidates


def candidate_urls(session: requests.Session, row: dict[str, str], source: str) -> list[Candidate]:
    doi = (row.get("doi") or "").strip()
    url = (row.get("download_url") or "").strip()
    candidates: list[Candidate] = []
    if source in {"all", "direct"}:
        candidates.extend(direct_candidates(doi, url))
    if source in {"all", "openalex"}:
        candidates.extend(openalex_candidates(session, doi))
    if source in {"all", "semantic-scholar"}:
        candidates.extend(semantic_scholar_candidates(session, doi, url))

    deduped: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.url and candidate.url not in seen:
            seen.add(candidate.url)
            deduped.append(candidate)
    return deduped


def response_is_pdf(response: requests.Response, first_bytes: bytes) -> bool:
    ctype = response.headers.get("content-type", "").lower()
    return "application/pdf" in ctype or first_bytes.startswith(PDF_MAGIC)


def download_pdf(session: requests.Session, url: str, output_path: Path) -> tuple[bool, str]:
    try:
        with session.get(url, stream=True, timeout=40, allow_redirects=True) as response:
            if response.status_code >= 400:
                return False, f"HTTP {response.status_code}"
            iterator = response.iter_content(chunk_size=8192)
            first = next(iterator, b"")
            if not response_is_pdf(response, first):
                ctype = response.headers.get("content-type", "unknown")
                return False, f"not PDF ({ctype})"
            tmp_path = output_path.with_suffix(output_path.suffix + ".part")
            with tmp_path.open("wb") as out:
                out.write(first)
                for chunk in iterator:
                    if chunk:
                        out.write(chunk)
            tmp_path.replace(output_path)
            return True, "downloaded"
    except requests.RequestException as exc:
        return False, exc.__class__.__name__
    except StopIteration:
        return False, "empty response"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_report(path: Path, rows: Iterable[dict[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checklist", required=True, type=Path)
    parser.add_argument("--pdf-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows to attempt; 0 means all.")
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument(
        "--source",
        choices=["all", "semantic-scholar", "openalex", "direct"],
        default="all",
        help="Restrict candidate discovery to one source.",
    )
    args = parser.parse_args()

    rows = read_rows(args.checklist)
    if args.limit:
        rows = rows[: args.limit]
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "paper-clustering-380 PDF downloader (mailto:research@example.com)",
            "Accept": "application/pdf,text/html,application/json;q=0.9,*/*;q=0.8",
        }
    )

    report_rows: list[dict[str, str]] = []
    downloaded = skipped_existing = failed = no_candidate = 0

    for index, row in enumerate(rows, start=1):
        filename = clean_filename(row.get("suggested_filename") or f"{row.get('paper_id')}.pdf")
        output_path = args.pdf_dir / filename
        result = {
            "paper_id": row.get("paper_id", ""),
            "title": row.get("title", ""),
            "doi": row.get("doi", ""),
            "suggested_filename": filename,
            "status": "",
            "source": "",
            "url": "",
            "message": "",
        }

        if output_path.exists() and output_path.stat().st_size > 0:
            skipped_existing += 1
            result.update(status="existing", message=str(output_path))
            report_rows.append(result)
            continue

        candidates = candidate_urls(session, row, args.source)
        if not candidates:
            no_candidate += 1
            result.update(status="no_open_pdf_candidate")
            report_rows.append(result)
            print(f"[{index}/{len(rows)}] no candidate: {row.get('title', '')[:70]}")
            time.sleep(args.sleep)
            continue

        success = False
        messages: list[str] = []
        for candidate in candidates:
            ok, message = download_pdf(session, candidate.url, output_path)
            messages.append(f"{candidate.source}: {message}")
            if ok:
                downloaded += 1
                result.update(
                    status="downloaded",
                    source=candidate.source,
                    url=candidate.url,
                    message=str(output_path),
                )
                print(f"[{index}/{len(rows)}] downloaded: {filename}")
                success = True
                break
            time.sleep(args.sleep)

        if not success:
            failed += 1
            result.update(
                status="failed",
                source="; ".join(c.source for c in candidates),
                url="; ".join(c.url for c in candidates),
                message=" | ".join(messages),
            )
            print(f"[{index}/{len(rows)}] failed: {row.get('title', '')[:70]}")
        report_rows.append(result)
        if index % 10 == 0:
            write_report(args.report, report_rows)
        time.sleep(args.sleep)

    write_report(args.report, report_rows)
    print(f"downloaded={downloaded}")
    print(f"existing={skipped_existing}")
    print(f"failed={failed}")
    print(f"no_open_pdf_candidate={no_candidate}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
