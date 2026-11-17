"""Parsers for PDF, Word, Excel and web pages. Each returns light markdown so one chunker handles all."""
import ipaddress
import re
import socket
from urllib.parse import urljoin, urlparse

from .types import ParsedDocument, RawPage

MAX_WEB_BYTES = 10 * 1024 * 1024


def rows_to_lines(rows: list[list[str]]) -> list[str]:
    """Render table rows as 'Header: value; Header: value' so every row is self-describing."""
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return []
    if len(rows) == 1:
        return ["; ".join(c for c in rows[0] if c)]
    header = [h.strip() or f"col{i + 1}" for i, h in enumerate(rows[0])]
    lines = []
    for row in rows[1:]:
        line = "; ".join(f"{h}: {v.strip()}" for h, v in zip(header, row) if v.strip())
        if line:
            lines.append(line)
    return lines


def parse_pdf(path: str) -> list[RawPage]:
    import pymupdf4llm  # imported lazily: heavy

    chunks = pymupdf4llm.to_markdown(path, page_chunks=True)
    pages = [RawPage(text=c["text"], location=f"p. {i}") for i, c in enumerate(chunks, start=1) if c["text"].strip()]
    if not pages:
        raise ValueError("No extractable text in PDF (scanned? run OCR first)")
    return pages


def parse_docx(path: str) -> list[RawPage]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(path)
    blocks: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            m = re.match(r"Heading (\d)", style)
            if m:
                blocks.append("#" * min(int(m.group(1)), 6) + " " + text)
            elif style == "Title":
                blocks.append("# " + text)
            else:
                blocks.append(text)
        elif tag == "tbl":
            table = Table(child, doc)
            rows = [[c.text.strip().replace("\n", " ") for c in r.cells] for r in table.rows]
            blocks.extend(rows_to_lines(rows))
    return [RawPage(text="\n\n".join(blocks))]


def parse_xlsx(path: str, rows_per_group: int = 30) -> list[RawPage]:
    """Each sheet is split into row groups. Header row is repeated into every row's text."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)  # data_only: cached formula values
    pages: list[RawPage] = []
    for ws in wb.worksheets:
        numbered = [
            (i, ["" if c is None else str(c).strip() for c in row])
            for i, row in enumerate(ws.iter_rows(values_only=True), start=1)
        ]
        numbered = [(i, r) for i, r in numbered if any(r)]
        if len(numbered) < 2:
            continue
        header = [h or f"col{j + 1}" for j, h in enumerate(numbered[0][1])]
        body = numbered[1:]
        for start in range(0, len(body), rows_per_group):
            group = body[start : start + rows_per_group]
            lines = ["; ".join(f"{h}: {v}" for h, v in zip(header, r) if v) for _, r in group]
            lines = [ln for ln in lines if ln]
            if lines:
                text = f"# {ws.title}\n\n" + "\n\n".join(lines)
                pages.append(RawPage(text=text, location=f"{ws.title}!rows {group[0][0]}-{group[-1][0]}"))
    if not pages:
        raise ValueError("Workbook has no data rows")
    return pages


def parse_file(path: str, source_type: str, fallback_title: str) -> ParsedDocument:
    parser = {"pdf": parse_pdf, "docx": parse_docx, "xlsx": parse_xlsx}[source_type]
    pages = parser(path)
    return ParsedDocument(title=fallback_title, pages=pages)


# ---- web ------------------------------------------------------------------------------------


def assert_public_url(url: str) -> None:
    """SSRF guard: only http(s) to publicly routable addresses.

    Note: DNS can change between this check and the fetch; for stricter setups, fetch through an
    egress proxy that enforces the same rule.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only http(s) URLs are allowed")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port)
    except socket.gaierror as e:
        raise ValueError(f"Cannot resolve host: {parsed.hostname}") from e
    for info in infos:
        if not ipaddress.ip_address(info[4][0].split("%")[0]).is_global:
            raise ValueError("URL resolves to a non-public address")


def fetch_web(url: str) -> ParsedDocument:
    import httpx
    import trafilatura

    html = None
    current = url
    with httpx.Client(timeout=20, follow_redirects=False, headers={"User-Agent": "enterprise-rag/1.0"}) as client:
        for _ in range(5):  # follow redirects manually so every hop is checked
            assert_public_url(current)
            resp = client.get(current)
            if resp.is_redirect:
                current = urljoin(current, resp.headers["location"])
                continue
            resp.raise_for_status()
            if len(resp.content) > MAX_WEB_BYTES:
                raise ValueError("Page too large")
            html = resp.text
            break
    if html is None:
        raise ValueError("Too many redirects")
    text = trafilatura.extract(html, output_format="markdown", include_tables=True, include_comments=False)
    if not text:
        raise ValueError("No extractable content (JS-rendered page? fetch it with Playwright instead)")
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else None) or url
    return ParsedDocument(title=title, pages=[RawPage(text=text)])
