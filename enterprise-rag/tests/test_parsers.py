import pytest

from app.ingestion.chunking import chunk_document
from app.ingestion.parsers import parse_docx, parse_xlsx, rows_to_lines


def test_rows_to_lines_repeats_headers():
    lines = rows_to_lines([["Item", "Limit"], ["Travel", "500"], ["Meals", ""]])
    assert lines == ["Item: Travel; Limit: 500", "Item: Meals"]


def test_xlsx_row_groups_and_locations(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Budget"
    ws.append(["Dept", "Amount"])
    for i in range(65):
        ws.append([f"dept{i}", i * 10])
    path = tmp_path / "b.xlsx"
    wb.save(path)

    pages = parse_xlsx(str(path), rows_per_group=30)
    assert [p.location for p in pages] == ["Budget!rows 2-31", "Budget!rows 32-61", "Budget!rows 62-66"]
    assert "Dept: dept0; Amount: 0" in pages[0].text
    parents = chunk_document(pages)
    assert parents[0].section == "Budget"


def test_docx_headings_and_tables(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Travel policy", level=1)
    d.add_heading("Limits", level=2)
    d.add_paragraph("Flights must be economy class.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Item", "Limit"
    t.cell(1, 0).text, t.cell(1, 1).text = "Hotel", "200"
    path = tmp_path / "p.docx"
    d.save(path)

    parents = chunk_document(parse_docx(str(path)))
    assert parents[0].section == "Travel policy > Limits"
    assert "Item: Hotel; Limit: 200" in parents[0].text
