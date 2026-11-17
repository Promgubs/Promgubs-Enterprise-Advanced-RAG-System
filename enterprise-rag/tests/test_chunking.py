from app.ingestion.chunking import chunk_document, pack, split_long
from app.ingestion.types import RawPage

DOC = """# Leave policy

Intro paragraph about leave.

## Annual leave

Full-time employees receive 25 days of annual leave. Unused days carry over up to 5 days.

## Sick leave

Sick leave requires a doctor's note after 3 days.
"""


def test_heading_paths_and_location():
    parents = chunk_document([RawPage(DOC, location="p. 2")])
    sections = [p.section for p in parents]
    assert sections == ["Leave policy", "Leave policy > Annual leave", "Leave policy > Sick leave"]
    assert all(p.location == "p. 2" for p in parents)
    assert "25 days" in parents[1].text


def test_heading_stack_carries_across_pages():
    pages = [RawPage("# Handbook\n\n## Travel\n\nPage one text."), RawPage("More travel text on page two.", "p. 2")]
    parents = chunk_document(pages)
    assert parents[1].section == "Handbook > Travel"


def test_children_respect_size_limit_and_lose_no_text():
    words = " ".join(f"word{i}" for i in range(2000))
    parents = chunk_document([RawPage(f"# T\n\n{words}")], child_chars=500, overlap=60, parent_max=2000)
    assert len(parents) > 1
    for p in parents:
        assert len(p.text) <= 2000 + 500  # parent soft limit
        for child in p.children:
            assert len(child) <= 500 + 60 + 2
    seen = " ".join(p.text for p in parents)
    assert all(f"word{i}" in seen for i in range(2000))


def test_code_fence_hash_is_not_a_heading():
    text = "# Real heading\n\n```\n# just a comment\n```\n\nafter"
    parents = chunk_document([RawPage(text)])
    assert [p.section for p in parents] == ["Real heading"]
    assert "# just a comment" in parents[0].text


def test_long_sentence_is_hard_split():
    assert all(len(part) <= 100 for part in split_long("x" * 450, 100))


def test_pack_overlap_repeats_tail():
    chunks = pack(["alpha " * 30, "beta " * 30, "gamma " * 30], max_chars=200, overlap=40)
    assert len(chunks) >= 2
    assert chunks[1].startswith("alpha") or chunks[1].startswith("beta")
