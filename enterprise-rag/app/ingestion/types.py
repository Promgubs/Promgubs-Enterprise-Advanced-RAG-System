from dataclasses import dataclass, field


@dataclass
class RawPage:
    """One page, or one slice of a sheet, as light markdown (headings written as '#')."""

    text: str
    location: str | None = None  # "p. 3", "Budget!rows 2-31"; None for docx and web


@dataclass
class ParsedDocument:
    title: str
    pages: list[RawPage] = field(default_factory=list)
