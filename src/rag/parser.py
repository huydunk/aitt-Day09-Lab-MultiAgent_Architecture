from __future__ import annotations


def parse_policy_markdown(markdown_text: str) -> list[dict]:
    """Parse policy markdown into chunks, one chunk per H3 section.

    For H2 sections that have content but no H3 children, one chunk is
    produced with section_h3 = "".
    """
    lines = markdown_text.split("\n")
    chunks: list[dict] = []

    current_h2 = ""
    current_h3 = ""
    current_content_lines: list[str] = []

    def flush() -> None:
        content = "\n".join(current_content_lines).strip()
        if not content:
            return
        if current_h3:
            citation = f"{current_h2} > {current_h3}"
            rendered_text = f"## {current_h2}\n### {current_h3}\n{content}"
        else:
            citation = current_h2
            rendered_text = f"## {current_h2}\n{content}"
        chunks.append(
            {
                "section_h2": current_h2,
                "section_h3": current_h3,
                "citation": citation,
                "rendered_text": rendered_text,
            }
        )

    for line in lines:
        if line.startswith("## "):
            flush()
            current_h2 = line[3:].strip()
            current_h3 = ""
            current_content_lines = []
        elif line.startswith("### "):
            flush()
            current_h3 = line[4:].strip()
            current_content_lines = []
        else:
            current_content_lines.append(line)

    flush()  # last pending chunk
    return chunks
