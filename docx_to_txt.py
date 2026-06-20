

import re
from pathlib import Path
from docx import Document

INPUT_DIR = Path("cleaned_docx")
OUTPUT_DIR = Path("cleaned_text")

OUTPUT_DIR.mkdir(exist_ok=True)


def extract_text_from_docx(filepath: Path) -> str:
    """
    Extracts clean plain text from a .docx file.
    Preserves headings and list structure as readable text.
    """
    doc = Document(filepath)
    lines = []

    for para in doc.paragraphs:
        text = para.text.strip()

        if not text:
            # Keep one blank line for spacing between sections
            if lines and lines[-1] != "":
                lines.append("")
            continue

        style = para.style.name.lower()

        # Add a blank line before headings for better section separation
        if "heading" in style:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(text)
            lines.append("")  # blank line after heading too

        # List items get a dash prefix if they don't already have one
        elif "list" in style:
            if not text.startswith(("-", "•", "*")):
                lines.append(f"- {text}")
            else:
                lines.append(text)

        else:
            lines.append(text)

    # Clean up: collapse 3+ consecutive blank lines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip()


def convert_all():
    docx_files = list(INPUT_DIR.glob("*.docx"))

    if not docx_files:
        print(f"No .docx files found in '{INPUT_DIR}/'")
        print("Make sure your cleaned Word documents are placed in that folder.")
        return

    print(f"Found {len(docx_files)} file(s). Converting...\n")

    for filepath in sorted(docx_files):
        try:
            text = extract_text_from_docx(filepath)
            out_path = OUTPUT_DIR / (filepath.stem + ".txt")
            out_path.write_text(text, encoding="utf-8")
            word_count = len(text.split())
            print(f"  ✓  {filepath.name}  →  {out_path.name}  ({word_count} words)")
        except Exception as e:
            print(f"  ✗  {filepath.name}  →  ERROR: {e}")

    print(f"\nDone. Text files saved to '{OUTPUT_DIR}/'")
    print("You can now run chunker.py.")


if __name__ == "__main__":
    convert_all()