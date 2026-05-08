import re
import json
from pathlib import Path

INPUT_DIR = Path("cleaned_text")
OUTPUT_DIR = Path("chunks")

OUTPUT_DIR.mkdir(exist_ok=True)


def extract_keywords(text):
    words = re.findall(r"\b[A-Za-z]{4,}\b", text.lower())
    stopwords = {
        "this", "that", "with", "from", "have",
        "will", "must", "into", "their", "they",
        "been", "were", "shall"
    }

    words = [w for w in words if w not in stopwords]
    return list(dict.fromkeys(words))[:8]


def split_sections(text):
    pattern = r"\n(?=[A-Z][A-Za-z\s&()/-]{3,}:?\n)"
    sections = re.split(pattern, text)
    return [s.strip() for s in sections if s.strip()]


def get_title(section):
    lines = section.split("\n")
    return lines[0].replace(":", "").strip()


def chunk_file(filepath, category):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    sections = split_sections(text)

    chunks = []

    for i, section in enumerate(sections, 1):
        lines = section.split("\n")

        title = get_title(section)
        body = "\n".join(lines[1:]).strip()

        chunk = {
            "id": f"{filepath.stem}_{i:03}",
            "source": f"{filepath.stem}.txt",
            "category": category,
            "section": title,
            "keywords": extract_keywords(body),
            "text": body
        }

        chunks.append(chunk)

    return chunks


CATEGORY_MAP = {
    "Academic Rules and Regulations.txt": "Rules",
    "Admission Information.txt": "Admissions",
    "Admission Schedule for 2025.txt": "Schedules",
    "Fee Information.txt": "Fees",
    "Financial Assistance and Scholarships.txt": "Scholarships",
    "Programs Offered and Campuses.txt": "Programs",
    "Transport and Hostel Facilities.txt": "Facilities"
}

all_chunks = []

for file in INPUT_DIR.glob("*.txt"):
    category = CATEGORY_MAP.get(file.name, "General")

    chunks = chunk_file(file, category)

    with open(OUTPUT_DIR / f"{file.stem}.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    all_chunks.extend(chunks)

with open(OUTPUT_DIR / "all_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print(f"Done. Created {len(all_chunks)} chunks.")