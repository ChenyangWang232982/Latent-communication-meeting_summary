import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from speech_embedding.paths import DATA_DIR, PROJECT_ROOT, SPLIT_DIR, project_path


METADATA_PATH = DATA_DIR / "metadata.jsonl"
WORDS_DIR = PROJECT_ROOT / "ami" / "words"
UPDATE_SPLITS = True
MAKE_BACKUP = True

PUNCTUATION_WITHOUT_LEADING_SPACE = {".", ",", "?", "!", ":", ";", "%"}


def read_jsonl(path):
    rows = []
    with project_path(path).open("r", encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        for row in rows:
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def backup_file(path):
    path = project_path(path)
    if not MAKE_BACKUP or not path.exists():
        return
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)


def meeting_id_from_words_path(path):
    # Example: ES2002a.A.words.xml -> ES2002a
    return path.name.split(".")[0]


def parse_float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def clean_token(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def append_token(tokens, token, is_punctuation):
    if not token:
        return

    if not tokens:
        tokens.append(token)
        return

    if is_punctuation or token in PUNCTUATION_WITHOUT_LEADING_SPACE:
        tokens[-1] = tokens[-1] + token
    else:
        tokens.append(token)


def extract_words_from_file(path):
    tree = ET.parse(path)
    root = tree.getroot()
    words = []

    for elem in root.iter():
        if not elem.tag.endswith("w"):
            continue
        token = clean_token(elem.text or "")
        if not token:
            continue

        start = parse_float(elem.attrib.get("starttime"))
        end = parse_float(elem.attrib.get("endtime"), start)
        is_punctuation = elem.attrib.get("punc") == "true"
        words.append(
            {
                "start": start,
                "end": end,
                "token": token,
                "is_punctuation": is_punctuation,
            }
        )

    return words


def build_transcripts(words_dir):
    words_dir = project_path(words_dir)
    transcripts = {}

    for words_path in sorted(words_dir.glob("*.words.xml")):
        meeting_id = meeting_id_from_words_path(words_path)
        transcripts.setdefault(meeting_id, []).extend(extract_words_from_file(words_path))

    merged = {}
    for meeting_id, words in transcripts.items():
        words.sort(key=lambda item: (item["start"], item["end"], item["token"]))
        tokens = []
        for word in words:
            append_token(tokens, word["token"], word["is_punctuation"])
        merged[meeting_id] = " ".join(tokens)

    return merged


def update_rows_with_transcripts(rows, transcripts):
    updated = []
    missing = []

    for row in rows:
        meeting_id = row["meeting_id"]
        transcript = transcripts.get(meeting_id, "")
        if not transcript:
            missing.append(meeting_id)
        row = dict(row)
        row["transcript"] = transcript
        updated.append(row)

    return updated, missing


def update_metadata_file(path, transcripts):
    rows = read_jsonl(path)
    updated_rows, missing = update_rows_with_transcripts(rows, transcripts)
    backup_file(path)
    write_jsonl(path, updated_rows)
    return len(updated_rows), missing


def update_split_files(transcripts):
    if not UPDATE_SPLITS:
        return

    for split_name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        split_path = SPLIT_DIR / split_name
        if not split_path.exists():
            continue
        count, missing = update_metadata_file(split_path, transcripts)
        print(
            f"updated {split_path}: rows={count}, "
            f"missing_transcripts={len(missing)}"
        )


def main():
    transcripts = build_transcripts(WORDS_DIR)
    print(f"loaded transcripts: {len(transcripts)}")

    count, missing = update_metadata_file(METADATA_PATH, transcripts)
    print(f"updated {METADATA_PATH}: rows={count}, missing_transcripts={len(missing)}")

    if missing:
        print("first missing meeting ids:", ", ".join(missing[:20]))

    update_split_files(transcripts)


if __name__ == "__main__":
    main()
