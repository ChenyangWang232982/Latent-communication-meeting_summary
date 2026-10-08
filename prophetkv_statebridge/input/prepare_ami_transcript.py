"""Extract one chronological speaker-labelled AMI transcript from official annotations."""

from __future__ import annotations

import argparse
import re
import urllib.request
import xml.etree.ElementTree as element_tree
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AMI_MANUAL_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ID_PATTERN = re.compile(r"id\(([^)]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting-id", default="ES2002a")
    parser.add_argument("--archive", type=Path, default=PROJECT_ROOT / "data" / "ami_raw" / "ami_public_manual_1.6.2.zip")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--download", action="store_true", help="Download the official 22 MB annotation archive if needed")
    return parser.parse_args()


def local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def normalize(tokens: list[str]) -> str:
    text = " ".join(token for token in tokens if token).strip()
    return re.sub(r"\s+([.,!?;:])", r"\1", text)


def read_words(archive: zipfile.ZipFile, meeting_id: str, speaker: str):
    root = element_tree.fromstring(archive.read(f"words/{meeting_id}.{speaker}.words.xml"))
    words: list[tuple[str, float, str]] = []
    for element in root.iter():
        if local_name(element.tag) != "w" or not element.text:
            continue
        word_id = next((value for key, value in element.attrib.items() if local_name(key) == "id"), None)
        if word_id is None:
            continue
        words.append((word_id, float(element.attrib.get("starttime", "0")), element.text.strip()))
    return words


def extract_dialogue_acts(archive: zipfile.ZipFile, meeting_id: str, speaker: str):
    words = read_words(archive, meeting_id, speaker)
    positions = {word_id: index for index, (word_id, _, _) in enumerate(words)}
    root = element_tree.fromstring(archive.read(f"dialogueActs/{meeting_id}.{speaker}.dialog-act.xml"))
    acts: list[tuple[float, str, str]] = []
    for element in root.iter():
        if local_name(element.tag) != "dact":
            continue
        hrefs = [child.attrib.get("href", "") for child in element if local_name(child.tag) == "child"]
        ids = [word_id for href in hrefs for word_id in ID_PATTERN.findall(href)]
        if not ids or ids[0] not in positions or ids[-1] not in positions:
            continue
        start, end = sorted((positions[ids[0]], positions[ids[-1]]))
        segment = words[start : end + 1]
        text = normalize([word for _, _, word in segment])
        if text:
            acts.append((segment[0][1], speaker, text))
    return acts


def ensure_archive(path: Path, download: bool) -> None:
    if path.exists():
        return
    if not download:
        raise FileNotFoundError(f"Missing AMI archive: {path}. Re-run with --download.")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official AMI manual annotations to {path}")
    urllib.request.urlretrieve(AMI_MANUAL_URL, path)


def main() -> None:
    args = parse_args()
    ensure_archive(args.archive, args.download)
    output = args.output or (Path(__file__).resolve().parent / f"ami_{args.meeting_id}.txt")
    with zipfile.ZipFile(args.archive) as archive:
        speakers = sorted({name.split(".")[1] for name in archive.namelist() if name.startswith(f"dialogueActs/{args.meeting_id}.") and name.endswith(".dialog-act.xml")})
        if not speakers:
            raise ValueError(f"No dialogue-act annotations found for {args.meeting_id}")
        acts = [act for speaker in speakers for act in extract_dialogue_acts(archive, args.meeting_id, speaker)]
    acts.sort(key=lambda item: item[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"AMI Meeting Corpus transcript: {args.meeting_id}",
        "Source: AMI manual annotations v1.6.2 (CC BY 4.0).",
        "",
    ]
    lines.extend(f"[{start:07.2f}] Speaker {speaker}: {text}" for start, speaker, text in acts)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(acts)} dialogue acts to {output}")


if __name__ == "__main__":
    main()
