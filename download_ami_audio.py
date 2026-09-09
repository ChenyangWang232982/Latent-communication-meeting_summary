import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATION_DIR = PROJECT_ROOT / "ami" / "abstractive"
OUTPUT_AUDIO_DIR = PROJECT_ROOT / "data" / "toy_meetings" / "audio"
OUTPUT_METADATA = PROJECT_ROOT / "data" / "toy_meetings" / "metadata.jsonl"
BASE_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
GB = 1024 ** 3


EXISTING_AUDIO_ALIASES = {
    "ES2002a": PROJECT_ROOT / "data" / "toy_meetings" / "audio" / "sample_001.wav",
    "ES2002b": PROJECT_ROOT / "data" / "toy_meetings" / "audio" / "sample_002.wav",
}


def project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_posix(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def extract_summary(summary_path):
    tree = ET.parse(summary_path)
    root = tree.getroot()
    sentences = []

    for elem in root.iter():
        if elem.tag.endswith("sentence") and elem.text:
            text = " ".join(elem.text.split())
            if text:
                sentences.append(text)

    return " ".join(sentences)


def get_audio_path(meeting_id):
    standard_path = OUTPUT_AUDIO_DIR / f"{meeting_id}.wav"
    if standard_path.exists():
        return standard_path

    alias_path = EXISTING_AUDIO_ALIASES.get(meeting_id)
    if alias_path is not None and alias_path.exists():
        return alias_path

    return standard_path


def audio_url(meeting_id):
    return f"{BASE_URL}/{meeting_id}/audio/{meeting_id}.Mix-Headset.wav"


def get_remote_size(url):
    request = Request(url, method="HEAD")
    try:
        with urlopen(request, timeout=30) as response:
            size = response.headers.get("Content-Length")
            return int(size) if size is not None else None
    except (HTTPError, URLError, TimeoutError):
        return None


def get_free_space_bytes(path):
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free


def download_audio(meeting_id, output_path):
    if output_path.exists():
        print(f"skip existing audio: {output_path}", flush=True)
        return True

    url = audio_url(meeting_id)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    print(f"download {meeting_id}: {url}", flush=True)

    try:
        with urlopen(url) as response:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with tmp_path.open("wb") as out_file:
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (25 * 1024 * 1024) < len(chunk):
                        mb = downloaded / (1024 * 1024)
                        print(f"  {meeting_id}: {mb:.1f} MB", flush=True)
    except HTTPError as exc:
        print(f"failed {meeting_id}: HTTP {exc.code}", flush=True)
        if tmp_path.exists():
            tmp_path.unlink()
        return False
    except URLError as exc:
        print(f"failed {meeting_id}: {exc}", flush=True)
        if tmp_path.exists():
            tmp_path.unlink()
        return False

    try:
        if output_path.exists():
            if output_path.stat().st_size > 0:
                print(
                    f"target appeared while downloading, keep existing: {output_path}",
                    flush=True,
                )
                tmp_path.unlink(missing_ok=True)
                return True
            output_path.unlink()

        tmp_path.replace(output_path)
    except PermissionError:
        print(
            f"failed {meeting_id}: cannot finalize {tmp_path} -> {output_path}. "
            "Close any program using the wav file, or delete the target file and retry.",
            flush=True,
        )
        return False

    print(f"saved: {output_path}", flush=True)
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download AMI Mix-Headset audio and build metadata.jsonl."
    )
    parser.add_argument(
        "--annotation-dir",
        default=str(ANNOTATION_DIR),
        help="Directory containing AMI *.abssumm.xml files.",
    )
    parser.add_argument(
        "--audio-dir",
        default=str(OUTPUT_AUDIO_DIR),
        help="Directory where downloaded wav files will be stored.",
    )
    parser.add_argument(
        "--metadata",
        default=str(OUTPUT_METADATA),
        help="Output JSONL metadata path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of meetings to process. Omit for all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only estimate download size and metadata rows. Do not download.",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=20.0,
        help="Maximum new audio to download in GB. Default: 20.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help="Stop before downloading if remaining disk space would fall below this GB.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    annotation_dir = project_path(args.annotation_dir)
    output_audio_dir = project_path(args.audio_dir)
    output_metadata = project_path(args.metadata)

    global OUTPUT_AUDIO_DIR
    OUTPUT_AUDIO_DIR = output_audio_dir

    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)

    summary_files = sorted(annotation_dir.glob("*.abssumm.xml"))
    if args.limit is not None:
        summary_files = summary_files[: args.limit]
    if not summary_files:
        raise FileNotFoundError(f"No summary files found in {annotation_dir}")

    print(f"summary files: {len(summary_files)}", flush=True)

    rows = []
    planned_new_bytes = 0
    max_new_bytes = int(args.max_gb * GB)
    min_free_bytes = int(args.min_free_gb * GB)

    for summary_path in summary_files:
        meeting_id = summary_path.name.replace(".abssumm.xml", "")
        summary = extract_summary(summary_path)
        audio_path = get_audio_path(meeting_id)

        if not audio_path.exists():
            url = audio_url(meeting_id)
            remote_size = get_remote_size(url)
            size_text = "unknown size"
            if remote_size is not None:
                size_text = f"{remote_size / (1024 ** 2):.1f} MB"
                if planned_new_bytes + remote_size > max_new_bytes:
                    print(
                        f"skip {meeting_id}: would exceed --max-gb={args.max_gb}",
                        flush=True,
                    )
                    continue

                free_after = get_free_space_bytes(output_audio_dir) - remote_size
                if free_after < min_free_bytes:
                    print(
                        f"skip {meeting_id}: disk would fall below "
                        f"--min-free-gb={args.min_free_gb}",
                        flush=True,
                    )
                    continue

                planned_new_bytes += remote_size

            print(f"plan {meeting_id}: {size_text}", flush=True)

            if args.dry_run:
                rows.append(
                    {
                        "meeting_id": meeting_id,
                        "audio_path": project_posix(audio_path),
                        "transcript": "",
                        "summary": summary,
                    }
                )
                continue

            if not download_audio(meeting_id, audio_path):
                continue
        else:
            print(f"plan {meeting_id}: already exists", flush=True)

        rows.append(
            {
                "meeting_id": meeting_id,
                "audio_path": project_posix(audio_path),
                "transcript": "",
                "summary": summary,
            }
        )

    if args.dry_run:
        print(
            f"dry run: planned metadata rows={len(rows)}, "
            f"known new download size={planned_new_bytes / GB:.2f} GB",
            flush=True,
        )
        return

    with output_metadata.open("w", encoding="utf-8") as out_file:
        for row in rows:
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {output_metadata}", flush=True)


if __name__ == "__main__":
    main()
