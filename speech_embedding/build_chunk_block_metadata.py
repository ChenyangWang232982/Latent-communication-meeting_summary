import json
import math
import wave
from pathlib import Path

from speech_embedding.paths import DATA_DIR, PROJECT_ROOT, project_path, project_posix


SOURCE_METADATA_PATH = DATA_DIR / "metadata.jsonl"
OUTPUT_DIR = DATA_DIR / "chunk_blocks"

CHUNK_SECONDS = 30.0
TRAIN_MEETINGS = 20
VAL_MEETINGS = 10
TEST_MEETINGS = 10
MIN_WORDS_PER_CHUNK = 3


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


def read_json(path):
    with project_path(path).open("r", encoding="utf-8") as in_file:
        return json.load(in_file)


def audio_duration_seconds(audio_path):
    audio_path = project_path(audio_path)
    if audio_path.suffix.lower() == ".wav":
        with wave.open(str(audio_path), "rb") as wav_file:
            return wav_file.getnframes() / float(wav_file.getframerate())

    import soundfile as sf

    info = sf.info(str(audio_path))
    return info.frames / float(info.samplerate)


def append_token(tokens, token, is_punctuation):
    if not token:
        return
    if not tokens:
        tokens.append(token)
        return
    if is_punctuation or token in {".", ",", "?", "!", ":", ";", "%"}:
        tokens[-1] = tokens[-1] + token
    else:
        tokens.append(token)


def transcript_for_window(words, start_time, end_time):
    tokens = []
    for word in words:
        word_start = float(word.get("start", 0.0))
        word_end = float(word.get("end", word_start))
        midpoint = (word_start + word_end) / 2.0
        if start_time <= midpoint < end_time:
            append_token(
                tokens,
                str(word.get("token", "")).strip(),
                bool(word.get("is_punctuation", False)),
            )
    return " ".join(tokens)


def build_chunks_for_meeting(row):
    audio_path = row["audio_path"]
    transcript_path = row.get("transcript_path")
    if not transcript_path:
        raise ValueError(
            "metadata row has no transcript_path. Run "
            "python -m speech_embedding.build_transcript_metadata first."
        )

    duration = audio_duration_seconds(audio_path)
    words = read_json(transcript_path)
    num_chunks = max(1, math.ceil(duration / CHUNK_SECONDS))

    chunks = []
    for chunk_index in range(num_chunks):
        start_time = chunk_index * CHUNK_SECONDS
        end_time = min(start_time + CHUNK_SECONDS, duration)
        transcript = transcript_for_window(words, start_time, end_time)
        word_count = len(transcript.split())
        if word_count < MIN_WORDS_PER_CHUNK:
            continue

        meeting_id = row["meeting_id"]
        chunks.append(
            {
                "sample_id": f"{meeting_id}_chunk_{chunk_index:04d}",
                "meeting_id": meeting_id,
                "chunk_index": chunk_index,
                "audio_path": project_posix(audio_path),
                "start_time": round(start_time, 3),
                "duration": round(end_time - start_time, 3),
                "transcript": transcript,
                "word_count": word_count,
            }
        )

    return chunks


def pick_meetings(rows):
    usable = []
    for row in rows:
        if not row.get("audio_path") or not row.get("transcript_path"):
            continue
        if not project_path(row["audio_path"]).exists():
            continue
        if not project_path(row["transcript_path"]).exists():
            continue
        usable.append(row)

    needed = TRAIN_MEETINGS + VAL_MEETINGS + TEST_MEETINGS
    if len(usable) < needed:
        raise ValueError(f"Need {needed} usable meetings, found {len(usable)}.")

    return {
        "train": usable[:TRAIN_MEETINGS],
        "val": usable[TRAIN_MEETINGS : TRAIN_MEETINGS + VAL_MEETINGS],
        "test": usable[
            TRAIN_MEETINGS + VAL_MEETINGS : TRAIN_MEETINGS + VAL_MEETINGS + TEST_MEETINGS
        ],
    }


def main():
    rows = read_jsonl(SOURCE_METADATA_PATH)
    splits = pick_meetings(rows)

    for split_name, split_rows in splits.items():
        chunk_rows = []
        for row in split_rows:
            meeting_chunks = build_chunks_for_meeting(row)
            chunk_rows.extend(meeting_chunks)

        output_path = OUTPUT_DIR / f"{split_name}.jsonl"
        write_jsonl(output_path, chunk_rows)
        meeting_ids = sorted({row["meeting_id"] for row in chunk_rows})
        print(
            f"{split_name}: meetings={len(meeting_ids)}, "
            f"chunks={len(chunk_rows)}, output={output_path}"
        )

    print(f"chunk_seconds={CHUNK_SECONDS}, min_words_per_chunk={MIN_WORDS_PER_CHUNK}")
    print(f"project_root={PROJECT_ROOT}")


if __name__ == "__main__":
    main()
