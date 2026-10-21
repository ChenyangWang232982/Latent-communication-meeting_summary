# Data Preparation

## MeetingBank

Downloads the official Hugging Face text split and normalizes it for
`meeting_evaluation`:

```bash
python -m data_preprocess.prepare_meetingbank
```

The default is `huuuyeah/meetingbank`, split `test`, and writes:

```text
input/meetingbank_test.jsonl
input/meetingbank_test.jsonl.report.json
```

The JSONL contains only the normalized fields used by the evaluation harness:

```json
{"id":"MeetingBank UID","transcript":"...","summary":"..."}
```

The source data provides `id`, `uid`, `transcript`, and `summary`; `uid` is
preferred as the stable record ID. The converter does not download audio or
video.

Useful variations:

```bash
# Small smoke-test export.
python -m data_preprocess.prepare_meetingbank --max-records 8

# Rebuild an existing export deliberately.
python -m data_preprocess.prepare_meetingbank --overwrite

# Use a data-disk Hugging Face cache.
python -m data_preprocess.prepare_meetingbank --cache-dir /root/autodl-tmp/hf-cache/datasets
```

It requires the `datasets` Python package. The generated JSONL is suitable for:

```bash
python -m meeting_evaluation.test --benchmark meetingbank --workflow receiver_kv --input input/meetingbank_test.jsonl
```
