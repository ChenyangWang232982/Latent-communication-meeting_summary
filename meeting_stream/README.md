# Meeting Summary -> AG-UI WebSocket

This service adapts the existing receiver KV workflow to the contract in the separate AG-UI project's `docs/interface.md` and `docs/output.md`. The workflow emits structured stage updates through the optional `on_state` callback; the service translates them into AG-UI `STATE_SNAPSHOT` events and emits the final summary as `TEXT_MESSAGE_*`. Output arrives as one completed message, not token-by-token.

## Install and run on the meeting machine

```bash
conda activate latent-agents
python -m pip install -r meeting_stream/requirements.txt
python -m meeting_stream \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --seed 12 \
  --review-seed 42 \
  --review-temperature 0.3 \
  --review-top-p 0.9
```

The service listens on `ws://127.0.0.1:8765/ws` by default and serves `GET /health`. The embedding model is loaded from `models/Qwen3-Embedding-0.6B` unless `--embedding-model` overrides it. First inference loads the models; subsequent runs reuse them in the same process. Use `python -m meeting_stream --help` for options. The service uses the existing routed KV workflow with review enabled and sends the draft with `reviewStatus=skipped` if review is skipped.

## Connect AG-UI

On the AG-UI bridge machine set `MEETING_WS_URL=ws://127.0.0.1:8765/ws` if both services share a machine or port 8765 has been forwarded locally. For a remote meeting machine, run on the bridge host:

```bash
ssh -L 8765:127.0.0.1:8765 user@meeting-host
```

Run the bridge and frontend using their own README. The browser does not connect to the model server directly. Keep the meeting service loopback-bound and use SSH port forwarding for remote access.

## Behavior and limits

The service accepts one AG-UI `RunAgentInput` JSON frame per connection, using the last plain-text user message as the transcript. It emits `RUN_STARTED`, state snapshots, final text events and `RUN_FINISHED`, or `RUN_ERROR` on failure. Input is limited to 5 MiB. The model is shared and tasks are serialized within one server process; do not launch multiple worker processes on one GPU without separate capacity management. A disconnected browser does not interrupt a running GPU inference: it continues holding its queue slot until complete. No transcript is intentionally logged or persisted by this adapter. The backend loads only the model and embedding weights configured at startup, and cannot guarantee the correctness of a generated summary.

## Verify without a model

```bash
python -m unittest meeting_stream.test_service -v
```

The tests use a fake runner and do not establish that GPU inference, model weights, or real transcript processing work in your environment.
