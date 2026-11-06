"""Expose the existing runner over AG-UI JSON event frames on WebSocket."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Callable

from ag_ui.core import (RunAgentInput, RunErrorEvent, RunFinishedEvent, RunStartedEvent,
                        StateSnapshotEvent, TextMessageContentEvent, TextMessageEndEvent,
                        TextMessageStartEvent)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from receiver_kv_workflow.engine import ReceiverKVConfig, ReceiverKVRunner

logger = logging.getLogger(__name__)
MAX_INPUT_BYTES = 5 * 1024 * 1024


def encode(event) -> str:
    return event.model_dump_json(by_alias=True, exclude_none=True)


def transcript_from_input(data: RunAgentInput) -> str:
    """Take the last plain-text user message; don't trust client paths or settings."""
    for message in reversed(data.messages):
        if message.role == "user":
            if not isinstance(message.content, str):
                raise ValueError("The meeting transcript must be a plain-text user message.")
            text = message.content.strip()
            if not text:
                raise ValueError("The meeting transcript is empty.")
            return text
    raise ValueError("A user message with the meeting transcript is required.")


class MeetingService:
    """One loaded model and one serialized inference queue per server process."""
    def __init__(self, config: ReceiverKVConfig, runner_factory: Callable = ReceiverKVRunner):
        self.config = config
        self.runner_factory = runner_factory
        self.runner = None
        self.lock = asyncio.Lock()
        self.background: set[asyncio.Task] = set()

    async def execute(self, data: RunAgentInput, transcript: str, queue: asyncio.Queue) -> None:
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        state = {"stage": "queued", "message": "Waiting for the meeting model", "reviewStatus": "pending", "metrics": {}}

        def publish(patch: dict) -> None:
            nonlocal state
            metrics = {**state.get("metrics", {}), **patch.get("metrics", {})}
            state = {**state, **patch, "metrics": metrics}
            state["metrics"]["elapsedSeconds"] = round(time.perf_counter() - started, 2)
            event = encode(StateSnapshotEvent(snapshot=state.copy()))
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if current_loop is loop:
                queue.put_nowait(event)
            else:
                loop.call_soon_threadsafe(queue.put_nowait, event)

        try:
            publish({})
            async with self.lock:
                def inference():
                    if self.runner is None:
                        publish({"stage": "queued", "message": "Loading the meeting model"})
                        self.runner = self.runner_factory(self.config)
                    return self.runner.run(data.run_id, transcript, variants=("routed_kv",), review=True,
                                           on_state=publish)
                result = await asyncio.to_thread(inference)
            review = result.get("review") or {"status": "skipped"}
            draft = result["answers"]["routed_kv"]
            summary = review.get("answer", "") if review.get("status") == "completed" else draft
            if not summary.strip():
                raise ValueError("The meeting workflow returned an empty summary.")
            publish({"stage": "completed", "message": "Review complete" if review.get("status") == "completed" else "Review skipped; showing the draft",
                     "reviewStatus": "completed" if review.get("status") == "completed" else "skipped"})
            message_id = str(uuid.uuid4())
            await queue.put(encode(TextMessageStartEvent(message_id=message_id, role="assistant")))
            await queue.put(encode(TextMessageContentEvent(message_id=message_id, delta=summary)))
            await queue.put(encode(TextMessageEndEvent(message_id=message_id)))
            await queue.put(encode(RunFinishedEvent(thread_id=data.thread_id, run_id=data.run_id)))
        except Exception as exc:
            logger.exception("Meeting run failed (%s)", type(exc).__name__)
            message = str(exc) if isinstance(exc, ValueError) else "Meeting processing failed; check the server log."
            await queue.put(encode(RunErrorEvent(message=message)))
        finally:
            await queue.put(None)

    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.background.add(task)
        task.add_done_callback(self.background.discard)
        return task


def create_app(config: ReceiverKVConfig | None = None, runner_factory: Callable = ReceiverKVRunner) -> FastAPI:
    service = MeetingService(config or ReceiverKVConfig(), runner_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if service.background:
            await asyncio.gather(*service.background, return_exceptions=True)

    app = FastAPI(title="Meeting Summary AG-UI stream", lifespan=lifespan)
    app.state.meeting_service = service

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": service.runner is not None}

    @app.websocket("/ws")
    async def stream(ws: WebSocket):
        await ws.accept()
        try:
            raw = await ws.receive_text()
            if len(raw.encode("utf-8")) > MAX_INPUT_BYTES:
                raise ValueError("Input exceeds the 5 MiB limit.")
            data = RunAgentInput.model_validate_json(raw)
        except WebSocketDisconnect:
            return
        except ValueError as exc:
            await ws.send_text(encode(RunErrorEvent(message=str(exc))))
            await ws.close(code=1008)
            return
        await ws.send_text(encode(RunStartedEvent(thread_id=data.thread_id, run_id=data.run_id)))
        try:
            transcript = transcript_from_input(data)
        except ValueError as exc:
            await ws.send_text(encode(RunErrorEvent(message=str(exc))))
            await ws.close(code=1008)
            return
        queue: asyncio.Queue = asyncio.Queue()
        service.spawn(service.execute(data, transcript, queue))

        async def disconnected():
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message["type"] == "websocket.receive":
                    await ws.close(code=1008, reason="Only one run is allowed per connection")
                    return

        watcher = asyncio.create_task(disconnected())
        try:
            while True:
                get_event = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait((get_event, watcher), return_when=asyncio.FIRST_COMPLETED)
                if watcher in done:
                    get_event.cancel()
                    await asyncio.gather(get_event, return_exceptions=True)
                    break
                event = get_event.result()
                if event is None:
                    break
                await ws.send_text(event)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            try:
                await ws.close()
            except RuntimeError:
                pass

    return app
