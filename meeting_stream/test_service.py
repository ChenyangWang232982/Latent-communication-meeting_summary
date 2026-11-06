"""WebSocket contract and model reuse without loading a GPU model."""
import asyncio
import threading
import time
import unittest
from ag_ui.core import Event, RunAgentInput
from fastapi.testclient import TestClient
from pydantic import TypeAdapter
from meeting_stream.service import MeetingService, create_app
from receiver_kv_workflow.engine import ReceiverKVConfig

class FakeRunner:
    loads = 0
    def __init__(self, config):
        type(self).loads += 1
    def run(self, record_id, transcript, *, variants, review, on_state):
        assert variants == ("routed_kv",) and review
        on_state({"stage":"prefill","message":"Reading", "metrics":{"sourcePositions":100}})
        on_state({"stage":"summary","message":"Writing", "metrics":{"selectedPositions":60}})
        on_state({"stage":"review","message":"Reviewing", "reviewStatus":"running"})
        result = {"status":"skipped"} if "skip" in transcript else {"status":"completed","answer":"Reviewed"}
        return {"answers":{"routed_kv":"Draft"}, "review":result}

def request(run_id, content):
    return {"threadId":"thread", "runId":run_id,
            "messages":[{"id":"message", "role":"user", "content":content}],
            "tools":[], "context":[], "forwardedProps":{}}

def receive_run(ws):
    events = []
    while True:
        event = ws.receive_json()
        TypeAdapter(Event).validate_python(event)
        events.append(event)
        if event["type"] in ("RUN_FINISHED", "RUN_ERROR"):
            return events

class MeetingStreamTest(unittest.TestCase):
    def test_review_fallback_status_and_reuse(self):
        FakeRunner.loads = 0
        app = create_app(runner_factory=FakeRunner)
        with TestClient(app) as client:
            self.assertFalse(client.get("/health").json()["model_loaded"])
            for run_id, transcript, expected, status in (("one","Alice: hello","Reviewed","completed"),
                                                          ("two","Alice: skip","Draft","skipped")):
                with client.websocket_connect("/ws") as ws:
                    ws.send_json(request(run_id, transcript))
                    events = receive_run(ws)
                self.assertEqual(events[0]["type"], "RUN_STARTED")
                self.assertEqual(events[-1]["type"], "RUN_FINISHED")
                self.assertEqual(events[-1]["runId"], run_id)
                snapshots = [e["snapshot"] for e in events if e["type"] == "STATE_SNAPSHOT"]
                self.assertEqual(snapshots[-1]["reviewStatus"], status)
                self.assertEqual(snapshots[-1]["metrics"]["sourcePositions"], 100)
                self.assertEqual(snapshots[-1]["metrics"]["selectedPositions"], 60)
                texts = [e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT"]
                self.assertEqual(texts, [expected])
            self.assertEqual(FakeRunner.loads, 1)
            self.assertTrue(client.get("/health").json()["model_loaded"])

    def test_empty_input_emits_terminal_error(self):
        with TestClient(create_app(runner_factory=FakeRunner)) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json(request("empty", " "))
                events = receive_run(ws)
            self.assertEqual([e["type"] for e in events], ["RUN_STARTED", "RUN_ERROR"])
            self.assertIn("empty", events[-1]["message"])

    def test_concurrent_requests_share_one_serialized_runner(self):
        class SlowRunner(FakeRunner):
            loads = 0
            active = 0
            peak = 0
            guard = threading.Lock()

            def run(self, *args, **kwargs):
                with self.guard:
                    type(self).active += 1
                    type(self).peak = max(type(self).peak, type(self).active)
                try:
                    time.sleep(0.05)
                    return super().run(*args, **kwargs)
                finally:
                    with self.guard:
                        type(self).active -= 1

        async def exercise():
            service = MeetingService(ReceiverKVConfig(), SlowRunner)
            inputs = [RunAgentInput.model_validate(request(str(i), "Alice: hello")) for i in range(2)]
            await asyncio.gather(*(service.execute(data, "Alice: hello", asyncio.Queue()) for data in inputs))

        asyncio.run(exercise())
        self.assertEqual(SlowRunner.loads, 1)
        self.assertEqual(SlowRunner.peak, 1)

if __name__ == "__main__":
    unittest.main()
