"""Terminal progress without pretending that EOS generation has a known length."""
import sys
import time

from tqdm.auto import tqdm


class ExperimentProgress:
    def __init__(self):
        self.terminal = sys.stderr.isatty()
        self.overall = tqdm(total=0, desc="Samples", unit="sample", position=0,
                            dynamic_ncols=True, disable=not self.terminal)
        self.activity = None
        self.key = None
        self.last_log = 0.0

    def log(self, message):
        if self.terminal:
            tqdm.write(str(message), file=sys.stderr)
        else:
            print(message, file=sys.stderr, flush=True)

    def sample(self, index, total, sample_id):
        self.overall.total = total
        self.overall.set_description(f"Samples [{index + 1}/{total}] {sample_id}")
        self.key = None

    def state(self, event):
        stage = event["stage"]
        phase = event.get("phase", "")
        key = (stage, phase)
        new = key != self.key
        if new:
            if self.activity is not None:
                self.activity.close()
            self.key = key
            decode = phase == "decode"
            self.activity = tqdm(total=event.get("total"), desc=f"{stage} {phase}".strip(),
                                 unit="tok" if decode else "step", position=1, leave=False,
                                 dynamic_ncols=True, mininterval=0.5, disable=not self.terminal)
        count = event.get("tokens", event.get("completed", 0))
        if "total" in event:
            self.activity.total = event["total"]
        self.activity.update(count - self.activity.n)
        # Disabled tqdm bars do not update n internally.
        if not self.terminal:
            self.activity.n = count
        detail = event.get("stop_reason", event.get("message", ""))
        if "prompt_tokens" in event:
            detail = f"reading {event['prompt_tokens']} prompt tokens"
        self.activity.set_postfix_str(detail, refresh=False)
        now = time.monotonic()
        if not self.terminal and (new or event.get("finished") or now - self.last_log >= 10):
            self.log(f"[experiment_1] {stage} {phase}: {count} "
                     f"{'tokens' if phase == 'decode' else 'steps'}; {detail}")
            self.last_log = now

    def finish_sample(self, status):
        if self.activity is not None:
            self.activity.close()
            self.activity = None
        self.overall.update(1)
        self.overall.set_postfix_str(status)

    def close(self):
        if self.activity is not None:
            self.activity.close()
        self.overall.close()
