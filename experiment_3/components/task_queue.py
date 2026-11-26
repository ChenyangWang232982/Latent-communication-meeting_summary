"""Serial subprocess queue with persistent step state and completed-step reuse."""
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from experiment_1.components.storage import atomic_json


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def queue_lock(root):
    with (root/'queue.lock').open('a+b') as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('This queue is already running; do not start a second copy') from exc
        yield


def run_command(command, log_path, cwd):
    """No shell, and no next process until this one has fully exited."""
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONUTF8='1')
    with log_path.open('w', encoding='utf-8') as log:
        log.write(json.dumps(command, ensure_ascii=False)+'\n'); log.flush()
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                log.write(line); log.flush()
                print(line, end='', flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise


def completed_artifact(kind, output):
    """Distinguish finished batches with failed samples from a crashed batch."""
    if kind == 'model':
        path = output/'model.json'
        if path.exists() and Path(json.loads(path.read_text(encoding='utf-8'))['path']).is_dir():
            return dict(artifact=str(output), completed=1, failed=0)
        return None
    if kind == 'evaluation':
        path = output/'summary.json'
        if path.exists() and (output/'summary.txt').exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            if data['paired_samples'] > 0:
                return dict(artifact=str(output), completed=data['paired_samples'],
                    failed=data['planned_workflow_samples']-data['paired_samples'])
        return None
    marker = 'progress.json' if kind == 'direct' else 'status.json'
    paths = list(output.glob('*/'+marker))
    if len(paths) != 1:
        return None
    path = paths[0]
    data = json.loads(path.read_text(encoding='utf-8'))
    complete = data.get('completed_records', data.get('completed', 0))
    failed = data.get('failed', 0)
    if kind == 'direct':
        ready = data.get('finished') and complete+failed == data['planned']
    elif kind == 'workflow':
        ready = complete+failed == data['planned']
    else:
        ready = (path.parent/'comparison.txt').exists() and data['processed'] == data['eligible']
        failed += len(data.get('skipped_source_ids', []))
    if ready and complete > 0:
        return dict(artifact=str(path.parent), completed=complete, failed=failed)
    return None


class SerialQueue:
    def __init__(self, root, state, *, cwd, executor=run_command):
        self.root, self.state, self.cwd, self.executor = root, state, cwd, executor

    def save(self):
        self.state['updated_at_utc'] = now()
        atomic_json(self.root/'queue.json', self.state)
        lines = ['# AMI100 task queue', '', f'Status: **{self.state["status"]}**', '',
            '| task | status | completed items | failed/missing | artifact |',
            '|---|---|---:|---:|---|']
        for name, task in self.state['tasks'].items():
            lines.append(f'| {name} | {task["status"]} | {task.get("completed", "")} | '
                         f'{task.get("failed", "")} | {task.get("artifact", "")} |')
        lines.extend(['', 'Individual sample failures are disclosed; no zero quality scores are invented.',
            'Resume skips completed steps. An interrupted step is rerun as a new attempt; '
            'this queue does not add per-record resume to the existing generators.'])
        (self.root/'QUEUE_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    def step(self, name, kind, build_command):
        task = self.state['tasks'][name]
        if task['status'] in ('completed', 'completed_with_failures'):
            artifact = completed_artifact(kind, Path(task['attempt_dir']))
            if artifact:
                print(f'[queue] Reusing {name}: {artifact["artifact"]}', flush=True)
                return Path(artifact['artifact'])
        # If an earlier task must rerun, later cached tasks may depend on its output.
        order = list(self.state['tasks'])
        for later in order[order.index(name)+1:]:
            self.state['tasks'][later]['status'] = 'pending'
            for key in ('artifact', 'completed', 'failed'):
                self.state['tasks'][later].pop(key, None)
        for key in ('artifact', 'completed', 'failed', 'error', 'returncode', 'finished_at_utc'):
            task.pop(key, None)
        attempt = task.get('attempts', 0)+1
        output = self.root/'runs'/name/f'attempt_{attempt:03d}'
        output.parent.mkdir(parents=True, exist_ok=True)
        log = self.root/'logs'/f'{name}_attempt_{attempt:03d}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        command = build_command(output)
        task.update(status='running', attempts=attempt, attempt_dir=str(output), log=str(log),
                    command=command, started_at_utc=now())
        self.state['status'] = 'running'
        self.save()
        print(f'\n[queue] Starting {name}; log: {log}', flush=True)
        try:
            code = self.executor(command, log, self.cwd)
            artifact = completed_artifact(kind, output)
            task.update(returncode=code, finished_at_utc=now())
            if code not in (0, 1) or artifact is None:
                raise RuntimeError(f'{name} did not finish a usable batch (exit={code}); inspect {log}')
            task.update(**artifact, status='completed_with_failures' if artifact['failed'] else 'completed')
            self.save()
            return Path(artifact['artifact'])
        except BaseException as exc:
            task.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=str(exc), finished_at_utc=now())
            self.state['status'] = task['status']
            self.save()
            raise
