"""Generic chat-completions HTTP client; no implicit retries or text truncation."""
import json
import time
import urllib.request
from urllib.parse import urlparse


class JudgeClient:
    def __init__(self, base_url, model, api_key, temperature=0.0, timeout=300):
        parsed = urlparse(base_url)
        if parsed.scheme not in ("https", "http") or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Expected an HTTP(S) API base URL without credentials/query/fragment")
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Use HTTPS for remote endpoints, or localhost for an SSH tunnel")
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.model, self.api_key, self.temperature, self.timeout = model, api_key, temperature, timeout

    def __call__(self, messages):
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "stream": False, "response_format": {"type": "json_object"}}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers=headers)
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        choice = result["choices"][0]
        return {"text": choice["message"].get("content") or "", "finish_reason": choice.get("finish_reason"),
                "usage": result.get("usage"), "seconds": time.perf_counter() - start,
                "response_model": result.get("model"), "system_fingerprint": result.get("system_fingerprint")}
