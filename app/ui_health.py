"""Non-blocking API health checks; workers never access Streamlit state."""
import logging
import time
import requests

logger = logging.getLogger(__name__)


def check_api(api_url):
    started = time.perf_counter()
    try:
        response = requests.get(f"{api_url}/status", timeout=(1, 1))
        response.raise_for_status()
        status = response.json()
        if not isinstance(status, dict):
            raise ValueError("Invalid status response")
        result = {"state": "online", "status": status}
    except (requests.RequestException, ValueError):
        result = {"state": "unavailable", "status": {}}
    elapsed = (time.perf_counter() - started) * 1000
    logger.info("ui_api_health state=%s duration_ms=%.1f", result["state"], elapsed)
    return result


class APIHealth:
    def __init__(self, ttl=15):
        self.ttl = ttl
        self.url = None
        self.future = None
        self.snapshot = {"state": "checking", "status": {}}
        self.next_check = 0

    def poll(self, api_url, executor, now=None):
        now = time.monotonic() if now is None else now
        if self.url != api_url:
            if self.future is not None:
                self.future.cancel()
            self.url, self.future = api_url, None
            self.snapshot = {"state": "checking", "status": {}}
            self.next_check = 0
        if self.future is not None and self.future.done():
            try:
                self.snapshot = self.future.result()
            except Exception:
                self.snapshot = {"state": "unavailable", "status": {}}
            self.future = None
            self.next_check = now + self.ttl
        if self.future is None and now >= self.next_check:
            self.future = executor.submit(check_api, api_url)
        return self.snapshot
