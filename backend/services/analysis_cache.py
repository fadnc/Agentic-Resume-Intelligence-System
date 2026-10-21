"""Small process-local cache for identical resume and job-description requests."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import threading
import time


_MAX_ENTRIES = 8
_TTL_SECONDS = 15 * 60
_CACHE: OrderedDict[str, tuple[float, dict]] = OrderedDict()
_LOCK = threading.Lock()


def make_analysis_cache_key(pdf_bytes: bytes, job_description: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"resume-analysis-v1\0")
    digest.update(pdf_bytes)
    digest.update(b"\0")
    digest.update(job_description.encode("utf-8"))
    return digest.hexdigest()


def get_cached_analysis(key: str):
    now = time.monotonic()
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        expires_at, response = entry
        if expires_at <= now:
            del _CACHE[key]
            return None
        _CACHE.move_to_end(key)
        return deepcopy(response)


def cache_analysis(key: str, response: dict) -> None:
    now = time.monotonic()
    with _LOCK:
        for old_key, (expires_at, _) in list(_CACHE.items()):
            if expires_at <= now:
                del _CACHE[old_key]
        _CACHE[key] = (now + _TTL_SECONDS, deepcopy(response))
        _CACHE.move_to_end(key)
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)
