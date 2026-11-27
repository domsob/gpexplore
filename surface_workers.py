"""Renders the frames of the rotating 3D plot in worker processes.

One frame costs about 60 ms in matplotlib, nearly all of it Python on one core. Rendered one
after the other, that's ~15 frames per second on a slower CPU. A few worker processes render
frames in parallel instead (the GIL rules out threads); the UI shows the newest finished one.
Until the workers are up (they are started when the pointer first enters a 3D plot), and
if they fail, the caller renders in its own process as before.
"""

import asyncio
import itertools
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import plots

# A frame takes ~60-80 ms and drag events come every ~30 ms, so about three frames are in
# flight; a fourth worker absorbs slower frames. Half the logical CPUs at most (about one per
# physical core), so smaller machines keep a core for the UI. Each worker needs ~165 MB.
WORKERS = max(1, min(int(os.environ.get("GPX_3D_WORKERS", 4)), (os.cpu_count() or 2) // 2))
IDLE_SECONDS = 120   # workers are stopped after this long without a frame (and restarted on hover)

_state = {"pool": None, "ready": False, "broken": False, "key": None, "token": None, "payload": None,
          "last_used": 0.0}
_tokens = itertools.count(1)


def _stop_when_idle():
    while True:
        time.sleep(10)
        pool = _state["pool"]
        if pool is None:
            return
        if time.monotonic() - _state["last_used"] > IDLE_SECONDS:
            _state["pool"], _state["ready"] = None, False
            pool.shutdown(wait=False, cancel_futures=True)
            return


def start():
    """Start the worker processes in the background (no-op if they run or failed)."""
    if _state["pool"] is not None or _state["broken"]:
        _state["last_used"] = time.monotonic()
        return
    _state["last_used"] = time.monotonic()
    threading.Thread(target=_stop_when_idle, daemon=True).start()
    try:
        pool = ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn"))
        _state["pool"] = pool
        warm = [pool.submit(plots.warm_up) for _ in range(WORKERS)]
    except Exception:
        _fail()
        return
    remaining = {"n": len(warm)}

    def done(future):
        if _state["pool"] is not pool:   # stopped as idle in the meantime
            return
        if future.cancelled() or future.exception() is not None:
            _fail()
            return
        remaining["n"] -= 1
        if remaining["n"] == 0:
            _state["ready"] = True

    for future in warm:
        future.add_done_callback(done)


def ready():
    return _state["ready"] and not _state["broken"]


def _fail():
    _state["broken"], _state["ready"] = True, False
    pool, _state["pool"] = _state["pool"], None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


def _payload(snapshot, problem, train_X, train_y, adjust, size):
    """(token, payload) of the plot's current state; a new token whenever any input changes."""
    key = (snapshot, problem, train_X, train_y, bool(adjust), tuple(size))
    old = _state["key"]
    if old is None or not all(a is b or (type(a) in (bool, tuple) and a == b) for a, b in zip(old, key)):
        _state["key"] = key
        _state["token"] = next(_tokens)
        _state["payload"] = plots.surface_payload(snapshot, problem, train_X, train_y, adjust, size)
    return _state["token"], _state["payload"]


async def render(snapshot, problem, train_X, train_y, adjust, size, view):
    """JPEG bytes of one frame, or None if the workers are not (or no longer) available."""
    if not ready():
        return None
    token, payload = _payload(snapshot, problem, train_X, train_y, adjust, size)
    _state["last_used"] = time.monotonic()
    try:
        future = _state["pool"].submit(plots.render_surface_frame, token, payload, tuple(view))
        return await asyncio.wrap_future(future)
    except Exception:
        _fail()
        return None
