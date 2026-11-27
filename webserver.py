"""Web mode: the Flet app plus a download route for exports.

In the browser, Flet saves a file by sending its bytes over the WebSocket and letting the
page click a blob link. Some browsers and embedded browser views ignore such downloads,
and large exports (PDF reports, raw CSVs of many runs) are slow over the socket. So every
export is kept here for a while and offered as an ordinary HTTP download instead.
"""

import secrets
import threading
import time
from urllib.parse import quote

KEEP_SECONDS = 15 * 60   # how long an export can be downloaded
MAX_FILES = 40           # older exports are dropped beyond this

_files = {}              # token -> (created, file name, media type, bytes)
_lock = threading.Lock()

MEDIA_TYPES = {"pdf": "application/pdf", "zip": "application/zip", "csv": "text/csv",
               "png": "image/png", "svg": "image/svg+xml", "py": "text/x-python"}


def offer(data, file_name):
    """Keep `data` for download and return its URL path (relative to the server root)."""
    ext = file_name.rsplit(".", 1)[-1].lower()
    token = secrets.token_urlsafe(16)
    with _lock:
        now = time.time()
        for key in [k for k, (created, *_) in _files.items() if now - created > KEEP_SECONDS]:
            del _files[key]
        while len(_files) >= MAX_FILES:
            del _files[next(iter(_files))]
        _files[token] = (now, file_name, MEDIA_TYPES.get(ext, "application/octet-stream"), bytes(data))
    return f"/download/{token}/{quote(file_name)}"


def _download(token, file_name=""):
    from starlette.responses import PlainTextResponse, Response
    with _lock:
        entry = _files.get(token)
    if entry is None or time.time() - entry[0] > KEEP_SECONDS:
        return PlainTextResponse("This download has expired. Export the file again in GPExplore.",
                                 status_code=404)
    _, name, media_type, data = entry
    disposition = f"attachment; filename=\"{name.encode('ascii', 'replace').decode()}\"; " \
                  f"filename*=UTF-8''{quote(name)}"
    return Response(data, media_type=media_type,
                    headers={"Content-Disposition": disposition, "Cache-Control": "no-store"})


def run(main, port):
    """Serve the app at http://127.0.0.1:<port> with the download route, then open the browser."""
    import asyncio

    import uvicorn
    import flet_web.fastapi as flet_fastapi
    from flet.utils.browser import open_in_browser

    app = flet_fastapi.FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    # before the Flet app, which is mounted at "/" and would answer every other path
    app.add_api_route("/download/{token}/{file_name}", _download, methods=["GET"])
    app.mount("/", flet_fastapi.app(main))
    url = f"http://127.0.0.1:{port}"

    async def serve():
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                               ws="websockets-sansio"))
        task = asyncio.create_task(server.serve())
        while not server.started and not task.done():
            await asyncio.sleep(0.05)
        if server.started:
            print(f"GPExplore is running at {url}", flush=True)
            open_in_browser(url)
        await task

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
