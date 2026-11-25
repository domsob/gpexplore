"""Put a PNG image on the system clipboard.

Flet's Clipboard.set_image is not available on every desktop platform, so the
image is handed to the operating system's own tools instead.
"""

import os
import platform
import shutil
import subprocess
import tempfile


class ClipboardError(RuntimeError):
    pass


def copy_png(data: bytes) -> None:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        system = platform.system()
        if system == "Darwin":
            script = f'set the clipboard to (read (POSIX file "{path}") as «class PNGf»)'
            _run(["osascript", "-e", script])
        elif system == "Windows":
            ps = ("Add-Type -AssemblyName System.Windows.Forms, System.Drawing; "
                  f"$img = [System.Drawing.Image]::FromFile('{path}'); "
                  "[System.Windows.Forms.Clipboard]::SetImage($img); $img.Dispose()")
            _run(["powershell", "-NoProfile", "-STA", "-Command", ps])
        elif shutil.which("wl-copy") and os.environ.get("WAYLAND_DISPLAY"):
            with open(path, "rb") as f:
                _run(["wl-copy", "--type", "image/png"], stdin=f)
        elif shutil.which("xclip"):
            _run(["xclip", "-selection", "clipboard", "-t", "image/png", "-i", path])
        else:
            raise ClipboardError("no clipboard tool found (install wl-clipboard or xclip)")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _run(cmd, stdin=None):
    try:
        result = subprocess.run(cmd, stdin=stdin, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ClipboardError(str(exc)) from None
    if result.returncode != 0:
        raise ClipboardError(result.stderr.decode(errors="replace").strip() or f"{cmd[0]} failed")
