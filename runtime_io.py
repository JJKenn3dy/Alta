import sys


def configure_stdio():
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if not stream or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except ValueError:
            # Some redirected streams can reject reconfigure(); keep best-effort behavior.
            continue
