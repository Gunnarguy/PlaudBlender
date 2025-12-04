import threading
from typing import Callable, Optional


def run_async(task: Callable, callback: Optional[Callable] = None, *, tk_root=None):
    """Run *task* in a daemon thread and hand the result to *callback* on the Tk thread."""

    def _dispatch(result):
        if not callback:
            return
        try:
            callback(result)
        except Exception as callback_exc:  # pylint: disable=broad-except
            print(f"Callback failed: {callback_exc}")

    def worker():
        try:
            result = task()
        except Exception as exc:  # pylint: disable=broad-except
            result = exc
        if tk_root:
            tk_root.after(0, lambda: _dispatch(result))
        else:
            _dispatch(result)

    threading.Thread(target=worker, daemon=True).start()
