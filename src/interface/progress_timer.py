import time
import threading
import logging
from streamlit.runtime.scriptrunner import add_script_run_ctx

logger = logging.getLogger(__name__)


class ProgressTracker:
    def __init__(self, progress_bar, status_text, estimated_seconds,
                 label_prefix, start_pct, cap_pct):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.estimated_seconds = estimated_seconds
        self.label_prefix = label_prefix
        self.start_pct = start_pct
        self.cap_pct = cap_pct
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None
        self.actual_elapsed = 0.0

    def __enter__(self):
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        add_script_run_ctx(self._thread)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self.actual_elapsed = time.time() - self._start_time
        return False

    def _animate(self):
        while not self._stop_event.is_set():
            elapsed = time.time() - self._start_time

            if self.estimated_seconds and self.estimated_seconds > 0:
                remaining = max(0, self.estimated_seconds - elapsed)
                fraction = min(elapsed / self.estimated_seconds, 1.0)
                pct = int(self.start_pct + fraction * (self.cap_pct - self.start_pct))
                pct = min(pct, self.cap_pct)

                mins, secs = divmod(int(remaining), 60)
                if remaining > 0:
                    time_str = f"~{mins}m{secs:02d}s remaining"
                else:
                    time_str = "finishing up..."

                try:
                    self.status_text.text(f"{self.label_prefix}... ({time_str})")
                    self.progress_bar.progress(pct)
                except Exception:
                    pass
            else:
                # First load: slow heartbeat, creep over ~10 minutes
                fraction = min(elapsed / 600.0, 1.0)
                pct = int(self.start_pct + fraction * (self.cap_pct - self.start_pct))
                pct = min(pct, self.cap_pct)

                elapsed_mins, elapsed_secs = divmod(int(elapsed), 60)
                try:
                    self.status_text.text(
                        f"{self.label_prefix}... (first load, elapsed {elapsed_mins}m{elapsed_secs:02d}s)"
                    )
                    self.progress_bar.progress(pct)
                except Exception:
                    pass

            self._stop_event.wait(timeout=1.0)


def run_timed_progress(progress_bar, status_text, estimated_seconds,
                       label_prefix="Step 2/2: Loading Weights",
                       start_pct=40, cap_pct=95):
    return ProgressTracker(progress_bar, status_text, estimated_seconds,
                           label_prefix, start_pct, cap_pct)
