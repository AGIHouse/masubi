"""Thread management for start/stop/pause of the autoresearch loop."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

from run_loop import run_autoresearch


class RunManager:
    """Manages the autoresearch loop in a background thread."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # set = NOT paused (normal running)
        self._current_run_id: str | None = None
        self._status: str = "idle"

    def start(self, max_experiments: int = 50) -> str:
        """Launch run_autoresearch in a daemon thread. Returns run_id."""
        if self._status == "running":
            raise RuntimeError("Already running. Stop the current run first.")

        # Generate a run_id
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self._current_run_id = run_id

        # Reset events
        self._stop_event.clear()
        self._pause_event.set()  # not paused

        self._status = "running"

        self._thread = threading.Thread(
            target=self._run_wrapper,
            args=(max_experiments,),
            daemon=True,
        )
        self._thread.start()
        return run_id

    def stop(self) -> None:
        """Signal graceful stop after current experiment."""
        if self._status not in ("running", "paused"):
            return
        self._status = "stopping"
        self._stop_event.set()
        # Also unpause so the loop can exit
        self._pause_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._status = "idle"
        self._thread = None

    def pause(self) -> None:
        """Pause between experiments."""
        if self._status == "running":
            self._pause_event.clear()  # clear = paused
            self._status = "paused"

    def resume(self) -> None:
        """Resume from pause."""
        if self._status == "paused":
            self._pause_event.set()  # set = not paused
            self._status = "running"

    @property
    def status(self) -> str:
        return self._status

    @property
    def current_run_id(self) -> str | None:
        return self._current_run_id

    def _stop_check(self) -> bool:
        """Check if stop has been requested."""
        return self._stop_event.is_set()

    def _pause_check(self) -> bool:
        """Check if pause is active. Returns True when paused."""
        return not self._pause_event.is_set()

    def _run_wrapper(self, max_experiments: int) -> None:
        """Wrapper that runs run_autoresearch and cleans up status on exit."""
        try:
            run_autoresearch(
                max_experiments=max_experiments,
                stop_check=self._stop_check,
                pause_check=self._pause_check,
            )
        except Exception:
            pass
        finally:
            if self._status != "idle":
                self._status = "idle"
