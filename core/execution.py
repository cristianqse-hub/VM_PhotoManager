from __future__ import annotations

import multiprocessing as mp


def execution_loop(stop_event: mp.Event) -> None:
    while not stop_event.is_set():
        stop_event.wait(timeout=0.2)
