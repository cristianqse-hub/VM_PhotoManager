#!/usr/bin/env python3
import multiprocessing as mp
import threading

from GUI.app import launch_gui
from core.execution import execution_loop
from core.logging_utils import redirect_process_output_to_log


def main() -> None:
    mp.set_start_method("spawn", force=True)

    log_path, log_handle = redirect_process_output_to_log()
    stop_event = mp.Event()

    worker_thread = threading.Thread(
        target=execution_loop,
        args=(stop_event,),
        name="ExecutionThread",
        daemon=True,
    )
    worker_thread.start()

    gui_process = mp.Process(
        target=launch_gui,
        args=(stop_event, str(log_path)),
        name="GUIProcess",
    )
    gui_process.start()
    print(f"[INF] Main iniciado. Log: {log_path}. GUI y hilo de ejecucion lanzados.")

    try:
        while gui_process.is_alive():
            gui_process.join(timeout=0.5)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()

        worker_thread.join(timeout=5.0)
        if worker_thread.is_alive():
            print("[WRN] El hilo de ejecucion no finalizo a tiempo. Cierre forzado.")

        if gui_process.is_alive():
            print("[WRN] La GUI no finalizo naturalmente. Terminando proceso GUI.")
            gui_process.terminate()
            gui_process.join(timeout=2.0)
        log_handle.flush()
        log_handle.close()


if __name__ == "__main__":
    main()
