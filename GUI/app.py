from __future__ import annotations

import multiprocessing as mp
import json
import os
from pathlib import Path
import re
import shutil
import tkinter as tk


class LogViewerApp:
    def __init__(self, root: tk.Tk, stop_event: mp.Event, log_path: str) -> None:
        self.root = root
        self.stop_event = stop_event
        self.log_path = Path(log_path)
        self.commands_dir = Path("Temps/Commands")
        self.photos_dir = Path("Temps/Photos")
        self.temps_dir = Path("Temps")
        self.runtime_state_path = Path("Temps/runtime_state.json")
        self.last_content = ""
        self.last_commands_content = ""
        self.last_photos_signature = ""
        self.debug_mode = False
        self.last_render_debug_mode = self.debug_mode

        self.root.title("takePhoto - GUI Basica")
        self.root.geometry("1000x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.build_layout()
        self.refresh_log()

    def build_layout(self) -> None:
        dark_bg = "#030d2e"
        panel_bg = "#061544"
        accent_bg = "#0a1f5e"

        self.root.configure(bg=dark_bg)

        container = tk.Frame(self.root, bg=dark_bg)
        container.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.96, relheight=0.94)

        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=12)
        container.columnconfigure(2, weight=1)
        container.rowconfigure(0, weight=1)

        content = tk.Frame(container, bg=dark_bg)
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1, uniform="cols")
        content.columnconfigure(1, weight=1, uniform="cols")
        content.rowconfigure(0, weight=0)
        content.rowconfigure(1, weight=1)

        top_bar = tk.Frame(content, bg=dark_bg, pady=4)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        top_bar.columnconfigure(0, weight=0)
        top_bar.columnconfigure(1, weight=0)
        top_bar.columnconfigure(2, weight=1)

        self.debug_button = tk.Button(
            top_bar,
            text="Debug: OFF",
            command=self.toggle_debug_mode,
            bg="#102861",
            fg="white",
            activebackground="#1f3f8a",
            activeforeground="white",
            relief="flat",
            padx=12,
            pady=6,
        )
        self.debug_button.grid(row=0, column=0, sticky="w")

        self.clear_button = tk.Button(
            top_bar,
            text="Clear temps",
            command=self.clear_temps,
            bg="#6b1b1b",
            fg="white",
            activebackground="#8d2323",
            activeforeground="white",
            relief="flat",
            padx=12,
            pady=6,
        )
        self.clear_button.grid(row=0, column=1, sticky="w", padx=(8, 0))

        left_frame = tk.Frame(content, bg=panel_bg, padx=8, pady=8)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=(4, 0))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)

        log_top = tk.Frame(left_frame, bg=panel_bg)
        log_top.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        log_top.columnconfigure(0, weight=1)
        log_top.rowconfigure(0, weight=1)

        commands_bottom = tk.Frame(left_frame, bg=panel_bg)
        commands_bottom.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        commands_bottom.columnconfigure(0, weight=1)
        commands_bottom.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_top, wrap="none")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(
            bg=panel_bg,
            fg="white",
            insertbackground="white",
            selectbackground="#2a4bbd",
            relief="flat",
            borderwidth=0,
        )
        self.log_text.tag_configure("err", foreground="#ff4d4d")
        self.log_text.tag_configure("wrn", foreground="#ffb347")
        self.log_text.tag_configure("deb", foreground="#22d3ee")
        self.log_text.tag_configure("default", foreground="white")

        scrollbar = tk.Scrollbar(
            log_top,
            orient="vertical",
            command=self.log_text.yview,
            bg=panel_bg,
            troughcolor=dark_bg,
            activebackground="#1e3c8f",
            highlightthickness=0,
            relief="flat",
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.commands_text = tk.Text(commands_bottom, wrap="none")
        self.commands_text.grid(row=0, column=0, sticky="nsew")
        self.commands_text.configure(
            bg=panel_bg,
            fg="white",
            insertbackground="white",
            selectbackground="#2a4bbd",
            relief="flat",
            borderwidth=0,
            state="disabled",
        )

        commands_scrollbar = tk.Scrollbar(
            commands_bottom,
            orient="vertical",
            command=self.commands_text.yview,
            bg=panel_bg,
            troughcolor=dark_bg,
            activebackground="#1e3c8f",
            highlightthickness=0,
            relief="flat",
        )
        commands_scrollbar.grid(row=0, column=1, sticky="ns")
        self.commands_text.configure(yscrollcommand=commands_scrollbar.set)

        right_frame = tk.Frame(content, bg=accent_bg, padx=12, pady=12)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(4, 0), pady=(4, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(2, weight=1)

        title = tk.Label(
            right_frame,
            text="Panel de fotos temporales",
            bg=accent_bg,
            fg="white",
            font=("TkDefaultFont", 12, "bold"),
            anchor="w",
        )
        title.grid(row=0, column=0, sticky="ew")

        self.photos_info_label = tk.Label(
            right_frame,
            text="Grupos detectados: 0",
            bg=accent_bg,
            fg="white",
            justify="left",
            anchor="w",
        )
        self.photos_info_label.grid(row=1, column=0, sticky="ew", pady=(8, 6))

        details_frame = tk.Frame(right_frame, bg=accent_bg)
        details_frame.grid(row=2, column=0, sticky="nsew")
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(1, weight=1)

        self.photos_ids_text = tk.Text(
            details_frame,
            height=7,
            wrap="none",
            bg="#0b2b76",
            fg="white",
            relief="flat",
            borderwidth=0,
            state="disabled",
        )
        self.photos_ids_text.grid(row=0, column=0, sticky="ew")

        ids_scroll = tk.Scrollbar(
            details_frame,
            orient="vertical",
            command=self.photos_ids_text.yview,
            bg=accent_bg,
            troughcolor=dark_bg,
            activebackground="#1e3c8f",
            highlightthickness=0,
            relief="flat",
        )
        ids_scroll.grid(row=0, column=1, sticky="ns")
        self.photos_ids_text.configure(yscrollcommand=ids_scroll.set)

        self.photos_canvas = tk.Canvas(
            details_frame,
            bg="#081a4e",
            highlightthickness=0,
            relief="flat",
        )
        self.photos_canvas.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))

    def toggle_debug_mode(self) -> None:
        self.debug_mode = not self.debug_mode
        self.debug_button.configure(
            text="Debug: ON" if self.debug_mode else "Debug: OFF",
            bg="#1f3f8a" if self.debug_mode else "#102861",
        )

    def render_log(self, content: str) -> None:
        lines = content.splitlines(keepends=True)
        for line in lines:
            if line.startswith("[DEB]") and not self.debug_mode:
                continue

            if line.startswith("[ERR]"):
                tag = "err"
            elif line.startswith("[WRN]"):
                tag = "wrn"
            elif line.startswith("[DEB]"):
                tag = "deb"
            else:
                tag = "default"

            self.log_text.insert("end", line, tag)

    def clear_temps(self) -> None:
        try:
            logs_dir = self.temps_dir / "Logs"
            latest_log_name = None
            if logs_dir.exists():
                log_files = sorted([p.name for p in logs_dir.iterdir() if p.is_file()])
                if log_files:
                    latest_log_name = log_files[-1]

            for subdir in self.temps_dir.iterdir():
                if not subdir.is_dir():
                    continue

                if subdir.name in {"Photos", "Commands"}:
                    for root, _, files in os.walk(subdir):
                        for file_name in files:
                            file_path = Path(root) / file_name
                            try:
                                file_path.unlink()
                            except FileNotFoundError:
                                pass
                    continue

                for child in subdir.iterdir():
                    if (
                        subdir.name == "Logs"
                        and latest_log_name is not None
                        and child.is_file()
                        and child.name == latest_log_name
                    ):
                        continue

                    if child.is_file():
                        try:
                            child.unlink()
                        except FileNotFoundError:
                            pass
                    elif child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)

            print("[INF] Clear temps ejecutado.")
        except Exception as exc:
            print(f"[ERR] Error en Clear temps: {exc}")

    def parse_photo_filename(self, file_name: str) -> dict | None:
        pattern = (
            r"^(?P<id>\d{8}_\d{6,9})Img_?"
            r"(?P<cols>\d+)x(?P<rows>\d+)_"
            r"(?P<i>\d+)-(?P<j>\d+)_"
            r".*\.(?P<ext>bmp|png|jpg|jpeg|tif|tiff)$"
        )
        match = re.match(pattern, file_name, flags=re.IGNORECASE)
        if not match:
            return None

        data = match.groupdict()
        return {
            "group_id": data["id"],
            "cols": int(data["cols"]),
            "rows": int(data["rows"]),
            "i": int(data["i"]),
            "j": int(data["j"]),
            "name": file_name,
        }

    def collect_photos_data(self) -> tuple[list[dict], str]:
        if not self.photos_dir.exists():
            return [], ""

        files = sorted([p for p in self.photos_dir.iterdir() if p.is_file()], key=lambda p: p.name.lower())
        parsed: list[dict] = []

        for file_path in files:
            parsed_info = self.parse_photo_filename(file_path.name)
            if parsed_info is not None:
                parsed.append(parsed_info)

        signature = "|".join([f["name"] for f in parsed])
        return parsed, signature

    def load_runtime_state(self) -> dict:
        if not self.runtime_state_path.exists():
            return {}
        try:
            return json.loads(self.runtime_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def update_photos_ids_panel(self, parsed_files: list[dict]) -> None:
        groups: dict[str, dict] = {}
        for item in parsed_files:
            group_id = item["group_id"]
            group = groups.setdefault(
                group_id,
                {
                    "cols": item["cols"],
                    "rows": item["rows"],
                    "coords": set(),
                    "invalid_dimensions": False,
                },
            )
            if group["cols"] != item["cols"] or group["rows"] != item["rows"]:
                group["invalid_dimensions"] = True
                group["cols"] = max(group["cols"], item["cols"])
                group["rows"] = max(group["rows"], item["rows"])
            group["coords"].add((item["i"], item["j"]))

        runtime_state = self.load_runtime_state()
        object_ids = set(runtime_state.get("object_ids", []))

        sorted_group_ids = sorted(groups.keys())
        groups_count = len(sorted_group_ids)
        self.photos_info_label.configure(text=f"Grupos detectados: {len(groups)}")

        lines: list[str] = []
        for group_id in sorted_group_ids:
            group = groups[group_id]
            cols = group["cols"]
            rows = group["rows"]
            expected = cols * rows
            found = len(group["coords"])
            marker = "O" if group_id in object_ids else "X"
            if group["invalid_dimensions"]:
                state = "INVALID_DIMS"
            elif found == expected:
                state = "COMPLETE"
            else:
                state = "INCOMPLETE"
            lines.append(f"{marker} {group_id} {found}/{expected} ({cols}x{rows}) {state}")

        self.photos_info_label.configure(text=f"Sets detectados: {groups_count}")

        self.photos_ids_text.configure(state="normal")
        self.photos_ids_text.delete("1.0", "end")
        if lines:
            self.photos_ids_text.insert("1.0", "\n".join(lines))
        else:
            self.photos_ids_text.insert("1.0", "Sin sets en Temps/Photos")
        self.photos_ids_text.configure(state="disabled")

    def draw_photos_grid(self, parsed_files: list[dict]) -> None:
        self.photos_canvas.delete("all")
        self.photos_canvas.create_text(
            10,
            10,
            anchor="nw",
            fill="white",
            text=(
                "Estado de sets:\n"
                "O = existe objeto en runtime para ese ID\n"
                "X = no existe objeto para ese ID\n"
                "COMPLETE = set completo\n"
                "INCOMPLETE = faltan imagenes"
            ),
        )

    def refresh_photos(self) -> None:
        parsed, signature = self.collect_photos_data()
        canvas_size = f"{self.photos_canvas.winfo_width()}x{self.photos_canvas.winfo_height()}"
        render_signature = f"{signature}|{canvas_size}"

        if render_signature != self.last_photos_signature:
            self.last_photos_signature = render_signature
            self.update_photos_ids_panel(parsed)
            self.draw_photos_grid(parsed)

        self.root.after(300, self.refresh_photos)

    def collect_commands_content(self) -> str:
        if not self.commands_dir.exists():
            return ""

        lines: list[str] = []
        files = sorted(
            [p for p in self.commands_dir.iterdir() if p.is_file()],
            key=lambda p: p.name.lower(),
        )

        for file_path in files:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            for raw_line in text.splitlines():
                command = raw_line.strip()
                if command:
                    lines.append(f"->{command}")

        return "\n".join(lines)

    def refresh_commands(self) -> None:
        content = self.collect_commands_content()
        if content != self.last_commands_content:
            self.last_commands_content = content
            at_bottom = self.commands_text.yview()[1] >= 0.999
            top_fraction = self.commands_text.yview()[0]

            self.commands_text.configure(state="normal")
            self.commands_text.delete("1.0", "end")
            if content:
                self.commands_text.insert("1.0", content)
            self.commands_text.configure(state="disabled")

            if at_bottom:
                self.commands_text.see("end")
            else:
                self.commands_text.yview_moveto(top_fraction)

        self.root.after(1000, self.refresh_commands)

    def refresh_log(self) -> None:
        if self.stop_event.is_set():
            self.root.after(200, self.refresh_log)
            return

        if self.log_path.exists():
            content = self.log_path.read_text(encoding="utf-8", errors="replace")
            if content != self.last_content or self.debug_mode != self.last_render_debug_mode:
                self.last_content = content
                self.last_render_debug_mode = self.debug_mode

                at_bottom = self.log_text.yview()[1] >= 0.999
                top_fraction = self.log_text.yview()[0]

                self.log_text.configure(state="normal")
                self.log_text.delete("1.0", "end")
                self.render_log(content)
                self.log_text.configure(state="disabled")

                if at_bottom:
                    self.log_text.see("end")
                else:
                    self.log_text.yview_moveto(top_fraction)

        self.root.after(200, self.refresh_log)

    def on_close(self) -> None:
        self.stop_event.set()
        self.root.destroy()


def launch_gui(stop_event: mp.Event, log_path: str) -> None:
    root = tk.Tk()
    app = LogViewerApp(root, stop_event=stop_event, log_path=log_path)
    app.refresh_commands()
    app.refresh_photos()
    root.mainloop()
