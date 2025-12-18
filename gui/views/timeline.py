import tkinter as tk
from tkinter import ttk, messagebox

from gui.views.base import BaseView
from gui.utils.tooltips import ToolTip


class TimelineView(BaseView):
    """Time-first exploration: weekday/year queries + storyboard generation."""

    def _build(self):
        header = ttk.Frame(self, style="Main.TFrame")
        header.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(header, text="🗓 Timeline", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Chronological exploration (weekday/year) + narrative/storyboard",
            style="Muted.TLabel",
        ).pack(anchor="w")

        controls = ttk.LabelFrame(
            self, text="Query", padding=8, style="Panel.TLabelframe"
        )
        controls.pack(fill=tk.X, pady=(0, 8))

        # Year
        ttk.Label(controls, text="Year:").grid(row=0, column=0, sticky="w")
        self.year_var = tk.StringVar(value="2025")
        year_entry = ttk.Entry(controls, textvariable=self.year_var, width=8)
        year_entry.grid(row=0, column=1, sticky="w", padx=(6, 16))
        ToolTip(year_entry, "Year to scan (uses recording created_at)")

        # Weekday
        ttk.Label(controls, text="Weekday:").grid(row=0, column=2, sticky="w")
        self.weekday_var = tk.StringVar(value="Monday")
        weekday_combo = ttk.Combobox(
            controls,
            textvariable=self.weekday_var,
            values=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            width=12,
            state="readonly",
        )
        weekday_combo.grid(row=0, column=3, sticky="w", padx=(6, 16))
        ToolTip(weekday_combo, "Filter recordings by local weekday")

        # Timezone
        ttk.Label(controls, text="Timezone (optional):").grid(
            row=0, column=4, sticky="w"
        )
        self.tz_var = tk.StringVar(value="")
        tz_entry = ttk.Entry(controls, textvariable=self.tz_var, width=22)
        tz_entry.grid(row=0, column=5, sticky="w", padx=(6, 0))
        ToolTip(
            tz_entry,
            "IANA timezone, e.g. America/Los_Angeles.\n"
            "Leave empty to use system local timezone.",
        )

        # Options
        opts = ttk.Frame(controls, style="Main.TFrame")
        opts.grid(row=1, column=0, columnspan=6, sticky="w", pady=(8, 0))

        self.snippets_var = tk.BooleanVar(value=True)
        snippets_cb = ttk.Checkbutton(
            opts, text="Include transcript snippets", variable=self.snippets_var
        )
        snippets_cb.pack(side=tk.LEFT)

        self.use_llm_var = tk.BooleanVar(value=False)
        llm_cb = ttk.Checkbutton(
            opts,
            text="Generate storyboard with Chat (needs OPENAI_API_KEY)",
            variable=self.use_llm_var,
        )
        llm_cb.pack(side=tk.LEFT, padx=(12, 0))

        # Buttons
        btns = ttk.Frame(controls, style="Main.TFrame")
        btns.grid(row=2, column=0, columnspan=6, sticky="w", pady=(10, 0))

        run_btn = ttk.Button(
            btns, text="Build timeline", style="Accent.TButton", command=self._run
        )
        run_btn.pack(side=tk.LEFT)
        ToolTip(
            run_btn, "Run the weekday-in-year query and render a chronological report"
        )

        narrative_btn = ttk.Button(
            btns, text="Build storyboard", command=self._storyboard
        )
        narrative_btn.pack(side=tk.LEFT, padx=(6, 0))
        ToolTip(
            narrative_btn,
            "Build a narrative/storyboard.\n"
            "- If Chat enabled, uses LLM for higher-level synthesis.\n"
            "- Otherwise renders a deterministic structured report.",
        )

        clear_btn = ttk.Button(btns, text="Clear", command=self._clear)
        clear_btn.pack(side=tk.LEFT, padx=(6, 0))

        # Output
        out_frame = ttk.LabelFrame(
            self, text="Output", padding=6, style="Panel.TLabelframe"
        )
        out_frame.pack(fill=tk.BOTH, expand=True)

        self.output = tk.Text(
            out_frame,
            bg="#0f172a",
            fg="#f8fafc",
            insertbackground="#f8fafc",
            wrap=tk.WORD,
            padx=12,
            pady=12,
            font=("JetBrains Mono", 10),
        )
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(out_frame, command=self.output.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output.config(yscrollcommand=scrollbar.set)

    def _weekday_index(self) -> int:
        name = (self.weekday_var.get() or "Monday").strip()
        mapping = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        return mapping.get(name.lower(), 0)

    def _parse_year(self) -> int:
        raw = (self.year_var.get() or "").strip()
        try:
            y = int(raw)
            if y < 1970 or y > 2100:
                raise ValueError
            return y
        except Exception:
            raise ValueError("Year must be a number like 2025")

    def _run(self):
        try:
            year = self._parse_year()
        except ValueError as exc:
            messagebox.showerror("Timeline", str(exc))
            return

        weekday = self._weekday_index()
        tz_name = (self.tz_var.get() or "").strip() or None

        include_snips = bool(self.snippets_var.get())

        self._set_output("Building timeline...\n")
        self.call(
            "timeline_build_report",
            year,
            weekday,
            tz_name,
            include_snips,
            self._set_output,
        )

    def _storyboard(self):
        try:
            year = self._parse_year()
        except ValueError as exc:
            messagebox.showerror("Timeline", str(exc))
            return

        weekday = self._weekday_index()
        tz_name = (self.tz_var.get() or "").strip() or None

        if bool(self.use_llm_var.get()):
            # Delegate to app action which runs in background with Chat service.
            self._set_output("Building storyboard...\n")
            self.call(
                "timeline_generate_storyboard", year, weekday, tz_name, self._set_output
            )
            return

        self._set_output("Building storyboard (deterministic)...\n")
        self.call(
            "timeline_build_report",
            year,
            weekday,
            tz_name,
            True,
            self._set_output,
        )

    def _clear(self):
        self.output.delete("1.0", tk.END)

    def _set_output(self, text: str):
        self.output.delete("1.0", tk.END)
        self.output.insert("1.0", text or "")
