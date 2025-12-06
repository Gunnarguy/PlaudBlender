import os
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from gui.views.base import BaseView


class ChatView(BaseView):
    """Lightweight chat UI powered by OpenAI Responses API."""

    def _build(self):  # pragma: no cover - UI code
        self.model_var = tk.StringVar(value=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1"))
        self.temp_var = tk.DoubleVar(value=0.7)
        self.system_var = tk.StringVar(value="You are PlaudBlender's assistant. Be concise.")
        self.show_adv = tk.BooleanVar(value=False)

        header = ttk.Frame(self, style="Panel.TFrame", padding=10)
        header.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(header, text="Chat", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="OpenAI Responses API", style="Muted.TLabel").pack(side=tk.LEFT, padx=(8, 0))

        controls = ttk.Frame(header, style="Panel.TFrame")
        controls.pack(side=tk.RIGHT)
        ttk.Label(controls, text="Model", style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 4))
        model_entry = ttk.Entry(controls, textvariable=self.model_var, width=18)
        model_entry.grid(row=0, column=1, sticky="w")

        ttk.Label(controls, text="Temp", style="Muted.TLabel").grid(row=0, column=2, sticky="w", padx=(10, 4))
        temp_spin = ttk.Spinbox(controls, from_=0.0, to=2.0, increment=0.1, textvariable=self.temp_var, width=5)
        temp_spin.grid(row=0, column=3, sticky="w")

        ttk.Button(controls, text="New chat", command=self.clear_chat).grid(row=0, column=4, padx=(10, 0))

        body = ttk.Frame(self, style="Main.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        self.transcript = scrolledtext.ScrolledText(body, wrap="word", state="disabled", height=20)
        self.transcript.grid(row=0, column=0, sticky="nsew")

        input_frame = ttk.Frame(body, padding=(0, 8, 0, 0))
        input_frame.grid(row=1, column=0, sticky="ew")
        input_frame.columnconfigure(0, weight=1)

        ttk.Label(input_frame, text="System (optional)", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.system_entry = ttk.Entry(input_frame, textvariable=self.system_var)
        self.system_entry.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(input_frame, text="Message", style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        self.prompt_box = scrolledtext.ScrolledText(input_frame, height=4, wrap="word")
        self.prompt_box.grid(row=3, column=0, sticky="ew")

        actions = ttk.Frame(input_frame)
        actions.grid(row=4, column=0, sticky="e", pady=(6, 0))
        self.send_btn = ttk.Button(actions, text="Send", command=self._on_send)
        self.send_btn.pack(side=tk.RIGHT)

        adv_toggle = ttk.Checkbutton(actions, text="Advanced overrides", variable=self.show_adv, command=self._toggle_adv)
        adv_toggle.pack(side=tk.LEFT, padx=(0, 8))

        self.adv_frame = ttk.LabelFrame(input_frame, text="Overrides (JSON)", padding=8)
        self.adv_frame.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        self.adv_frame.columnconfigure(0, weight=1)
        self.adv_text = scrolledtext.ScrolledText(self.adv_frame, height=4, wrap="word")
        self.adv_text.grid(row=0, column=0, sticky="ew")
        self.adv_frame.grid_remove()

        self.status_label = ttk.Label(input_frame, text="", style="Muted.TLabel")
        self.status_label.grid(row=6, column=0, sticky="w", pady=(4, 0))

        self.history = []

    def on_show(self):  # pragma: no cover - UI code
        try:
            self.prompt_box.focus_set()
        except Exception:
            pass

    def clear_chat(self):  # pragma: no cover - UI code
        self.history = []
        self._append_text("system", "Chat cleared.")

    # ------------------- Internal helpers ----------------------------
    def _on_send(self):  # pragma: no cover - UI code
        prompt = self.prompt_box.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Chat", "Enter a message first.")
            return

        system = self.system_var.get().strip()
        # We pass system prompt via instructions, not as a message, to avoid duplication
        self.history.append({"role": "user", "content": prompt})
        self._append_text("you", prompt)
        self.prompt_box.delete("1.0", tk.END)
        self._set_sending(True)

        overrides = None
        if self.show_adv.get():
            raw = self.adv_text.get("1.0", tk.END).strip()
            if raw:
                try:
                    overrides = json.loads(raw)
                except json.JSONDecodeError as exc:
                    messagebox.showerror("Chat", f"Invalid JSON in overrides: {exc}")
                    self._set_sending(False)
                    return

        payload = {
            "messages": self.history,
            "model": self.model_var.get().strip() or "gpt-4.1",
            "temperature": float(self.temp_var.get()),
            "instructions": system or None,
            "overrides": overrides,
        }

        def after_response(result):
            text = result.get("text", "") if isinstance(result, dict) else ""
            self.history.append({"role": "assistant", "content": text})
            self._append_text("assistant", text or "(no text)")

        def always():
            self._set_sending(False)

        self.call("chat_send", payload, after_response, always)

    def _append_text(self, speaker: str, text: str):  # pragma: no cover - UI code
        self.transcript.configure(state="normal")
        self.transcript.insert(tk.END, f"{speaker}: {text}\n\n")
        self.transcript.see(tk.END)
        self.transcript.configure(state="disabled")

    def _set_sending(self, sending: bool):  # pragma: no cover - UI code
        state = tk.DISABLED if sending else tk.NORMAL
        self.send_btn.configure(state=state)
        self.status_label.configure(text="Sending..." if sending else "")

    def _toggle_adv(self):  # pragma: no cover - UI code
        if self.show_adv.get():
            self.adv_frame.grid()
        else:
            self.adv_frame.grid_remove()