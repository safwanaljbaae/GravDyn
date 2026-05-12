#!/usr/bin/env python3

from tkinter import ttk


class SettingsTab(ttk.Frame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self._create_widgets()

    def _create_widgets(self):
        ttk.Label(
            self,
            text="Settings Tab\n\nThis tab will contain application settings.",
            font=('Arial', 16),
            justify='center'
        ).pack(expand=True, fill='both', padx=20, pady=20)

        info_frame = ttk.LabelFrame(self, text="Coming Soon", padding=20)
        info_frame.pack(expand=True, fill='both', padx=50, pady=20)

        for setting in [
            "- Display preferences",
            "- Default paths configuration",
            "- Model parameters",
            "- Export options"
        ]:
            ttk.Label(info_frame, text=setting, font=('Arial', 10)).pack(anchor='w', pady=2)
