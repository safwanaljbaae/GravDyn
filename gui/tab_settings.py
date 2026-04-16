#!/usr/bin/env python3
"""
Settings Tab - Placeholder for application settings.
"""

import tkinter as tk
from tkinter import ttk


class SettingsTab(ttk.Frame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        
        self._create_widgets()
        
    def _create_widgets(self):
        placeholder_label = ttk.Label(
            self, 
            text="Settings Tab\n\nThis tab will contain application settings.",
            font=('Arial', 16),
            justify='center'
        )
        placeholder_label.pack(expand=True, fill='both', padx=20, pady=20)
        
        info_frame = ttk.LabelFrame(self, text="Coming Soon", padding=20)
        info_frame.pack(expand=True, fill='both', padx=50, pady=20)
        
        settings = [
            "- Display preferences",
            "- Default paths configuration",
            "- Model parameters",
            "- Export options"
        ]
        
        for setting in settings:
            ttk.Label(info_frame, text=setting, font=('Arial', 10)).pack(anchor='w', pady=2)
