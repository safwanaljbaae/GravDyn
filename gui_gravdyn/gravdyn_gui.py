#!/usr/bin/env python3

import sys
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

try:
    import tkinter as tk
except ImportError:
    raise RuntimeError(
        "Tkinter is required for the GUI.\n"
        "Install it with: sudo apt install python3-tk"
    )
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tab_shape_viewer import ShapeViewerTab
from tab_potential import PotentialTab
from tab_settings import SettingsTab


class GravDynGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GravDyn - Gravitational Potential Calculator")
        self.root.geometry("1200x800")

        try:
            self.root.state('zoomed')
        except Exception:
            try:
                self.root.attributes('-zoomed', True)
            except Exception:
                self.root.attributes('-fullscreen', True)

        self.base_dir = Path(__file__).parent.parent / 'Data'
        self.current_asteroid = None
        self.asteroid_data = None
        self.log_text = None

        self._create_tabs()

    def _create_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.shape_viewer_tab = ShapeViewerTab(self.notebook, self)
        self.potential_tab = PotentialTab(self.notebook, self)
        self.settings_tab = SettingsTab(self.notebook, self)

        self.notebook.add(self.shape_viewer_tab, text="Shape Viewer")
        self.notebook.add(self.potential_tab, text="Potential")
        self.notebook.add(self.settings_tab, text="Settings")

    def _log(self, message):
        if self.log_text:
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.see(tk.END)
            self.root.update_idletasks()

    def get_asteroid_data(self):
        return self.asteroid_data

    def set_asteroid_data(self, data):
        self.asteroid_data = data

    def get_current_asteroid(self):
        return self.current_asteroid

    def set_current_asteroid(self, name):
        self.current_asteroid = name

    def get_base_dir(self):
        return self.base_dir


def main():
    root = tk.Tk()
    GravDynGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
