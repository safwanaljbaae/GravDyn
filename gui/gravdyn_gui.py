#!/usr/bin/env python3
"""
GravDyn GUI - Graphical interface for computing gravitational potential fields.

This application provides a user-friendly interface to:
- Select or enter an asteroid name
- Visualize the asteroid shape
- Define computational grid parameters
- Choose gravitational model
- Compute and save potential fields
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
try:
    import tkinter as tk
except ImportError:
    raise RuntimeError(
        "Tkinter is required for the GUI.\n"
        "Install it with: sudo apt install python3-tk"
    )
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import gravdyn
from gravdyn import (
    shape_verification,
    prepare_polyhedral_model,
    pot_point_mass,
    pot_polyhedral_model,
    batched_pot_mascon,
    pot_expansion,
    build_potential_derivatives,
    generate_layered_mascons,
    load_tetrahedron_data
)
from gravdyn.shape_tools import load_vertices, load_faces
from gravdyn.plot_tools import plot_projection

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
        except:
            try:
                self.root.attributes('-zoomed', True)
            except:
                self.root.attributes('-fullscreen', True)
        
        self.base_dir = Path(__file__).parent.parent / 'Data'
        self.current_asteroid = None
        self.asteroid_data = None
        self.expansion_funcs = None
        
        self._setup_styles()
        self._create_tabs()
        
    def _setup_styles(self):
        style = ttk.Style()
        style.configure('Custom.TFrame', background='#f0f0f0')
        style.configure('Custom.TLabelframe', background='#f0f0f0')
        style.configure('Custom.TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 11, 'bold'))
        
    def _create_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.shape_viewer_tab = ShapeViewerTab(self.notebook, self)
        self.potential_tab = PotentialTab(self.notebook, self)
        self.settings_tab = SettingsTab(self.notebook, self)
        
        self.notebook.add(self.shape_viewer_tab, text="Shape Viewer")
        self.notebook.add(self.potential_tab, text="Potential")
        self.notebook.add(self.settings_tab, text="Settings")
        
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)
        
    def _on_tab_changed(self, event=None):
        pass
        
    def _log(self, message):
        if hasattr(self, 'log_text'):
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
        
    def load_asteroid_list(self):
        asteroids = []
        if self.base_dir.exists():
            for item in self.base_dir.iterdir():
                if item.is_dir() and self._is_valid_asteroid(item.name):
                    asteroids.append(item.name)
        return sorted(asteroids)
        
    def _is_valid_asteroid(self, name):
        asteroid_dir = self.base_dir / name
        required_files = ['modified_v.dat', 'modified_f.dat']
        return all((asteroid_dir / f).exists() for f in required_files)


def main():
    root = tk.Tk()
    app = GravDynGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
