#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import os
import numpy as np
import trimesh
import requests
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from gravdyn.shape_tools import load_vertices, load_faces
import gravdyn

OWNER = "safwanaljbaae"
REPO = "GravDyn"
BRANCH = "main"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def _get_headers():
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def load_shape_characteristics(vertices, faces, asteroid_name, mass, density, base_dir):

    base_path = Path(base_dir)
    asteroid_dir = base_path / asteroid_name
    asteroid_dir.mkdir(parents=True, exist_ok=True)

    log_file = asteroid_dir / "shape_verification.log"

    print(log_file.read_text())
    return log_file.read_text() if log_file.exists() else "Shape verification log not found."


class ShapeViewerTab(ttk.Frame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.log_text = None
        self._create_widgets()

    def _create_widgets(self):
        left_frame = ttk.Frame(self, width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)

        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._create_selection_frame(left_frame)
        self._create_characteristics_frame(left_frame)
        self._create_custom_verification_frame(left_frame)

        viz_log_frame = ttk.Frame(right_frame)
        viz_log_frame.pack(fill=tk.BOTH, expand=True)

        self._create_visualization_frame(viz_log_frame)

        self.log_toggle_frame = ttk.Frame(viz_log_frame)
        self.log_toggle_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.log_toggle_btn = ttk.Button(
            self.log_toggle_frame,
            text="▼ Show Log",
            command=self._toggle_log,
            width=15
        )
        self.log_toggle_btn.pack()

        self._create_log_frame(viz_log_frame)

    def _create_selection_frame(self, parent):
        sel_frame = ttk.LabelFrame(parent, text="Asteroid Selection", padding=10)
        sel_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(sel_frame, text="Available Asteroids:", style='Header.TLabel').pack(anchor='w')

        self.asteroid_var = tk.StringVar()
        self.asteroid_combo = ttk.Combobox(
            sel_frame,
            textvariable=self.asteroid_var,
            state='readonly',
            width=25
        )
        self.asteroid_combo.pack(fill=tk.X, pady=(5, 0))
        self.asteroid_combo.bind('<<ComboboxSelected>>', self._on_asteroid_selected)

        ttk.Label(sel_frame, text="Or enter name:").pack(anchor='w', pady=(10, 0))
        self.custom_name_var = tk.StringVar()
        custom_entry = ttk.Entry(sel_frame, textvariable=self.custom_name_var, width=25)
        custom_entry.pack(fill=tk.X, pady=(5, 0))
        custom_entry.bind('<Return>', lambda e: self._check_asteroid())

        self.check_btn = ttk.Button(sel_frame, text="Check & Load", command=self._check_asteroid)
        self.check_btn.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(sel_frame, text="", foreground="blue")
        self.status_label.pack(anchor='w')

        ttk.Separator(sel_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        self._load_asteroid_list()

    def _create_characteristics_frame(self, parent):
        char_frame = ttk.LabelFrame(parent, text="Shape Characteristics", padding=10)
        char_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        text_scroll_frame = ttk.Frame(char_frame)
        text_scroll_frame.pack(fill=tk.BOTH, expand=True)

        self.char_text = tk.Text(text_scroll_frame, height=10, width=50, state='disabled', font=('Courier', 10), wrap='none')

        v_scrollbar = ttk.Scrollbar(text_scroll_frame, command=self.char_text.yview)
        h_scrollbar = ttk.Scrollbar(text_scroll_frame, orient=tk.HORIZONTAL, command=self.char_text.xview)

        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.char_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.char_text.config(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

    def _create_custom_verification_frame(self, parent):
        cv_frame = ttk.LabelFrame(parent, text="Custom Shape Verification", padding=10)
        cv_frame.pack(fill=tk.X, padx=5, pady=5)

        file_frame = ttk.Frame(cv_frame)
        file_frame.pack(fill=tk.X, pady=5)

        ttk.Label(file_frame, text="Asteroid Name:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.asteroid_name_var = tk.StringVar(value="custom")
        name_entry = ttk.Entry(file_frame, textvariable=self.asteroid_name_var, width=25)
        name_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=2)

        ttk.Label(file_frame, text="Vertices File:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.vertices_file_var = tk.StringVar()
        vertices_entry = ttk.Entry(file_frame, textvariable=self.vertices_file_var, width=25)
        vertices_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(file_frame, text="Browse", command=self._browse_vertices).grid(row=1, column=2, padx=5)

        ttk.Label(file_frame, text="Faces File:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.faces_file_var = tk.StringVar()
        faces_entry = ttk.Entry(file_frame, textvariable=self.faces_file_var, width=25)
        faces_entry.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(file_frame, text="Browse", command=self._browse_faces).grid(row=2, column=2, padx=5)

        file_frame.columnconfigure(1, weight=1)

        param_frame = ttk.Frame(cv_frame)
        param_frame.pack(fill=tk.X, pady=5)

        ttk.Label(param_frame, text="Mass (kg):").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.mass_var = tk.StringVar(value="1e12")
        mass_entry = ttk.Entry(param_frame, textvariable=self.mass_var, width=20)
        mass_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=2)

        ttk.Label(param_frame, text="Density (kg/m³):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.density_var = tk.StringVar(value="1.75")
        density_entry = ttk.Entry(param_frame, textvariable=self.density_var, width=20)
        density_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=2)

        param_frame.columnconfigure(1, weight=1)

        self.apply_btn = ttk.Button(cv_frame, text="Apply Shape Verification", command=self._apply_custom_verification)
        self.apply_btn.pack(fill=tk.X, pady=5)

    def _browse_vertices(self):
        filename = filedialog.askopenfilename(
            title="Select Vertices File",
            filetypes=[("Data files", "*.dat"), ("All files", "*.*")]
        )
        if filename:
            self.vertices_file_var.set(filename)

    def _browse_faces(self):
        filename = filedialog.askopenfilename(
            title="Select Faces File",
            filetypes=[("Data files", "*.dat"), ("All files", "*.*")]
        )
        if filename:
            self.faces_file_var.set(filename)

    def _create_visualization_frame(self, parent):
        viz_frame = ttk.LabelFrame(parent, text="Asteroid Shape Visualization", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(fill=tk.X)

        ttk.Button(toolbar_frame, text="XY", command=lambda: self._plot_projection('xy')).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="XZ", command=lambda: self._plot_projection('xz')).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="YZ", command=lambda: self._plot_projection('yz')).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="3D", command=self._plot_3d).pack(side=tk.LEFT, padx=2)

    def _create_log_frame(self, parent):
        self.log_content_frame = ttk.Frame(parent)

        log_frame = ttk.LabelFrame(self.log_content_frame, text="Log / Results", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(log_frame, height=10, yscrollcommand=scrollbar.set)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)

        self.log_visible = True
        self._log("Shape Viewer initialized.")
        self._log("Load an asteroid to see detailed logs.")
        self.log_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_toggle_btn.config(text="▲ Hide Log")

    def _toggle_log(self):
        if self.log_visible:
            self.log_content_frame.pack_forget()
            self.log_toggle_btn.config(text="▼ Show Log")
            self.log_visible = False
        else:
            self.log_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.log_toggle_btn.config(text="▲ Hide Log")
            self.log_visible = True

    def _load_asteroid_list(self):
        try:
            url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/Data"
            response = requests.get(url, params={"ref": BRANCH}, headers=_get_headers(), timeout=20)
            if response.status_code == 200:
                data = response.json()
                asteroids = sorted(item["name"] for item in data if item["type"] == "dir")
                self.asteroid_combo['values'] = asteroids
                self._log(f"Loaded {len(asteroids)} asteroids from GitHub repository")
            else:
                self._log(f"GitHub API error: {response.status_code}")
                self.asteroid_combo['values'] = []
        except Exception as e:
            self._log(f"Error loading asteroid list from GitHub: {e}")
            self.asteroid_combo['values'] = []

    def _on_asteroid_selected(self, event=None):
        name = self.asteroid_var.get()
        if name:
            self.custom_name_var.set(name)
            self._load_asteroid(name)

    def _check_asteroid(self):
        name = self.custom_name_var.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter an asteroid name")
            return
        self._load_asteroid(name)

    def _load_asteroid(self, name):
        self.char_text.config(state='normal')
        self.char_text.delete('1.0', tk.END)
        self.char_text.config(state='disabled')
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state='disabled')
        self.status_label.config(text=f"Downloading {name} from GitHub...", foreground="blue")

        gui_dir = Path(__file__).parent
        data_dir = gui_dir / "Data"
        asteroid_dir = data_dir / name

        if not asteroid_dir.exists():
            self._log("Downloading asteroid data from GitHub...")
            try:
                url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/Data/{name}"
                response = requests.get(url, params={"ref": BRANCH}, headers=_get_headers(), timeout=20)
                if response.status_code != 200:
                    raise FileNotFoundError(f"Asteroid '{name}' not found on GitHub")

                data_dir.mkdir(parents=True, exist_ok=True)
                asteroid_dir.mkdir(parents=True, exist_ok=True)

                files = response.json()
                for item in files:
                    if item["type"] == "file":
                        download_url = item["download_url"]
                        file_name = item["name"]
                        file_path = asteroid_dir / file_name

                        file_response = requests.get(download_url, headers=_get_headers(), timeout=30)
                        if file_response.status_code == 200:
                            with open(file_path, 'wb') as f:
                                f.write(file_response.content)
                            self._log(f"  Downloaded: {file_name}")

            except Exception as e:
                self.status_label.config(text=f"Error downloading: {str(e)}", foreground="red")
                self._log(f"Download error: {e}")
                return
        else:
            self._log(f"Using cached data for {name}")

        try:
            vertices_path = asteroid_dir / "input_v.dat"
            faces_path = asteroid_dir / "input_f.dat"

            if not vertices_path.exists() or not faces_path.exists():
                vertices_path = asteroid_dir / "modified_v.dat"
                faces_path = asteroid_dir / "modified_f.dat"

            vertices = load_vertices(str(vertices_path))
            faces = load_faces(str(faces_path))

            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

            self.main_app.set_current_asteroid(name)
            self.main_app.set_asteroid_data({
                'name': name,
                'vertices': vertices,
                'faces': faces,
                'mesh': mesh,
                'dir': asteroid_dir
            })

            self._log(f"Loaded asteroid: {name}")
            self._log(f"  Vertices: {len(vertices)}")
            self._log(f"  Faces: {len(faces)}")

            self.status_label.config(text=f"✓ {name} loaded successfully", foreground="green")

            self._plot_3d()

            mass = float(self.mass_var.get())
            density = float(self.density_var.get())
            self._update_characteristics(vertices, faces, name, mass, density, str(data_dir))

        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")
            self._log(f"Error loading asteroid: {e}")
            self.main_app.set_current_asteroid(None)

    def _update_characteristics(self, vertices, faces, asteroid_name, mass, density, base_dir):
        self.char_text.config(state='normal')
        self.char_text.delete('1.0', tk.END)

        info = load_shape_characteristics(vertices, faces, asteroid_name, mass, density, base_dir)
        self.char_text.insert('1.0', info)

        self.char_text.config(state='disabled')

    def _apply_custom_verification(self):
        asteroid_name = self.asteroid_name_var.get().strip()
        vertices_path = self.vertices_file_var.get().strip()
        faces_path = self.faces_file_var.get().strip()

        if not asteroid_name:
            messagebox.showwarning("Warning", "Please enter an asteroid name")
            return

        if not vertices_path or not faces_path:
            messagebox.showwarning("Warning", "Please select both vertices and faces files")
            return

        try:
            mass = float(self.mass_var.get())
            density = float(self.density_var.get())
            if mass <= 0 or density <= 0:
                raise ValueError("Mass and density must be positive")
        except ValueError as e:
            messagebox.showwarning("Invalid Input", f"Please enter valid positive numbers.\nError: {e}")
            return

        try:
            vertices = load_vertices(vertices_path)
            faces = load_faces(faces_path)

            gui_dir = Path(__file__).parent
            data_dir = gui_dir / "Data"
            asteroid_dir = data_dir / asteroid_name

            self._log("\n===== Shape Verification Output =====\n")
            self._update_characteristics(vertices, faces, asteroid_name, mass, density, str(data_dir))

            verified_vertices = load_vertices(str(asteroid_dir / "modified_v.dat"))
            verified_faces = load_faces(str(asteroid_dir / "modified_f.dat"))

            verified_mesh = trimesh.Trimesh(vertices=verified_vertices, faces=verified_faces, process=False)

            current_data = self.main_app.get_asteroid_data() or {}
            current_data['input_vertices'] = vertices
            current_data['input_faces'] = faces
            current_data['verified_vertices'] = verified_vertices
            current_data['verified_faces'] = verified_faces
            current_data['verified_mesh'] = verified_mesh
            current_data['input_mesh'] = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            current_data['name'] = asteroid_name
            current_data['dir'] = asteroid_dir
            self.main_app.set_asteroid_data(current_data)
            self.main_app.set_current_asteroid(asteroid_name)

            self._plot_verified_shape(verified_mesh)

            messagebox.showinfo("Success", "Shape verification completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Shape verification failed: {str(e)}")

    def _plot_surf(self, mesh, cmap='viridis', title=None):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            alpha=0.7,
            cmap=cmap
        )
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title(f"{self.main_app.get_current_asteroid()} - 3D Shape")
        max_range = np.max(np.abs(mesh.vertices)) * 1.1
        self.ax.set_xlim(-max_range, max_range)
        self.ax.set_ylim(-max_range, max_range)
        self.ax.set_zlim(-max_range, max_range)
        self.ax.set_box_aspect((1, 1, 1))
        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_3d(self):
        data = self.main_app.get_asteroid_data()
        if not data:
            return
        self._plot_surf(data['mesh'])

    def _plot_verified_shape(self, mesh):
        self._plot_surf(mesh, cmap='plasma', title=f"{self.main_app.get_current_asteroid()} - Verified Shape")

    def _plot_projection(self, plane):
        data = self.main_app.get_asteroid_data()
        if not data:
            return

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        mesh = data['mesh']

        projection_map = {'xy': (0, 1, 'X', 'Y'), 'xz': (0, 2, 'X', 'Z'), 'yz': (1, 2, 'Y', 'Z')}
        i, j, xlabel, ylabel = projection_map.get(plane, (0, 1, 'X', 'Y'))

        for face in mesh.faces:
            poly = mesh.vertices[face][:, [i, j]]
            poly = np.vstack([poly, poly[0]])
            self.ax.plot(poly[:, 0], poly[:, 1], 'k-', linewidth=0.5)

        coords = mesh.vertices[:, [i, j]]
        max_extent = np.max(np.abs(coords)) * 1.1
        self.ax.set_xlim(-max_extent, max_extent)
        self.ax.set_ylim(-max_extent, max_extent)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_aspect('equal')
        self.ax.set_title(f"{self.main_app.get_current_asteroid()} - {plane.upper()} Projection")

        self.fig.tight_layout()
        self.canvas.draw()

    def _log(self, message):
        if self.log_text:
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, message + '\n')
            self.log_text.see(tk.END)
            self.log_text.config(state='disabled')
