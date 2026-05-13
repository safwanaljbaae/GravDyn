#!/usr/bin/env python3

import os
import re
import requests
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

from gravdyn.shape_tools import load_vertices, load_faces
from gravdyn.prepare_polyhedral_model import prepare_werner_model
from gravdyn.build_potential_derivatives import build_potential_derivatives
from gravdyn.generate_layered_mascons import generate_layered_mascons, load_tetrahedron_data
from gravdyn.pot_functions import (
    pot_expansion,
    batched_werner_potential,
    compute_pseudo_potential,
    batched_pot_mascon,
)
from gravdyn.constants import GRAVITATIONAL_CONSTANT


G = GRAVITATIONAL_CONSTANT

OWNER = "safwanaljbaae"
REPO = "GravDyn"
BRANCH = "main"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def _get_headers():
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


class PotentialTab(ttk.Frame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app

        self.model_type = tk.StringVar(value="polyhedral")
        self.plane_type = tk.StringVar(value="xy")
        self.radius_var = tk.StringVar(value="1.5")
        self.num_points = tk.IntVar(value=100)
        self.rot_period_var = tk.StringVar(value="30.0")
        self.layers_var = tk.StringVar(value="10")

        self._current_vertices = None
        self._current_faces = None
        self._asteroid_name = None
        self._shape_source = None
        self.asteroid_dir = None
        self.potential_data = None

        self._create_widgets()

    @staticmethod
    def _extract_mass(file_path):
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if "the considered mass is" in line:
                        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                        if m:
                            return float(m.group())
        except (FileNotFoundError, OSError):
            pass
        return None

    def _create_widgets(self):
        left_frame = ttk.Frame(self, width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)

        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._create_selection_frame(left_frame)
        self._create_settings_frame(left_frame)
        self._create_compute_frame(left_frame)

        self._create_plot_area(right_frame)

    def _create_selection_frame(self, parent):
        sel_frame = ttk.LabelFrame(parent, text="Asteroid Selection", padding=10)
        sel_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(sel_frame, text="Available Asteroids:").pack(anchor='w')

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

        name_frame = ttk.Frame(sel_frame)
        name_frame.pack(fill=tk.X, pady=5)

        self.custom_name_var = tk.StringVar()
        name_entry = ttk.Entry(name_frame, textvariable=self.custom_name_var, width=25)
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(name_frame, text="Load", command=self._check_custom_name, width=8).pack(side=tk.LEFT, padx=2)

        self.status_label = ttk.Label(sel_frame, text="", foreground="blue")
        self.status_label.pack(anchor='w')

        self._load_asteroid_list()

    def _create_settings_frame(self, parent):
        set_frame = ttk.LabelFrame(parent, text="Potential Settings", padding=10)
        set_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(set_frame, text="Model:").pack(anchor='w')

        ttk.Radiobutton(
            set_frame, text="Classical Polyhedral", variable=self.model_type,
            value="polyhedral", command=self._toggle_layers_entry
        ).pack(anchor='w', padx=15)

        ttk.Radiobutton(
            set_frame, text="Expansion Method", variable=self.model_type,
            value="expansion", command=self._toggle_layers_entry
        ).pack(anchor='w', padx=15)

        ttk.Radiobutton(
            set_frame, text="Mascon (SHAPED Polyhedral)", variable=self.model_type,
            value="mascon", command=self._toggle_layers_entry
        ).pack(anchor='w', padx=15)

        self.layers_frame = ttk.Frame(set_frame)
        self.layers_frame.pack(fill=tk.X, padx=20, pady=(0, 5))

        ttk.Label(self.layers_frame, text="Layers:").pack(side=tk.LEFT)
        self.layers_entry = ttk.Entry(self.layers_frame, textvariable=self.layers_var, width=8)
        self.layers_entry.pack(side=tk.LEFT, padx=5)
        self.layers_entry.config(state='disabled')

        ttk.Separator(set_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(set_frame, text="Grid Plane:").pack(anchor='w')

        ttk.Radiobutton(set_frame, text="XY", variable=self.plane_type, value="xy").pack(anchor='w', padx=15)
        ttk.Radiobutton(set_frame, text="YZ", variable=self.plane_type, value="yz").pack(anchor='w', padx=15)
        ttk.Radiobutton(set_frame, text="XZ", variable=self.plane_type, value="xz").pack(anchor='w', padx=15)

        ttk.Separator(set_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        grid_frame = ttk.Frame(set_frame)
        grid_frame.pack(fill=tk.X)

        ttk.Label(grid_frame, text="Radius:").grid(row=0, column=0, sticky='w')
        self.radius_entry = ttk.Entry(grid_frame, textvariable=self.radius_var, width=10)
        self.radius_entry.grid(row=0, column=1, padx=5, sticky='ew')

        ttk.Label(grid_frame, text="Points:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(grid_frame, textvariable=self.num_points, width=10).grid(row=1, column=1, padx=5, sticky='ew', pady=2)

        ttk.Label(grid_frame, text="Rot. Period (h):").grid(row=2, column=0, sticky='w', pady=2)
        ttk.Entry(grid_frame, textvariable=self.rot_period_var, width=10).grid(row=2, column=1, padx=5, sticky='ew', pady=2)

        grid_frame.columnconfigure(1, weight=1)

    def _create_compute_frame(self, parent):
        comp_frame = ttk.Frame(parent)
        comp_frame.pack(fill=tk.X, padx=5, pady=10)

        self.compute_status_label = ttk.Label(comp_frame, text="", font=('TkDefaultFont', 9, 'italic'))
        self.compute_status_label.pack(fill=tk.X, pady=(0, 5))

        self.compute_btn = ttk.Button(
            comp_frame,
            text="Compute & Plot Potential",
            command=self._compute_potential
        )
        self.compute_btn.pack(fill=tk.X)

    def _create_plot_area(self, parent):
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _load_asteroid_list(self):
        try:
            url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/Data"
            response = requests.get(url, params={"ref": BRANCH}, headers=_get_headers(), timeout=20)
            if response.status_code == 200:
                data = response.json()
                self._asteroid_list = sorted(item["name"] for item in data if item["type"] == "dir")
                self.asteroid_combo['values'] = self._asteroid_list
            else:
                self._asteroid_combo['values'] = []
        except Exception as e:
            self._log(f"Error loading asteroid list from GitHub: {e}")
            self.asteroid_combo['values'] = []

    def _log(self, message):
        if hasattr(self.main_app, '_log'):
            self.main_app._log(message)

    def _on_asteroid_selected(self, event=None):
        name = self.asteroid_var.get()
        if name:
            self._log(f"Selected: {name}")
            self._load_asteroid(name)

    def _toggle_layers_entry(self):
        state = 'normal' if self.model_type.get() == "mascon" else 'disabled'
        self.layers_entry.config(state=state)

    def _load_asteroid(self, name):
        self.status_label.config(text=f"Loading {name} from GitHub...", foreground="blue")

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

                self._log(f"Downloaded files for {name}")

            except Exception as e:
                self.status_label.config(text=f"Error downloading: {str(e)}", foreground="red")
                self._log(f"Download error: {e}")
                return
        else:
            self._log(f"Using cached data for {name}")

        vertices_path = asteroid_dir / "input_v.dat"
        faces_path = asteroid_dir / "input_f.dat"

        if not vertices_path.exists() or not faces_path.exists():
            vertices_path = asteroid_dir / "shape_v.dat"
            faces_path = asteroid_dir / "shape_f.dat"

        if not vertices_path.exists() or not faces_path.exists():
            self.status_label.config(text="Shape files not found", foreground="red")
            self._log("Shape files not found")
            return

        try:
            self._current_vertices = load_vertices(str(vertices_path))
            self._current_faces = load_faces(str(faces_path))

            self.asteroid_dir = asteroid_dir
            self._asteroid_name = name
            self._shape_source = 'asteroid'

            self.status_label.config(text=f"Loaded: {name}", foreground="green")
            self._log(f"Loaded shape for {name}")

        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")
            self._log(f"Error loading asteroid: {e}")

    def _check_custom_name(self):
        name = self.custom_name_var.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter an asteroid name")
            return
        self._load_asteroid(name)

    def _load_from_main_app(self):
        if self._asteroid_name is not None or self._current_vertices is not None:
            return

        main_data = self.main_app.get_asteroid_data()
        if main_data and 'verified_vertices' in main_data:
            self._current_vertices = main_data['verified_vertices']
            self._current_faces = main_data['verified_faces']
            self._asteroid_name = main_data.get('name', 'asteroid')
            self.asteroid_dir = main_data.get('dir')
            self._shape_source = 'viewer'
            self._log(f"Using shape from Shape Viewer: {self._asteroid_name}")
            return

        ast_var = self.asteroid_var.get().strip()
        if ast_var:
            self._asteroid_name = ast_var
            self._shape_source = 'asteroid'
            return

        custom_name = self.custom_name_var.get().strip()
        if custom_name:
            self._load_asteroid(custom_name)

    def _compute_potential(self):
        self._load_from_main_app()

        source = getattr(self, '_shape_source', None)

        if source not in ('asteroid', 'files'):
            if not self.asteroid_var.get() and not self.custom_name_var.get():
                messagebox.showwarning("No Data", "Please select an asteroid or enter a name first")
                return
            messagebox.showwarning("No Data", "Please load an asteroid shape first")
            return

        asteroid_name = getattr(self, '_asteroid_name', None)

        self._log(f"Computing potential for {asteroid_name or 'custom'}...")

        try:
            R = float(self.radius_var.get())
            n = self.num_points.get()
            model = self.model_type.get()
            plane = self.plane_type.get()

            self.compute_status_label.config(text="Processing...")
            self.compute_btn.config(state='disabled')
            self._log(f"Computing potential ({model})...")

            self.after(100, lambda: self._do_compute(R, n, model, plane, asteroid_name))

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for radius and points")

    def _do_compute(self, rmax, n, model, plane, asteroid_name):
        try:
            if plane == "xy":
                x = np.linspace(-rmax, rmax, n)
                y = np.linspace(-rmax, rmax, n)
                X, Y = np.meshgrid(x, y)
                points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n * n)])
                plane_slice = (X, Y, np.zeros_like(X))
            elif plane == "yz":
                y = np.linspace(-rmax, rmax, n)
                z = np.linspace(-rmax, rmax, n)
                Y, Z = np.meshgrid(y, z)
                points = np.column_stack([np.zeros(n * n), Y.ravel(), Z.ravel()])
                plane_slice = (np.zeros_like(Y), Y, Z)
            else:
                x = np.linspace(-rmax, rmax, n)
                z = np.linspace(-rmax, rmax, n)
                X, Z = np.meshgrid(x, z)
                points = np.column_stack([X.ravel(), np.zeros(n * n), Z.ravel()])
                plane_slice = (X, np.zeros_like(X), Z)

            dist_from_center = np.linalg.norm(points, axis=1)

            if self._current_vertices is not None:
                r_brillouin = np.linalg.norm(self._current_vertices, axis=1).min()
            else:
                r_brillouin = rmax * 0.8

            inside_mask = dist_from_center < r_brillouin

            self._log(f"Computing potential at {n * n} grid points...")

            if model == "polyhedral":
                p, acc = self._compute_polyhedral(asteroid_name, points)
            elif model == "mascon":
                nlayers = int(self.layers_var.get())
                p, acc = self._compute_mascon(asteroid_name, points, nlayers)
            elif model == "expansion":
                p, acc = self._compute_expansion(asteroid_name, points)
            else:
                return

            p = np.asarray(p).ravel().copy()
            p[inside_mask] = np.nan
            p = p.reshape(n, n)

            rot_period_hours = float(self.rot_period_var.get())
            pseudo_p = compute_pseudo_potential(points, p.ravel(), rot_period_hours)
            pseudo_p = np.asarray(pseudo_p).ravel().copy()
            pseudo_p[inside_mask] = np.nan
            pseudo_p = pseudo_p.reshape(n, n)

            self.potential_data = {
                'U': p,
                'pseudo': pseudo_p,
                'plane': plane,
                'model': model,
                'points': points,
                'plane_slice': plane_slice,
            }

            if self.asteroid_dir:
                data_to_save = np.column_stack([points, p.ravel(), pseudo_p.ravel()])
                save_path = self.asteroid_dir / f"potential_{model}_{plane}.dat"
                np.savetxt(save_path, data_to_save, fmt="%.10e",
                           header="x y z potential pseudo_potential")
                self._log(f"Saved to {save_path}")

            self._plot_zvc()

            if model in ("expansion", "mascon"):
                self._plot_relative_error_inset()

            self.compute_status_label.config(text="Done")
            self.compute_btn.config(state='normal')
            self._log(f"Done. Potential range: {np.nanmin(p):.4e} to {np.nanmax(p):.4e}")

        except Exception as e:
            self.compute_status_label.config(text="")
            self.compute_btn.config(state='normal')
            self._log(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def _compute_polyhedral(self, asteroid_name, points):
        self._log("Preparing polyhedral model...")
        polyhedral_data = prepare_werner_model(
            asteroid=asteroid_name,
            base_dir="Data",
            verbose=False,
        )

        file_path = f"Data/{asteroid_name}/shape_verification.log"
        mass = self._extract_mass(file_path)

        if mass is None:
            raise ValueError(f"Could not extract mass from {file_path}")

        gm_body = mass * G
        p, acc = batched_werner_potential(
            gm_body=gm_body,
            stat=points,
            polyhedral_data=polyhedral_data,
            batch_size=2000,
        )
        return p, acc

    def _compute_mascon(self, asteroid_name, points, nlayers):
        file_path = f"Data/{asteroid_name}/shape_verification.log"
        mass = self._extract_mass(file_path)

        if mass is None:
            raise ValueError(f"Could not extract mass from {file_path}")

        generate_layered_mascons(
            base_dir='Data/',
            asteroid=asteroid_name,
            total_mass=mass,
            densities=[mass / (G * 1e12)] * nlayers,
            output_csv="layered_mascons.csv",
        )
        data_shape = load_tetrahedron_data(
            asteroid=asteroid_name,
            base_dir="Data",
            tetrahedron_data_file="layered_mascons.csv",
        )

        p, acc = batched_pot_mascon(points, data_shape, batch_size=2000)
        return p, acc

    def _compute_expansion(self, asteroid_name, points):
        file_path = f"Data/{asteroid_name}/shape_verification.log"
        mass = self._extract_mass(file_path)

        if mass is None:
            raise ValueError(f"Could not extract mass from {file_path}")

        gm_body = mass * G

        f_pot_expansion, f_d_pot_expansion = build_potential_derivatives(
            name_central_body=asteroid_name,
            pattern="pot_*.dat",
            n_files=700,
            gm0=gm_body,
            lambdify_backend="jax",
            base_dir="Data",
            verbose=True,
        )

        p, acc = pot_expansion(
            stat=points,
            f_pot_expansion=f_pot_expansion,
            f_d_pot_expansion=f_d_pot_expansion,
        )
        return p, acc

    def _plot_zvc(self):
        if self.potential_data is None:
            return

        self.fig.clear()

        pseudo = self.potential_data.get('pseudo')
        plane = self.potential_data.get('plane', 'xy')
        X, Y, Z = self.potential_data['plane_slice']
        model = self.potential_data.get('model', 'polyhedral')

        coord_map = {
            'xy': (X, Y, 'X', 'Y'),
            'yz': (Y, Z, 'Y', 'Z'),
            'xz': (X, Z, 'X', 'Z'),
        }
        x_coord, y_coord, xlabel, ylabel = coord_map.get(plane, coord_map['xy'])

        if model in ("expansion", "mascon"):
            ax = self.fig.add_subplot(121)
        else:
            ax = self.fig.add_subplot(111)

        if pseudo is not None:
            valid_mask = ~np.isnan(pseudo)
            if valid_mask.any():
                pseudo_min, pseudo_max = np.nanmin(pseudo), np.nanmax(pseudo)
                levels = np.linspace(pseudo_min, pseudo_max, 2500)
                ax.contour(
                    x_coord, y_coord, pseudo,
                    levels=levels,
                    colors='royalblue',
                    linewidths=1.2,
                    linestyles='solid',
                    alpha=0.8,
                )

        if self._current_vertices is not None:
            vertices = np.asarray(self._current_vertices)
            faces = np.asarray(self._current_faces)

            proj_map = {'xy': slice(0, 2), 'yz': slice(1, 3), 'xz': (0, 2)}
            v_proj = vertices[:, proj_map.get(plane, slice(0, 2))]

            for face in faces:
                pts = v_proj[face]
                ax.plot(pts[:, 0], pts[:, 1], 'k-', linewidth=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Zero Velocity Curves - {model.title()}')
        ax.set_aspect('equal')
        self.canvas.draw()

    def _load_classical_potential(self, plane):
        if not self.asteroid_dir:
            return None
        classical_file = self.asteroid_dir / f"potential_polyhedral_{plane}.dat"
        if not classical_file.exists():
            return None
        try:
            return np.loadtxt(classical_file)
        except Exception:
            return None

    def _plot_relative_error_inset(self):
        model = self.potential_data.get('model', '')
        plane = self.potential_data.get('plane', 'xy')

        classical_data = self._load_classical_potential(plane)
        if classical_data is None:
            messagebox.showwarning(
                "Missing Reference Data",
                "Classical Polyhedral potential data not found.\n"
                "Please run the Classical Polyhedral method first."
            )
            return

        classical_potential = classical_data[:, 3]
        current_potential = self.potential_data['U'].ravel()
        points = self.potential_data['points']

        dist_from_center = np.linalg.norm(points, axis=1)
        if self._current_vertices is not None:
            r_brillouin = np.linalg.norm(self._current_vertices, axis=1).min()
        else:
            r_brillouin = float(self.radius_var.get()) * 0.8

        inside_mask = dist_from_center < r_brillouin

        valid_mask = (
            ~np.isnan(current_potential)
            & ~np.isnan(classical_potential)
            & ~inside_mask
            & (np.abs(classical_potential) > 1e-20)
        )

        r = dist_from_center
        d_pot = np.zeros_like(current_potential)
        d_pot[valid_mask] = (
            (current_potential[valid_mask] - classical_potential[valid_mask])
            * 100 / classical_potential[valid_mask]
        )
        d_pot[~valid_mask] = np.nan

        ax = self.fig.add_subplot(122)
        ax.scatter(r, d_pot, marker='.', color='green', s=2, label="Current method")
        ax.axvline(x=r_brillouin, color='red', linestyle='--', label='R_brillouin')
        ax.set_xlabel(r"$r(\text{km})$")
        ax.set_ylabel("Relative Error (%)")

        valid_r = r[valid_mask]
        if len(valid_r) > 0:
            ax.set_xlim(valid_r.min(), valid_r.max())
        ax.set_ylim(-2e-2, 2e-2)
        ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.15), fancybox=True, shadow=True, ncol=2)
        ax.set_title("Relative Error vs Classical Polyhedral")

        self.canvas.draw()
