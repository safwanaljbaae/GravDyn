#!/usr/bin/env python3
"""
Potential Tab - Compute and visualize gravitational potential.
"""
import re
import os
import requests
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jax.numpy as jnp
import trimesh
from scipy.spatial.distance import cdist
import tkinter as tk
from pathlib import Path
from matplotlib.figure import Figure
from tkinter import ttk, messagebox, filedialog
from gravdyn.shape_tools import load_vertices, load_faces
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from gravdyn.prepare_polyhedral_model import prepare_werner_model
from gravdyn.build_potential_derivatives import build_potential_derivatives
from gravdyn.generate_layered_mascons import generate_layered_mascons
from gravdyn.generate_layered_mascons import load_tetrahedron_data
from gravdyn.pot_functions import (pot_expansion,
                                   batched_wrener_potential,
                                   compute_pseudo_potential,
                                   batched_pot_mascon)

matplotlib.use('TkAgg')

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
        
        self._create_widgets()

    def _extract_mass_density(self, file_path):
        mass = None
        density = None

        with open(file_path, 'r') as f:
            for line in f:
                # Extract mass
                if "the considered mass is" in line:
                    mass_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                    if mass_match:
                        mass = float(mass_match.group())

                # Extract density
                if "the considered density is" in line:
                    density_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                    if density_match:
                        density = float(density_match.group())

        return mass, density

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
        
        ttk.Radiobutton(
            set_frame, text="XY", variable=self.plane_type,
            value="xy"
        ).pack(anchor='w', padx=15)
        
        ttk.Radiobutton(
            set_frame, text="YZ", variable=self.plane_type,
            value="yz"
        ).pack(anchor='w', padx=15)
        
        ttk.Radiobutton(
            set_frame, text="XZ", variable=self.plane_type,
            value="xz"
        ).pack(anchor='w', padx=15)
        
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
        
        self.potential_data = None
        self.asteroid_dir = None
        
    def _load_asteroid_list(self):
        if not hasattr(self, '_asteroid_list'):
            self._asteroid_list = []
        
        try:
            url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/Data?ref={BRANCH}"
            response = requests.get(url, timeout=20, headers=_get_headers())
            # print(f"GitHub API response status: {response.status_code}")
            if response.status_code == 200:
                contents = response.json()
                self._asteroid_list = [
                    item['name'] for item in contents 
                    if item['type'] == 'dir' and not item['name'].startswith('.')
                ]
                # print(f"Loaded {len(self._asteroid_list)} asteroids: {self._asteroid_list}")
                self.asteroid_combo['values'] = self._asteroid_list
            else:
                self._load_local_asteroids()
        except Exception as e:
            print(f"Error loading asteroid list: {e}")
            self._load_local_asteroids()
            
    def _load_local_asteroids(self):
        local_data_dir = Path("Data")
        if local_data_dir.exists():
            self._asteroid_list = [
                d.name for d in local_data_dir.iterdir() 
                if d.is_dir() and not d.name.startswith('.')
            ]
            # print(f"Loaded local asteroids: {self._asteroid_list}")
            self.asteroid_combo['values'] = self._asteroid_list
            
    def _log(self, message):
        if hasattr(self.main_app, '_log'):
            self.main_app._log(message)
    
    def _on_asteroid_selected(self, event=None):
        asteroid_name = self.asteroid_var.get()
        if asteroid_name:
            self._log(f"Selected: {asteroid_name}")
            self._load_from_github(asteroid_name)
            
    def _toggle_layers_entry(self):
        if self.model_type.get() == "mascon":
            self.layers_entry.config(state='normal')
        else:
            self.layers_entry.config(state='disabled')

    def _download_folder_recursive(self, api_url, local_dir):
        contents = requests.get(api_url, timeout=30, headers=_get_headers()).json()
        for item in contents:
            if item['type'] == 'file':
                file_response = requests.get(item['download_url'], timeout=30, headers=_get_headers())
                if file_response.status_code == 200:
                    file_path = local_dir / item['name']
                    with open(file_path, 'wb') as f:
                        f.write(file_response.content)
                    self._log(f"Downloaded: {item['path']}")
            elif item['type'] == 'dir':
                sub_dir = local_dir / item['name']
                sub_dir.mkdir(parents=True, exist_ok=True)
                self._download_folder_recursive(item['url'], sub_dir)

    def _load_from_github(self, asteroid_name):
        self._log(f"Loading {asteroid_name} folder from GitHub...")
        
        gui_dir = Path(__file__).parent
        data_dir = gui_dir / "Data"
        asteroid_dir = data_dir / asteroid_name
        asteroid_dir.mkdir(parents=True, exist_ok=True)

        api_url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/Data/{asteroid_name}"

        try:
            response = requests.get(api_url, timeout=30, headers=_get_headers())
            
            if response.status_code == 200:
                self._download_folder_recursive(api_url, asteroid_dir)
                
                vertices_path = asteroid_dir / "modified_v.dat"
                faces_path = asteroid_dir / "modified_f.dat"
                
                if vertices_path.exists() and faces_path.exists():
                    vertices = np.loadtxt(vertices_path)
                    faces = np.loadtxt(faces_path, dtype=int)
                    
                    self.asteroid_dir = asteroid_dir
                    self._current_vertices = vertices
                    self._current_faces = faces
                    self._asteroid_name = asteroid_name
                    self._shape_source = 'asteroid'
                    
                    self.status_label.config(text=f"Loaded: {asteroid_name}")
                    self._log(f"Loaded shape for {asteroid_name}")
                else:
                    self._log(f"Modified shape files not found in {asteroid_name}")
                    self.status_label.config(text="Shape files not found")
            else:
                self._log(f"Failed to fetch folder contents: {response.status_code}")
                self.status_label.config(text="Folder not found on GitHub")
                
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            self._log(f"Error: {str(e)}")
            
    def _check_custom_name(self):
        name = self.custom_name_var.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter an asteroid name")
            return
        self._load_asteroid_by_name(name)
        
    def _load_asteroid_by_name(self, name):
        self.status_label.config(text=f"Loading {name} from GitHub...", foreground="blue")
        
        gui_dir = Path(__file__).parent
        data_dir = gui_dir / "data"
        asteroid_dir = data_dir / name
        
        if not asteroid_dir.exists():
            self._log(f"Downloading asteroid data from GitHub...")
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
                
                self._log(f"Downloaded {len(files)} files for {name}")
                
            except Exception as e:
                self.status_label.config(text=f"Error: {str(e)}")
                self._log(f"Error: {str(e)}")
                return
        
        v_path = asteroid_dir / "modified_v.dat"
        f_path = asteroid_dir / "modified_f.dat"
        

        if not v_path.exists():
            self.status_label.config(text="Shape files not found")
            return
        
        try:
            vertices = load_vertices(str(v_path))
            faces = load_faces(str(f_path))
            
            self.asteroid_dir = asteroid_dir
            self._current_vertices = vertices
            self._current_faces = faces
            self._asteroid_name = name
            self._shape_source = 'asteroid'
            
            self.status_label.config(text=f"Loaded: {name}")
            self._log(f"Loaded shape for {name}")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            self._log(f"Error: {str(e)}")
    
    def _browse_vertices(self):
        filename = filedialog.askopenfilename(
            title="Select vertices file",
            filetypes=[("Data files", "*.dat"), ("All files", "*.*")]
        )
        if filename:
            self.vertices_file_var.set(filename)
            
    def _browse_faces(self):
        filename = filedialog.askopenfilename(
            title="Select faces file",
            filetypes=[("Data files", "*.dat"), ("All files", "*.*")]
        )
        if filename:
            self.faces_file_var.set(filename)
            
    def _load_shape(self):
        vertices_path = self.vertices_file_var.get().strip()
        faces_path = self.faces_file_var.get().strip()
        
        if not vertices_path or not faces_path:
            messagebox.showwarning("Warning", "Please select both vertices and faces files")
            return
        
        try:
            vertices = load_vertices(vertices_path)
            faces = load_faces(faces_path)
            
            asteroid_name = Path(vertices_path).stem.replace('_v', '').replace('input_v', '').strip()
            if not asteroid_name:
                asteroid_name = "custom"
            
            gui_dir = Path(__file__).parent
            data_dir = gui_dir / "data"
            asteroid_dir = data_dir / asteroid_name
            asteroid_dir.mkdir(parents=True, exist_ok=True)
            
            np.savetxt(asteroid_dir / "modified_v.dat", vertices, fmt="%.10e")
            np.savetxt(asteroid_dir / "modified_f.dat", faces, fmt="%d")
            
            self.asteroid_dir = asteroid_dir
            self._current_vertices = vertices
            self._current_faces = faces
            self._asteroid_name = asteroid_name
            self._shape_source = 'files'
            
            self.status_label.config(text=f"Loaded: {asteroid_name}")
            self._log(f"Loaded shape for {asteroid_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load shape: {str(e)}")
    
    def _load_from_main_app(self):
        if hasattr(self, '_asteroid_name') and self._asteroid_name is not None:
            return
        
        if hasattr(self, '_current_vertices') and self._current_vertices is not None:
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
            self._load_asteroid_by_name(custom_name)
            
    def _compute_potential(self):
        self._load_from_main_app()
        
        source = getattr(self, '_shape_source', None)
        
        has_asteroid = source == 'asteroid'
        has_files = source == 'files'

        if not has_asteroid and not has_files:
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
                points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n*n)])
                plane_slice = (X, Y, np.zeros_like(X))
            elif plane == "yz":
                y = np.linspace(-rmax, rmax, n)
                z = np.linspace(-rmax, rmax, n)
                Y, Z = np.meshgrid(y, z)
                points = np.column_stack([np.zeros(n*n), Y.ravel(), Z.ravel()])
                plane_slice = (np.zeros_like(Y), Y, Z)
            else:
                x = np.linspace(-rmax, rmax, n)
                z = np.linspace(-rmax, rmax, n)
                X, Z = np.meshgrid(x, z)
                points = np.column_stack([X.ravel(), np.zeros(n*n), Z.ravel()])
                plane_slice = (X, np.zeros_like(X), Z)
            
            dist_from_center = np.linalg.norm(points, axis=1)
            
            if hasattr(self, '_current_vertices') and self._current_vertices is not None:
                r_brillouin = np.linalg.norm(self._current_vertices, axis=1).min()
            else:
                r_brillouin = rmax * 0.8
            
            inside_mask = dist_from_center < r_brillouin
            
            self._log(f"Computing potential at {n*n} grid points...")
            
            if model == "polyhedral":
                p, acc = self._compute_polyhedral(asteroid_name, points)
                p = np.array(p, copy=True)
            elif model == "mascon":
                nlayers = int(self.layers_var.get())
                p, acc = self._compute_mascon(asteroid_name, points, nlayers)
                p = np.array(p, copy=True)
            elif model == "expansion":
                p, acc = self._compute_expansion(asteroid_name, points)
                p = np.array(p, copy=True)
            else:
                return 'Error'

            p.ravel()[inside_mask] = np.nan
            p = p.reshape(n, n)
            
            rot_period_hours = float(self.rot_period_var.get())
            pseudo_p = compute_pseudo_potential(points, p.ravel(), rot_period_hours)
            pseudo_p = np.array(pseudo_p, copy=True)
            pseudo_p.ravel()[inside_mask] = np.nan
            pseudo_p = pseudo_p.reshape(n, n)
            # print(f"Rotational Period: {rot_period_hours} hours")
            
            self.potential_data = {
                'U': p,
                'pseudo': pseudo_p,
                'plane': plane,
                'model': model,
                'points': points,
                'plane_slice': plane_slice
            }
            
            save_dir = self.asteroid_dir
            if save_dir:
                data_to_save = np.column_stack([points, p.ravel(), pseudo_p.ravel()])
                np.savetxt(save_dir / f"potential_{model}_{plane}.dat", data_to_save, fmt="%.10e", header="x y z potential pseudo_potential")
                self._log(f"Saved to {save_dir / f'potential_{model}_{plane}.dat'}")
            
            self._plot_zvc()
            
            if model in ["expansion", "mascon"]:
                self.compare_pot()
            
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
        mass, density = self._extract_mass_density(file_path)

        gravitation = 6.674101262875753845e-20
        gm_body = mass * gravitation
        batch_size = 2000
        p, acc = batched_wrener_potential(gm_body=gm_body,
                                          stat=points,
                                          polyhedral_data=polyhedral_data,
                                          batch_size=batch_size)
        return p, acc

    def _compute_mascon(self, asteroid_name, points, nlayers):

        print('Mascon')
        file_path = f"Data/{asteroid_name}/shape_verification.log"
        mass, density = self._extract_mass_density(file_path)

        print()
        df_mascons = generate_layered_mascons(
            base_dir='Data/',
            asteroid=asteroid_name,
            total_mass=mass,
            densities=[density] * nlayers,
            output_csv="layered_mascons.csv",
        )
        data_shape = load_tetrahedron_data(
            asteroid=asteroid_name,
            base_dir="Data",
            tetrahedron_data_file="layered_mascons.csv",
        )

        p, acc = batched_pot_mascon(points, data_shape, batch_size=20000)
        return p, acc
        
    def _compute_expansion(self, asteroid_name, points):
        print('Expansion')

        file_path = f"Data/{asteroid_name}/shape_verification.log"
        mass, density = self._extract_mass_density(file_path)

        gravitation = 6.674101262875753845e-20
        gm_body = mass * gravitation

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
        return  p, acc
        
    def _plot_zvc(self):
        if self.potential_data is None:
            return
        
        self.fig.clear()

        pseudo = self.potential_data.get('pseudo')
        plane = self.potential_data.get('plane', 'xy')
        X, Y, Z = self.potential_data['plane_slice']
        model = self.potential_data.get('model', 'polyhedral')

        if plane == "xy":
            x_coord, y_coord = X, Y
            xlabel, ylabel = 'X', 'Y'
            plane_label = 'XY'
        elif plane == "yz":
            x_coord, y_coord = Y, Z
            xlabel, ylabel = 'Y', 'Z'
            plane_label = 'YZ'
        else:
            x_coord, y_coord = X, Z
            xlabel, ylabel = 'X', 'Z'
            plane_label = 'XZ'

        if model in ["expansion", "mascon"]:
            ax = self.fig.add_subplot(121)
            ax_zvc = ax
        else:
            ax = self.fig.add_subplot(111)
            ax_zvc = ax

        pseudo_2d = pseudo

        if pseudo_2d is not None:
            valid_mask = ~np.isnan(pseudo_2d)
            if valid_mask.any():
                valid_pseudo = np.where(valid_mask, pseudo_2d, np.nan)
                pseudo_min = np.nanmin(valid_pseudo)
                pseudo_max = np.nanmax(valid_pseudo)
                levels = np.linspace(pseudo_min, pseudo_max, 2500)
                ax_zvc.contour(
                    x_coord, y_coord, pseudo_2d,
                    levels=levels,
                    colors='royalblue',
                    linewidths=1.2,
                    linestyles='solid',
                    alpha=0.8
                )

        if hasattr(self, '_current_vertices') and self._current_vertices is not None:
            vertices = np.array(self._current_vertices)
            faces = np.array(self._current_faces)
            
            if plane == "xy":
                v_proj = vertices[:, :2]
            elif plane == "yz":
                v_proj = vertices[:, 1:3]
            else:
                v_proj = vertices[:, [0, 2]]
            
            for face in faces:
                pts = v_proj[face]
                ax_zvc.plot(pts[:, 0], pts[:, 1], 'k-', linewidth=0.5)
        
        ax_zvc.set_xlabel(xlabel)
        ax_zvc.set_ylabel(ylabel)
        ax_zvc.set_title(f'Zero Velocity Curves - {model.title()}')
        ax_zvc.set_aspect('equal')
        self.canvas.draw()

    def _load_classical_potential(self, plane):
        if not self.asteroid_dir:
            return None
        
        classical_file = self.asteroid_dir / f"potential_polyhedral_{plane}.dat"
        if not classical_file.exists():
            return None
        
        try:
            data = np.loadtxt(classical_file)
            return data
        except Exception:
            return None

    def _plot_relative_error(self, model, plane, show_in_window=True):
        classical_data = self._load_classical_potential(plane)
        
        if classical_data is None:
            messagebox.showwarning(
                "Missing Reference Data",
                "Classical Polyhedral potential data not found.\n"
                "Please run the Classical Polyhedral method first to compute the reference potential."
            )
            return
        
        classical_potential = classical_data[:, 3]
        current_potential = self.potential_data['U'].ravel()
        points = self.potential_data['points']
        
        dist_from_center = np.linalg.norm(points, axis=1)
        if hasattr(self, '_current_vertices') and self._current_vertices is not None:
            r_brillouin = np.linalg.norm(self._current_vertices, axis=1).min()
        else:
            r_brillouin = float(self.radius_var.get()) * 0.8
        
        inside_mask = dist_from_center < r_brillouin
        
        valid_mask = ~np.isnan(current_potential) & ~np.isnan(classical_potential) & ~inside_mask
        valid_mask = valid_mask & (np.abs(classical_potential) > 1e-20)
        
        relative_error = np.zeros_like(current_potential)
        relative_error[valid_mask] = np.abs((current_potential[valid_mask] - classical_potential[valid_mask]) / classical_potential[valid_mask])
        relative_error[~valid_mask] = np.nan
        
        n = int(np.sqrt(len(relative_error)))
        relative_error_2d = relative_error.reshape(n, n)
        
        X, Y, Z = self.potential_data['plane_slice']
        
        if plane == "xy":
            x_coord, y_coord = X, Y
        elif plane == "yz":
            x_coord, y_coord = Y, Z
        else:
            x_coord, y_coord = X, Z
        
        if show_in_window:
            fig_error = Figure(figsize=(7, 6))
            ax_error = fig_error.add_subplot(111)
            
            valid_error = ~np.isnan(relative_error_2d)
            if valid_error.any():
                error_min = np.nanmin(relative_error_2d[valid_error])
                error_max = np.nanmax(relative_error_2d[valid_error])
                
                levels = np.linspace(error_min, error_max, 50)
                
                contour = ax_error.contourf(
                    x_coord, y_coord, relative_error_2d,
                    levels=levels,
                    cmap='hot_r',
                    extend='both'
                )
                fig_error.colorbar(contour, ax=ax_error, label='Relative Error')
            
            ax_error.set_title(f'Relative Error (Potential) vs Classical Polyhedral\n{model.title()} Method ({plane.upper()} plane)')
            ax_error.set_aspect('equal')
            
            error_window = tk.Toplevel(self)
            error_window.title(f"Relative Error - {model.title()}")
            error_window.geometry("700x650")
            
            canvas_error = FigureCanvasTkAgg(fig_error, error_window)
            canvas_error.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ax_error = self.fig.add_subplot(122)
            
            valid_error = ~np.isnan(relative_error_2d)
            if valid_error.any():
                error_min = np.nanmin(relative_error_2d[valid_error])
                error_max = np.nanmax(relative_error_2d[valid_error])
                
                levels = np.linspace(error_min, error_max, 50)
                
                contour = ax_error.contourf(
                    x_coord, y_coord, relative_error_2d,
                    levels=levels,
                    cmap='hot_r',
                    extend='both'
                )
                self.fig.colorbar(contour, ax=ax_error, label='Relative Error')
            
            ax_error.set_title(f'Relative Error (Potential)')
            ax_error.set_aspect('equal')
            
            self.canvas.draw()

    def compare_pot(self, asteroid=None):
        if self.potential_data is None:
            return

        classical_data = self._load_classical_potential(self.potential_data.get('plane', 'xy'))
        
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
        if hasattr(self, '_current_vertices') and self._current_vertices is not None:
            r_brillouin = np.linalg.norm(self._current_vertices, axis=1).min()
        else:
            r_brillouin = float(self.radius_var.get()) * 0.8
        
        inside_mask = dist_from_center < r_brillouin
        
        r = dist_from_center
        valid_mask = ~np.isnan(current_potential) & ~np.isnan(classical_potential) & ~inside_mask
        valid_mask = valid_mask & (np.abs(classical_potential) > 1e-20)
        
        d_pot = np.zeros_like(current_potential)
        d_pot[valid_mask] = (current_potential[valid_mask] - classical_potential[valid_mask]) * 100 / classical_potential[valid_mask]
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