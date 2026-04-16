# -*- coding: utf-8 -*-
"""
# !===============================================================
# !==   Dr. Safwan Aljbaae                                      ==
# !==   October 2025                                            ==
# !===============================================================
# python3 -m pip install -r requirements.txt                    ==
# !===============================================================
"""
import os
from dataclasses import dataclass

@dataclass
class PolyFiles:
    base_dir: str = "DATA"
    asteroid: str = "BENNU"

    @property
    def root(self) -> str:
        return os.path.join(self.base_dir, self.asteroid)

    @property
    def file_vertices(self) -> str:
        return os.path.join(self.root, "modified_v.dat")

    @property
    def file_faces(self) -> str:
        return os.path.join(self.root, "modified_f.dat")

    @property
    def file_edges(self) -> str:
        return os.path.join(self.root, "edges.dat")

    @property
    def file_centroid_faces(self) -> str:
        return os.path.join(self.root, "centroid_faces.dat")

    @property
    def file_centroid_edges(self) -> str:
        return os.path.join(self.root, "centroid_edges.dat")
    
    @property
    def file_e_e(self) -> str:
        return os.path.join(self.root, "e_e.dat")
    
    @property
    def file_n_f(self) -> str:
        return os.path.join(self.root, "n_f.dat")
    
    @property
    def file_n_f_e(self) -> str:
        return os.path.join(self.root, "n_f_e.dat")
    
    @property
    def file_n_fp_e(self) -> str:       
        return os.path.join(self.root, "n_fp_e.dat")
    
    @property
    def file_r_e_1(self) -> str:
        return os.path.join(self.root, "r_e_1.dat")
    
    @property
    def file_r_e_2(self) -> str:
        return os.path.join(self.root, "r_e_2.dat")
    
    @property
    def file_r_f_3(self) -> str:
        return os.path.join(self.root, "r_f_3.dat")
    
    @property
    def file_r_f_1(self) -> str:
        return os.path.join(self.root, "r_f_1.dat") 
    
    @property
    def file_r_f_2(self) -> str:
        return os.path.join(self.root, "r_f_2.dat") 
    
    
