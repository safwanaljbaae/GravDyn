from gravdyn.polyhedral_model.poly_files import PolyFiles


class TestPolyFiles:
    def test_default_values(self):
        pf = PolyFiles()
        assert pf.base_dir == "DATA"
        assert pf.asteroid == "BENNU"

    def test_custom_values(self):
        pf = PolyFiles(base_dir="/my/data", asteroid="Apophis")
        assert pf.base_dir == "/my/data"
        assert pf.asteroid == "Apophis"

    def test_root_path(self):
        pf = PolyFiles(base_dir="/data", asteroid="Eros")
        assert pf.root == "/data/Eros"

    def test_file_vertices(self):
        pf = PolyFiles(base_dir="/d", asteroid="A")
        assert pf.file_vertices == "/d/A/modified_v.dat"

    def test_file_faces(self):
        pf = PolyFiles(base_dir="/d", asteroid="A")
        assert pf.file_faces == "/d/A/modified_f.dat"

    def test_file_edges(self):
        pf = PolyFiles(base_dir="/d", asteroid="A")
        assert pf.file_edges == "/d/A/edges.dat"

    def test_file_centroid_faces(self):
        pf = PolyFiles(base_dir="/d", asteroid="A")
        assert pf.file_centroid_faces == "/d/A/centroid_faces.dat"

    def test_file_centroid_edges(self):
        pf = PolyFiles(base_dir="/d", asteroid="A")
        assert pf.file_centroid_edges == "/d/A/centroid_edges.dat"

    def test_file_e_e(self):
        pf = PolyFiles(base_dir="/d", asteroid="A")
        assert pf.file_e_e == "/d/A/e_e.dat"

    def test_file_n_f(self):
        pf = PolyFiles(base_dir="/d", asteroid="A")
        assert pf.file_n_f == "/d/A/n_f.dat"

    def test_file_n_f_e(self):
        pf = PolyFiles(base_dir="/d", asteroid="A")
        assert pf.file_n_f_e == "/d/A/n_f_e.dat"

    def test_file_n_fp_e(self):
        pf = PolyFiles(base_dir="/d", asteroid="A")
        assert pf.file_n_fp_e == "/d/A/n_fp_e.dat"

    def test_file_r_e_1(self):
        pf = PolyFiles(base_dir="/d", asteroid="A")
        assert pf.file_r_e_1 == "/d/A/r_e_1.dat"

    def test_file_r_e_2(self):
        pf = PolyFiles(base_dir="/d", asteroid="A")
        assert pf.file_r_e_2 == "/d/A/r_e_2.dat"

    def test_file_r_f_1(self):
        pf = PolyFiles(base_dir="/d", asteroid="A")
        assert pf.file_r_f_1 == "/d/A/r_f_1.dat"

    def test_file_r_f_2(self):
        pf = PolyFiles(base_dir="/d", asteroid="A")
        assert pf.file_r_f_2 == "/d/A/r_f_2.dat"

    def test_file_r_f_3(self):
        pf = PolyFiles(base_dir="/d", asteroid="A")
        assert pf.file_r_f_3 == "/d/A/r_f_3.dat"

    def test_all_file_properties_return_strings(self):
        pf = PolyFiles()
        for attr in ["file_vertices", "file_faces", "file_edges",
                      "file_centroid_faces", "file_centroid_edges",
                      "file_e_e", "file_n_f", "file_n_f_e", "file_n_fp_e",
                      "file_r_e_1", "file_r_e_2", "file_r_f_1", "file_r_f_2", "file_r_f_3"]:
            val = getattr(pf, attr)
            assert isinstance(val, str), f"{attr} should be str, got {type(val)}"
