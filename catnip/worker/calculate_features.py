import os
import uuid

import autode
from morfeus import SASA, XTB, Dispersion

from rdkit import Chem
from dimorphite_dl import DimorphiteDL

import json

from rq import get_current_job


class Substrate:
    def __init__(self, smiles, job=None):
        self.smiles = smiles.split(' ')[0]
        self.smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.smiles))

        self.ph_corrected_smiles = self.correct_ph(self.smiles, 7.5)

        if job:
            job.meta["ph_corrected_smiles"] = self.ph_corrected_smiles
            job.save_meta()

        self.config = {
            "n_cores": 1,
            "method": "xTB",
        }

        self.job = job

        self.features = {}

    def correct_ph(self, smiles, ph):
        dimorphite_dl = DimorphiteDL(
            min_ph=ph,
            max_ph=ph,
            max_variants=1,
            label_states=False,
            pka_precision=1.0
        )

        mol = dimorphite_dl.protonate(smiles)[0]
        return mol

    def get_geometry_and_basic_features(self):
        self._setup_config()
        opt_method = self._get_optimization_method()
        mol = self._optimize_geometry(opt_method)
        self._cleanup_temp_files(mol.name)
        self._extract_geometry_and_features(mol)

    def _setup_config(self):
        autode.Config.n_cores = self.config["n_cores"]

    def _get_optimization_method(self):
        method_mapping = {
            "xTB": autode.methods.XTB
        }
        try:
            return method_mapping[self.config["method"]]()
        except KeyError:
            raise ValueError(f"Method '{self.config['method']}' not supported")

    def _get_current_geometry(self, mol, name):
        autode.input_output.atoms_to_xyz_file(mol.atoms, f"{name}.xyz")
        os.system(f"obabel -ixyz {name}.xyz -osdf -O {name}.sdf")
        with open(f"{name}.sdf", "r") as f:
            return f.read()

    def _optimize_geometry(self, opt_method):
        temp_name = uuid.uuid4().hex
        mol = autode.Molecule(smiles=self.ph_corrected_smiles, name=temp_name)

        if self.job:
            self.job.meta["unoptimized_sdf"] = self._get_current_geometry(mol, f"{temp_name}_unoptimized")
            self.job.save_meta()

        mol.optimise(method=opt_method)

        if self.job:
            self.job.meta["optimized_sdf"] = self._get_current_geometry(mol, f"{temp_name}_optimized")
            self.job.save_meta()

        return mol

    def _cleanup_temp_files(self, temp_name):
        files_to_remove = [
            f"{temp_name}_optimised_xtb.xyz",
            f"{temp_name}_opt_xtb.xyz",
            f"{temp_name}_opt_xtb.out",
            "xtbopt.xyz",
            ".autode_calculations",
            f"{temp_name}_unoptimized.xyz",
            f"{temp_name}_unoptimized.sdf",
            f"{temp_name}_optimized.xyz"
            f"{temp_name}_optimized.sdf"
        ]
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)

    def _extract_geometry_and_features(self, mol):
        self.elements = mol.atomic_symbols
        self.coordinates = mol.coordinates.tolist()
        self.features.update({
            "mass": mol.mass.real,
            "radius": mol.radius.real,
            "charge": mol.charge
        })

    def calculate_features(self):
        if not (hasattr(self, "elements") and hasattr(self, "coordinates")):
            raise ValueError("Geometry not calculated")

        self._set_sasa_features()
        self._set_dispersion_features()
        self._set_xtb_features()

    def _synchronize_features(self):
        if self.job:
            self.job.meta["features"] = json.dumps(self.features)
            self.job.save_meta()

    def _set_sasa_features(self):
        sasa = SASA(self.elements, self.coordinates)
        self.features["sasa_area"] = sasa.area
        self.features["sasa_volume"] = sasa.volume

    def _set_dispersion_features(self):
        disp = Dispersion(self.elements, self.coordinates)
        disp.compute_coefficients()
        disp.compute_p_int()
        self.features.update({
            "disp_area": disp.area,
            "disp_volume": disp.volume,
            "disp_p_int": disp.p_int,
            "disp_p_max": disp.p_max,
            "disp_p_min": disp.p_min
        })
        self._synchronize_features()

    def _set_xtb_features(self):
        xtb = XTB(self.elements, self.coordinates)
        self.features.update({
            "ip": xtb.get_ip(),
            "ea": xtb.get_ea(),
            "homo": xtb.get_homo(),
            "lumo": xtb.get_lumo(),
        })
        self._synchronize_features()

        descriptors = ["electrophilicity", "nucleophilicity", "electrofugality", "nucleofugality"]
        for desc in descriptors:
            self.features[desc] = xtb.get_global_descriptor(desc, corrected=True)

        self._synchronize_features()

        dipole = xtb.get_dipole()
        axes = ["x", "y", "z"]
        for i, axis in enumerate(axes):
            self.features[f"dipole_{axis}"] = dipole[i]

        self._synchronize_features()


def run_calculations(smiles):
    job = get_current_job()

    print("Running:", smiles)

    substrate = Substrate(smiles, job)
    print("Created substrate")

    substrate.get_geometry_and_basic_features()
    print("Got geometry and basic features")

    substrate.calculate_features()
    print("Calculated features")

    return json.dumps(substrate.features)
