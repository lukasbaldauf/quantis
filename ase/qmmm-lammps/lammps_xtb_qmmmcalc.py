import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ctypes import c_double
from lammps import lammps
from ase.calculators.qmmm import SimpleQMMM
from xtb.ase.calculator import XTB
import pathlib

class LAMMPSASECalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, lammps_inp):
        super().__init__()
        self.lammps_inp = lammps_inp # Run this file when starting up LAMMPS
        self.lmp = None  # LAMMPS instance placeholder
        self.initialized = False

    def initialize_lammps(self, atoms):
        """Initialize the LAMMPS instance and set up the system"""
        cnt = 0
        uselog = False
        while cnt < 100 and not uselog:
            log = pathlib.Path(f"log.lammps{cnt:03d}")
            if not log.exists():
                uselog = str(log)
                print(f"Using log {uselog}")
            cnt+=1
        self.lmp = lammps(cmdargs = ["-screen", "none", "-log", uselog])  # Start a new LAMMPS instance
        self.lmp.file(self.lammps_inp)

        self.natoms = self.lmp.get_natoms()
        self.n3 = self.natoms*3

        # Set the atomic positions in LAMMPS
        self.update_positions(atoms)

        # Flag as initialized
        self.initialized = True

    def update_positions(self, atoms):
        """Update atomic positions in LAMMPS based on ASE atoms"""
        pos = atoms.positions
        x = (c_double*self.n3)(*pos.flatten())
        self.lmp.scatter_atoms("x", 1, 3, x)

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        if not self.initialized:
            self.initialize_lammps(atoms)

        atoms.wrap()
        self.update_positions(atoms)

        self.lmp.command("run 0 post no")

        energy = self.lmp.get_thermo("pe")

        natoms = len(atoms)
        forces = np.array(self.lmp.gather_atoms("f", 1, 3)).reshape(natoms, 3)

        self.results = {"energy": energy*0.0433641, "forces": forces*0.0433641}

    def close(self):
        """Close the LAMMPS instance"""
        if self.lmp is not None:
            self.lmp.close()

class QMMMCalc(Calculator):
    def __init__(self):
        super().__init__()
        self.qmcalc = XTB(method="GFN2-xTB")
        self.mmcalc1 = LAMMPSASECalculator("lammps_input/lammps_vac.input")
        self.mmcalc2 = LAMMPSASECalculator("lammps_input/lammps.input")
        self.qmmmcalc = SimpleQMMM([i for i in range(22)], self.qmcalc, self.mmcalc1, self.mmcalc2)
        self.implemented_properties = ["energy","forces"]

    def calculate(self, atoms = None, properties = None, system_changes = all_changes):
        self.qmmmcalc.calculate(atoms, properties, system_changes)
        self.results["energy"] = self.qmmmcalc.results["energy"]
        self.results["forces"] = self.qmmmcalc.results["forces"]
        self.results["stress"] = np.zeros(6)
