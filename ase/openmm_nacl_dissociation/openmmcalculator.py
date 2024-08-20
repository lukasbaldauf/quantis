import openmm
import openmm.app
import numpy as np

from ase.calculators.calculator import Calculator, all_changes

class OpenMMCalculator(Calculator):
    """
    Minimalistic OpenMM calculator.

    To control the number of cpus, run

        export OPENMM_NUM_THREADS=1
    """
    def __init__(self, simulation):
        super().__init__()
        self.context = simulation.context
        self.implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms, properties = None, system_changes = all_changes):
        self.context.setPositions(atoms.positions/10)
        self.context.setPeriodicBoxVectors(*atoms.cell.array/10)
        state = self.context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy()._value
        forces = state.getForces(asNumpy=True)._value

        self.results["energy"] = energy * 0.0103641
        self.results["forces"] = forces * 0.00103641
        self.results["stress"] = np.zeros(6)

def get_simulation(system_xml, topology_pdb):
    """
    Generate an OpenMM simulation from a system.xml and topology.pdb file.
    """

    with open(system_xml) as rfile:
        system = openmm.XmlSerializer.deserialize(rfile.read())

    pdb = openmm.app.PDBFile(topology_pdb)
    topology = pdb.topology
    integrator = openmm.VerletIntegrator(0.5)
    simulation = openmm.app.Simulation(
            topology, system, integrator)
    return simulation

class Calculator0(OpenMMCalculator):
    """
    Same as system1 but all water parameters are scaled by 1.1
    """
    def __init__(self):
        self.xml_file = "openmm_input/system0.xml"
        self.pdb_file = "openmm_input/topology.pdb"
        super().__init__(get_simulation(self.xml_file, self.pdb_file))


class Calculator1(OpenMMCalculator):
    def __init__(self):
        self.xml_file = "openmm_input/system1.xml"
        self.pdb_file = "openmm_input/topology.pdb"
        super().__init__(get_simulation(self.xml_file, self.pdb_file))

class DPSwitching:
    def __init__(self, l0, l_1):
        self.l0 = l0
        self.l_1 = l_1

    def mix(self, x):
        s = (x - self.l_1)/(self.l0 - self.l_1)
        return (s**3*(-6*s**2 + 15*s - 10) + 1)

class ForceMixingCalc(Calculator):
    """
    Mix two calculators.
    """
    def __init__(self):
        super().__init__()
        self.implemented_properties = ["energy", "forces", "stress"]
        self.mm_calc = Calculator0()
        self.dft_calc = Calculator1()
        self.intf0 = 2.8
        self.intf_1 = 2.7
        self.Forcemixing = DPSwitching(self.intf0, self.intf_1)



    def calculate(self, atoms, properties = None, system_changes = all_changes):
        # we need to calculate the order parameter at every step
        order = atoms.calculate_order(atoms.system, xyz=atoms.positions, vel=atoms.get_velocities(), box=atoms.cell.diagonal(),)[0]

        if order < self.intf_1:
            self.mm_calc.calculate(atoms)
            e = self.mm_calc.results["energy"]
            f = self.mm_calc.results["forces"]
            rho = 1.0

        elif order < self.intf0:
            self.mm_calc.calculate(atoms)
            self.dft_calc.calculate(atoms)

            e0 = self.mm_calc.results["energy"]
            f0 = self.mm_calc.results["forces"]
            e1 = self.dft_calc.results["energy"]
            f1 = self.dft_calc.results["forces"]

            rho = self.Forcemixing.mix(order)
            e = e1
            f = rho * f0 + (1 - rho) * f1

        else:
            self.dft_calc.calculate(atoms)
            e = self.dft_calc.results["energy"]
            f = self.dft_calc.results["forces"]
            rho = 0.0

        self.results = {"energy":e, "forces":f, "stress":np.zeros(6)}