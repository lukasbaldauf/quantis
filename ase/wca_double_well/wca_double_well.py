from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
import numpy as np
from ase.calculators import cp2k
from pathlib import Path
from copy import deepcopy
from numba import njit

#import deepmd
#from deepmd.infer.model_devi import calc_model_devi_efv
#from deepmd import DeepPotential as DP

@njit
def distance_matrix(positions, box_length):
    num_particles = positions.shape[0]
    dist_matrix = np.zeros((num_particles, num_particles, 3))
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            delta = positions[i] - positions[j]
            for k in range(3):
                delta[k] -= box_length * np.round(delta[k] / box_length)
            dist_matrix[i, j] = delta
            dist_matrix[j, i] = -delta
    return dist_matrix

@njit
def compute_distances(dist_matrix):
    num_particles = dist_matrix.shape[0]
    distances = np.zeros((num_particles, num_particles))
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            distances[i, j] = np.linalg.norm(dist_matrix[i, j])
            distances[j, i] = distances[i, j]
    return distances

@njit
def wca_potential(r, epsilon, sigma):
    r_cut = 2**(1/6) * sigma
    if r < r_cut:
        r6 = (sigma / r)**6
        r12 = r6 * r6
        return 4 * epsilon * (r12 - r6) + epsilon
    else:
        return 0.0

@njit
def wca_force(r, epsilon, sigma):
    r_cut = 2**(1/6) * sigma
    if r < r_cut:
        r6 = (sigma / r)**6
        r12 = r6 * r6
        force = 24 * epsilon * (2 * r12 - r6) / r
        return force
    else:
        return 0.0

@njit
def double_well_potential(r, a, b, c, shift):
    r = r - shift
    return a * r**4 - b * (r - c)**2

@njit
def double_well_force(r, a, b, c, shift):
    r = r - shift
    return -4 * a * r **3 + 2 * b * (r - c)

@njit
def harmonic_potential(r, k, r0):
    return 1/2 * k * (r - r0)**2

@njit
def harmonic_force(r, k, r0):
    return -k * (r - r0)

@njit
def forces_and_energy_DW(dist_matrix, distances, N, a=1.0, b=1.0, c = 0.0, shift = 0.0):
    forces = np.zeros((2*N, 3))
    potential_energy = 0.0

    for i in range(N):
        idx = 2 * i
        bond_idx = idx + 1
        # Double well interaction with bonded partner
        r_ij = distances[idx, bond_idx]
        r_vec = dist_matrix[idx, bond_idx]
        potential_energy += double_well_potential(r_ij, a, b, c, shift)
        force = double_well_force(r_ij, a, b, c, shift)
        force = force * r_vec / r_ij
        forces[idx] += force
        forces[bond_idx] -= force

    return potential_energy, forces

@njit
def forces_and_energy_harmonic(dist_matrix, distances, N, k = 1.0, r0 = 1.0):
    forces = np.zeros((2*N, 3))
    potential_energy = 0.0

    for i in range(N):
        idx = 2 * i
        bond_idx = idx + 1
        # Double well interaction with bonded partner
        r_ij = distances[idx, bond_idx]
        r_vec = dist_matrix[idx, bond_idx]
        potential_energy += harmonic_potential(r_ij, k, r0)
        force = harmonic_force(r_ij, k, r0)
        force = force * r_vec / r_ij
        forces[idx] += force
        forces[bond_idx] -= force

    return potential_energy, forces

@njit
def forces_and_energy_LJ(dist_matrix, distances, N, epsilon=1.0, sigma=1.0):
    forces = np.zeros((2*N, 3))
    potential_energy = 0.0

    for i in range(2 * N):
        for j in range(i + 1, 2 * N):
            if (i // 2) != (j // 2):  # Ensure particles are not bonded
                r_ij = distances[i, j]
                r_vec = dist_matrix[i, j]
                potential_energy += wca_potential(r_ij, epsilon, sigma)
                force = wca_force(r_ij, epsilon, sigma)
                force = force * r_vec / r_ij
                forces[i] += force
                forces[j] -= force

    return potential_energy, forces

class DPSwitching:
    def __init__(self, l0, l_1):
        self.l0 = l0
        self.l_1 = l_1

    def mix(self, x):
        s = (x - self.l_1)/(self.l0 - self.l_1)
        return (s**3*(-6*s**2 + 15*s - 10) + 1)

class ForceMixingCalc(Calculator):
    """
    Subcycles, how would that work? Because the OP
    is calculated every `subcycles` number of steps,
    so the dynamics sees the same OP for this amount
    of steps, when it in reality should change at each
    step...
    We use 1 for reactions, so not neccessary?
    """
    def __init__(self):
        super().__init__()
        self.implemented_properties = ["energy", "forces", "stress"]
        #self.mm_calc = MMCalculator()
        #self.dft_calc = DFTCalculator()
        self.intf0 = 0.8559
        self.intf_1 = 0.81
        self.Forcemixing = DPSwitching(self.intf0, self.intf_1)

        self.epsilon = 1.0
        self.sigma = 1.2
        self.a = 42.03604637628602
        self.b = 6.593074736188063
        self.shift = 1.0200388364613227
        self.c = 0
        self.k = 0.3691*97.171 # Hartree/Bohr^2  *= 97.171 to ev/angsrom^2
        self.r0 = 0.74 # 1.4 Bohr https://doi.org/10.1016/0009-2614(72)87211-7
        # * k harmonic approximation of dw is 6b-2b*(b/(2*a))**0.5
        # * find k such that double well is approximation to bond lengths of H2 at eq (state A)
        #   and state B is some pre-set length?
        # * barrier height is -b**2/(4*a) and should be 5 kBT
        # * kB in eV/K is 8.617333262e-5
        # * so 2 eq.: 1 for k and 1 for height



    def calculate(self, atoms, properties = None, system_changes = all_changes):
        order = atoms.calculate_order(
            atoms.system,
            xyz=atoms.positions,
            vel=atoms.get_velocities(),
            box=atoms.cell.diagonal(),
                )[0]
        #order = atoms.order

        N = int(atoms.positions.shape[0]/2)

        dist_matrix = distance_matrix(atoms.positions, atoms.cell.array[0,0])
        distances = compute_distances(dist_matrix)

        elj, flj = forces_and_energy_LJ(dist_matrix, distances, N,
                epsilon = self.epsilon, sigma = self.sigma)

        if order < self.intf_1:
            e, f = forces_and_energy_harmonic(dist_matrix, distances, N,
                    k = self.k, r0 = self.r0)

        elif order < self.intf0:
            e0, f0 = forces_and_energy_harmonic(dist_matrix, distances, N,
                    k = self.k, r0 = self.r0)

            e1, f1 = forces_and_energy_DW(dist_matrix, distances, N,
                    a = self.a, b = self.b, c = self.c, shift = self.shift)

            rho = self.Forcemixing.mix(order)
            e = e1
            f = rho * f0 + (1 - rho) * f1

        else:
            e, f = forces_and_energy_DW(dist_matrix, distances, N,
                    a = self.a, b = self.b, c = self.c, shift = self.shift)

        self.results = {"energy":e + elj, "forces":f + flj, "stress":np.zeros(6)}

class HarmonicCalc(Calculator):
    def __init__(self):
        super().__init__()
        self.implemented_properties = ["energy", "forces", "stress", "delta_f", "rho"]
        self.epsilon = 1.0
        self.sigma = 1.2
        self.k = 0.3691*97.171 # Hartree/Bohr^2  *= 97.171 to ev/angsrom^2
        self.r0 = 0.74 # 1.4 Bohr https://doi.org/10.1016/0009-2614(72)87211-7

    def calculate(self, atoms, properties = None, system_changes = all_changes):

        N = int(atoms.positions.shape[0]/2)
        dist_matrix = distance_matrix(atoms.positions, atoms.cell.array[0,0])
        distances = compute_distances(dist_matrix)

        elj, flj = forces_and_energy_LJ(dist_matrix, distances, N,
                epsilon = self.epsilon, sigma = self.sigma)

        e, f = forces_and_energy_harmonic(dist_matrix, distances, N,
                k = self.k, r0 = self.r0)

        self.results = {"energy":e + elj, "forces":f + flj, "stress":np.zeros(6)}

class DWCalc(Calculator):
    def __init__(self):
        super().__init__()
        self.implemented_properties = ["energy", "forces", "stress"]
        self.epsilon = 1.0
        self.sigma = 1.2
        self.a = 42.03604637628602
        self.b = 6.593074736188063
        self.shift = 1.0200388364613227
        self.c = 0

    def calculate(self, atoms, properties = None, system_changes = all_changes):

        N = int(atoms.positions.shape[0]/2)
        dist_matrix = distance_matrix(atoms.positions, atoms.cell.array[0,0])
        distances = compute_distances(dist_matrix)

        elj, flj = forces_and_energy_LJ(dist_matrix, distances, N,
                epsilon = self.epsilon, sigma = self.sigma)

        e, f = forces_and_energy_DW(dist_matrix, distances, N,
                a = self.a, b = self.b, c = self.c, shift = self.shift)

        self.results = {"energy":e + elj, "forces":f + flj, "stress":np.zeros(6)}
