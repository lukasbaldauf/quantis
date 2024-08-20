import numpy as np
from infretis.classes.orderparameter import OrderParameter
from MDAnalysis.analysis.distances import distance_array
from ase.geometry import find_mic
from ase.cell import Cell

class MaxDistance(OrderParameter):
    def __init__(self):
        txt = "The maximum distance between two bonded atoms"
        super().__init__(description=txt, velocity=False)

    def calculate(self, system):
        Natoms = system.pos.shape[0]
        cell = Cell.fromcellpar([*system.box, 90, 90, 90])
        order = -69
        imin = -69
        for i in range(Natoms//2):
            vec, dist = find_mic(system.pos[2 * i] - system.pos[2*i + 1], cell = cell)
            if dist > order:
                order = dist
                imin = 2*i
        return [order, imin]
