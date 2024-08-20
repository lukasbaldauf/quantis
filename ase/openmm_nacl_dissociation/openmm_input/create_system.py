import openmm
import openmm.app
import openmm.unit
from openff.toolkit import ForceField, Molecule, unit

from openff.interchange import Interchange
from openff.interchange.components._packmol import UNIT_CUBE, pack_box

Na = Molecule.from_smiles("[Na+]")
Cl = Molecule.from_smiles("[Cl-]")
water = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")

### Naming the residue is not needed to parameterize the system or run the simulation, but makes visualization easier
for atom in water.atoms:
    atom.metadata["residue_name"] = "HOH"

topology = pack_box(
    molecules=[Na, Cl, water],
    number_of_copies=[1, 1, 383],
    box_vectors=2.3 * UNIT_CUBE * unit.nanometer,
)

sage = ForceField("/home/lukas/quantis/diels-alder/openmm_stuff/myopenff-2.2.0.offxml")

interchange: Interchange = Interchange.from_smirnoff(
    force_field=sage, topology=topology
)


from ase.io import read
atoms = read("initconf.gro")

integrator = openmm.VerletIntegrator(2.0/1000)
simulation = interchange.to_openmm_simulation(
        combine_nonbonded_forces=False,
        integrator=integrator)


simulation.topology.setPeriodicBoxVectors(atoms.cell.array/10)
simulation.context.setPeriodicBoxVectors(*atoms.cell.array/10)

with open ("topology.pdb", "w") as pdbfile:
    openmm.app.PDBFile.writeFile(simulation.topology, atoms.positions, pdbfile)

with open("system.xml", "w") as output:
    output.write(openmm.XmlSerializer.serialize(simulation.system))
