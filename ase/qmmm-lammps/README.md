```bash
export OMP_NUM_THREADS=1
```

```python
from ase.io import read
from lammps_xtb_qmmmcalc import QMMMCalc
from ase.md import VelocityVerlet
from ase import units

atoms = read("atoms.traj") # also contains unit cell, which doesn't change in lammpscalc
atoms.calc = QMMMCalc()

dyn = VelocityVerlet(atoms, dt = 0.5*units.fs)

for i in range(10):
	dyn.step()
	atoms.write("traj.pdb", append = True)
```
