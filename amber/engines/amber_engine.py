"""An Amber engine interface."""

import logging
from typing import Dict, Tuple, Union, Any, Optional
import numpy as np
from infretis.classes.engines.enginebase import EngineBase
from infretis.classes.formatter import FileIO
from infretis.classes.path import Path as InfPath
from infretis.classes.system import System
import netCDF4 as nc
from ase.cell import Cell
import time
from pathlib import Path
import shutil
import os
import subprocess
import signal
import parmed as pmd
from infretis.classes.engines.cp2k import kinetic_energy, reset_momentum

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


class AmberEngine(EngineBase):
    """A simple Amber engine class.

    The engine asssumes the units angstrom, picoseconds and kcal/mol.

    TODO:
        ntpr: mdout printed every ntp steps; read energy from this


    """

    def __init__(
        self,
        sander: str,
        timestep: float,
        temperature: float,
        subcycles: int,
        input_path: str,
        sleep: float = 0.1,
        infretis_genvel: bool = True,
        exe_path: Union[str, Path] = Path(".").resolve(),
        ):
        """
        Initialize the Amber engine.

        Args:
            sander: the sander exe (e.g. 'srun -c 1 -n 1 sander')
            timestep: timestep in ps
            temperature: temperature in K
            subcylces: MD steps per infretis step
            input_path: path where amber files are found
            exe_path: from where we execute infretisrun
        """
        super().__init__("Amber engine", timestep, subcycles)
        self.timestep = timestep
        self.subcycles = subcycles
        self.temperature = temperature
        self.exe_path = Path(exe_path)
        self.input_path = self.exe_path / input_path
        self.sleep = sleep
        if not infretis_genvel:
            raise ValueError("Only infretis_genvel is supported atm")
        self.infretis_genvel = infretis_genvel

        self.input_files = {
                "prmtop":(self.input_path / "amber.prmtop").resolve(),
                "inpcrd":(self.input_path / "amber.inpcrd").resolve(),
                "mdin":(self.input_path / "amber.mdin").resolve(),
                "mdcrd":(self.input_path / "amber.mdcrd").resolve(),
                "restrt":(self.input_path / "amber.restrt").resolve(),
                }


        # Load masses for velocity generation
        structure = pmd.load_file(str(self.input_files["prmtop"]))
        self.masses = np.array([atom.mass for atom in structure.atoms]).reshape(-1,1)

        # check that input files exist
        for ifile in self.input_files.values():
            if not ifile.exists():
                raise ValueError(f"Did not find {ifile.resolve()}")
        self.sander_cmd = (
                f"{sander} "
                f"-p {self.input_files['prmtop']} "
        )

        # Check that temperature matches with the infretis temperature
        # and that we have one line per keyword-value pair in the input file
        with open(self.input_files["mdin"], "r") as ifile:
            for i,line in enumerate(ifile):
                # comment line
                if i == 0 or line.startswith("&") or line.startswith("\\"):
                    continue
                spl = line.split(",")
                if len(spl) > 2:
                    raise ValueError(
                    f"Input file {self.input_files['mdin']}"
                    f" should contain one value per line, not more: {line}"
                    )
                if "temp" in line:
                    amber_temp = float(spl[0].split("=")[1])
                    if amber_temp != self.temperature:
                        raise ValueError(
                        f"The temperature in {self.input_files['mdin']}"
                        f" {amber_temp} != {self.temperature} from the .toml"
                        )

        self.ext = "mdcrd"
        self.name = "amber"
        # kb and beta only needed for quantis and internal velocity generation
        self.kb = 1.987204259e-3  # kcal/(mol*K)
        self._beta = 1 / (self.kb * self.temperature)
        #self.exe_dir = self.input_path.resolve()

    def _propagate_from(
        self,
        name: str,
        path: InfPath,
        system: System,
        ens_set: Dict[str, Any],
        msg_file: FileIO,
        reverse: bool = False,
    ) -> Tuple[bool, str]:
        """Propagate the equations of motion with Amber.

        Since amber doesn't add the initial phasepoint to the trajectories, we
        manually add the first point to the infretis path.
        """
        status = f"Propagating with Sander (reverse = {reverse})"
        interfaces = ens_set["interfaces"]
        logger.debug(status)
        success = False,
        left, _, right = interfaces
        initial_conf = system.config[0]

        xyzi, veli, boxi, _ = self._read_configuration(initial_conf)
        order = self.calculate_order(system, xyz=xyzi, vel=veli, box=boxi)
        msg_file.write(
                f"# Initial order parameter: {' '.join([str(i) for i in order])}"
                )

        traj_file = os.path.join(self.exe_dir, f"{name}.{self.ext}")
        msg_file.write(f"# Trajectory file is: {traj_file}")
        run_settings = {
                "dt":self.timestep, # timestep in picoseconds
                "nstlim": path.maxlen * self.subcycles, # nr of md steps
                "nstwx": self.subcycles, # write coordinates (and vels) every subs
                "ntpr": self.subcycles, # mdout energy, to be read
                "irest": 1, # restart simulation from restrt
                "ntx": 5, # read both coords and velocities
                "ntwv": -1, # check also that all keyowrds are set
                }
        run_inp = os.path.join(self.exe_dir, "run.mdin")
        write_amber_input(self.input_files["mdin"], run_inp, run_settings)

        # first phasepoint is added manually to the path as a single frame trajectory
        step_nr = 0
        first_phasepoint = os.path.join(self.exe_dir, f"{traj_file.split('.')[0]}_sp.{self.ext}")
        self._add_restrt(initial_conf, first_phasepoint)
        msg_file.write(f'{step_nr} {" ".join([str(j) for j in order])}')
        snapshot = {"order": order, "config": (first_phasepoint, step_nr), "vel_rev": reverse}
        phase_point = self.snapshot_to_system(system, snapshot)
        status, success, stop, add = self.add_to_path(path, phase_point, left, right)
        step_nr += 1
        if stop:
            return success, status

        cmd = f"{self.sander_cmd} -A -i {run_inp} -x {traj_file} -c {initial_conf}".split()
        cwd = self.exe_dir
        # capture stdout and stderr
        out_name = os.path.join(self.exe_dir, "stdout.txt")
        err_name = os.path.join(self.exe_dir, "stderr.txt")

        amber_was_terminated = False
        with open(out_name, "wb") as fout, open(err_name, "wb") as ferr:
            exe = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=fout,
                stderr=ferr,
                shell=False,
                cwd=cwd,
                start_new_session = True,
            )

            # wait for trajectories to appear
            while not os.path.exists(traj_file):
                time.sleep(self.sleep)
                if exe.poll is not None:
                    logger.debug("Amber execution stopped")
                    break

            # same as lammps basically
            while not stop:
                nc_file = nc.Dataset(traj_file)
                current_len = len(nc_file.variables['coordinates'])
                if not current_len != step_nr and len(nc_file.variables["coordinates"].shape)==3:
                    time.sleep(self.sleep)
                else:
                    for i in range(step_nr-1, current_len):
                        xyz = nc_file.variables['coordinates'][i, :, :].data
                        vel = nc_file.variables['velocities'][i, :, :].data
                        box_len = list(nc_file.variables['cell_lengths'][i, :].data)
                        box_ang = list(nc_file.variables['cell_angles'][i, :].data)
                        box = Cell.fromcellpar(box_len + box_ang)
                        order = self.calculate_order(
                            system, xyz=xyz, vel=vel, box=box
                        )
                        msg_file.write(
                            f'{step_nr} {" ".join([str(j) for j in order])}'
                        )
                        snapshot = {
                            "order": order,
                            "config": (traj_file, i),
                            "vel_rev": reverse,
                        }
                        phase_point = self.snapshot_to_system(system, snapshot)
                        status, success, stop, add = self.add_to_path(
                            path, phase_point, left, right
                        )
                        if stop:
                            # terminate process if still running
                            if exe.poll() is None:
                                logger.debug("Terminating Amber execution")
                                os.killpg(os.getpgid(exe.pid), signal.SIGTERM)
                                # wait for process to die, necessary for mpi
                                exe.wait(timeout=360)
                                logger.debug(
                                    "Amber propagation ended at %i. Reason: %s",
                                    step_nr,
                                    status,
                                )
                                amber_was_terminated = True
                            break

                        step_nr += 1

            return_code = exe.returncode
            if return_code != 0 and not amber_was_terminated:
                logger.error(f"Execution of external program {self.description} failed!")
                logger.error("Attempted command: %s", ' '.join(cmd))
                logger.error("Execution directory: %s", cwd)
                logger.error(
                    "Return code from external program: %i", return_code
                )
                logger.error("STDOUT, see file: %s", out_name)
                logger.error("STDERR, see file: %s", err_name)
                msg = (
                    f"Execution of external program ({self.description}) "
                    f"failed with command:\n {' '.join(cmd)}.\n"
                    f"Return code: {return_code}"
                )
                raise RuntimeError(msg)
            if (return_code is not None) and (
                return_code == 0 or amber_was_terminated
            ):
                self._removefile(out_name)
                self._removefile(err_name)

            msg_file.write("# Propagation done.")
            self._removefile(run_inp)
            return success, status

    def _read_configuration(
        self,
        filename: str
    ) -> Tuple[
        np.ndarray, np.ndarray, Optional[np.ndarray], Optional[list[str]]
        ]:
        """Read a configuration file (a single frame trajectory)"""
        data = nc.Dataset(filename)
        pos = data.variables["coordinates"]
        if len(pos.shape)==2:
            pos = pos[:].data
            vel = data.variables["velocities"][:].data
            box = Cell.fromcellpar(
                    list(data.variables["cell_lengths"][:].data) + list(data.variables["cell_angles"][:].data)
                    )
        else:
            pos = pos[0].data
            vel = data.variables["velocities"][0].data
            box = Cell.fromcellpar(
                    list(data.variables["cell_lengths"][0].data) + list(data.variables["cell_angles"][0].data)
                    )
        return pos, vel, box, None

    def _extract_frame(self, traj_file, idx, out_file):
        """Exrtact a frame into a restrt file."""
        with nc.Dataset(traj_file) as readf:
            # file is in trajectory format
            if len(readf.variables["coordinates"].shape)==3:
                shutil.copy(self.input_files["restrt"], out_file)
                xyz = readf.variables["coordinates"][idx]
                vel = readf.variables["velocities"][idx]
                box_lens = readf.variables["cell_lengths"][idx]
                box_angs = readf.variables["cell_angles"][idx]

                with nc.Dataset(out_file, 'r+') as writef:
                    writef.variables["velocities"][:] = vel
                    writef.variables["coordinates"][:] = xyz
                    writef.variables["cell_lengths"][:] = box_lens
                    writef.variables["cell_angles"][:] = box_angs
            # file is allready in restrt format
            else:
                shutil.copy(traj_file, out_file)

    def _add_restrt(self, restart, traj_file):
        """Add a restart file to an (empty) traj_file

        Used for adding teh initial phasepoint to amber trajectories, since
        they are not added by default.

        """
        shutil.copy(self.input_files["mdcrd"], traj_file)
        with nc.Dataset(restart) as readf:
            shape = readf.variables["coordinates"].shape
            if not len(shape)==2:
                raise ValueError("Trying read a restart file, but doesn't seem to be a restart")
            xyz = readf.variables["coordinates"][:]
            vel = readf.variables["velocities"][:]
            box_lens = readf.variables["cell_lengths"][:]
            box_angs = readf.variables["cell_angles"][:]

            with nc.Dataset(traj_file, 'r+') as writef:
                writef.variables["velocities"][:] = vel.reshape(1, *shape)
                writef.variables["coordinates"][:] = xyz.reshape(1, *shape)
                writef.variables["cell_lengths"][:] = box_lens.reshape(1, shape[1])
                writef.variables["cell_angles"][:] = box_angs.reshape(1, shape[1])

    def _write_restrt(self, xyz, vel, box, out_file):
        box = Cell(box)
        shutil.copy(self.input_files["restrt"], out_file)
        with nc.Dataset(out_file, 'r+') as writef:
            writef.variables["velocities"][:] = vel
            writef.variables["coordinates"][:] = xyz
            writef.variables["cell_lengths"][:] = box.lengths()
            writef.variables["cell_angles"][:] = box.angles()



    def _reverse_velocities(self, filename, outfile):
       shutil.copy(filename, outfile)
       with nc.Dataset(outfile, 'r+') as out:
           out.variables["velocities"][:] *= -1

    def modify_velocities(self, system, vel_settings):
        """Run sander with 0 steps with initial velocity generation."""
        kin_old = system.ekin
        conf_in = self.dump_frame(system)
        conf_out = os.path.join(self.exe_dir, f"genvel.{self.ext}")

        if not self.infretis_genvel:
            settings = {"itemp": self.temperature, "ntx": 1, "nstlim": 1, "ntwx":1}
            run_inp = os.path.join(self.exe_dir, "genvel.mdin")
            write_amber_input(self.input_files['mdin'], run_inp, settings)
            cmd = self.sander_cmd + f" -i {run_inp} -r {conf_out} -c {conf_in} -O"
            stderr = os.path.join(self.exe_dir, "stderr")
            stdout = os.path.join(self.exe_dir, "stdout")
            with open(stdout, "wb") as fout, open(stderr, "wb") as ferr:
                exe = subprocess.run(cmd.split(), stdout=fout, stderr=ferr)
                if exe.returncode != 0:
                    raise ValueError(
                            f"Modify vels command {cmd} failed with returncode"
                            f"{exe.returncode} stderr in {stderr} and stdout in {stdout}"
                            )
        else:
            # just use the dumped frame, change name to genvel and modify vels
            os.rename(conf_in, conf_out)
            with nc.Dataset(conf_out, 'r+') as wfile:
                vel = wfile.variables["velocities"][:].data
                vel, _ = self.draw_maxwellian_velocities(vel, self.masses, self.beta)
                if vel_settings.get("zero_momentum", False):
                    vel = reset_momentum(vel, self.masses)
                wfile.variables["velocities"][:] = vel
        system.config = (conf_out, 0)
        return 0.0, 0.0


    def set_mdrun(self, md_items):
        """Set the executional directory for workers."""
        self.exe_dir = md_items["exe_dir"]

def write_amber_input(template, outfile, settings):
    """Write an input file that is used to run Sander.

    If values are to be replaced, we need the following format:

    key0 = value0,
    ...
    keyN = valueN,

    Args:
        template: a template file with the settings
        outfile: the output file we use to run Sander with
        settings: settings  that are changed/added to the template file
    """
    with open(template, "r") as rfile:
        with open(outfile, "w") as outfile:
            for line in rfile:
                if "=" in line:
                    key, val = line.split(",")[0].split("=")
                    if key in settings.keys():
                        line = f"{key}={settings[key]},\n"
                outfile.write(line)
