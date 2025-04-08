import numpy as np
from infretis.classes.engines.amber_engine import AmberEngine
from infretis.classes.system import System
from infretis.classes.path import Path
from infretis.classes.formatter import FileIO, OutputFormatter
from infretis.classes.orderparameter import Distance

engine = AmberEngine("sander", 0.001, 300, 1, "amber_input/")
engine.rgen = np.random.default_rng()
engine.order_function = Distance([0,1], periodic = True)

# test reverse_velocities on frame and restrt
for infile, outfile in zip(["amber.mdcrd", "amber.restrt",],["rev_vel.mdcrd", "rev_vel.restrt"]):
    engine._reverse_velocities(engine.input_path / infile, outfile)
    x0,v0,b0,_ = engine._read_configuration(engine.input_path / infile)
    x1,v1,b1,_ = engine._read_configuration(outfile)
    assert np.allclose(x0,x1)
    assert np.allclose(v0,-v1)
    assert np.allclose(b0,b1)

# test that velocity modification only affects velocities
for infile, outfile in zip(["amber.mdcrd", "amber.restrt",],["rev_vel.mdcrd", "rev_vel.restrt"]):
    system = System()
    system.config = (engine.input_path/ infile, 0)
    engine.modify_velocities(system, {})
    x0,v0,b0,_ = engine._read_configuration(engine.input_path / infile)
    x1,v1,b1,_ = engine._read_configuration(f"genvel.{engine.ext}")
    assert np.allclose(x0,x1)
    assert not np.allclose(v0,v1)
    assert np.allclose(b0,b1)

# test that extracting a frame into a restart doesn't change anything
engine._extract_frame(engine.input_path / "amber.mdcrd", 0, "test.restrt")
x0,v0,b0,_ = engine._read_configuration(engine.input_path / "amber.mdcrd")
x1,v1,b1,_ = engine._read_configuration(f"test.restrt")
assert np.allclose(x0,x1)
assert np.allclose(v0,v1)
assert np.allclose(b0,b1)

# test that extracting a frame from a restrt doesn't change anything
engine._extract_frame(engine.input_path / "amber.restrt", 0, "test.mdcrd")
x0,v0,b0,_ = engine._read_configuration(engine.input_path / "amber.restrt")
x1,v1,b1,_ = engine._read_configuration(f"test.mdcrd")
assert np.allclose(x0,x1)
assert np.allclose(v0,v1)
assert np.allclose(b0,b1)

# test that changing a restrt into a frame doesn't change anything
engine._add_restrt(engine.input_path / "amber.restrt", "test_add.mdcrd")
x0,v0,b0,_ = engine._read_configuration(engine.input_path / "amber.restrt")
x1,v1,b1,_ = engine._read_configuration(f"test_add.mdcrd")
assert np.allclose(x0,x1)
assert np.allclose(v0,v1)
assert np.allclose(b0,b1)

# test _write_restrt
box_length = 30
xyz = np.zeros((2,3))
xyz[0,0] = 0.05
vel = np.zeros((2,3))
vel[0,0] = 100 # 2*50 angstrom/ps = 0.1 angstrom/fs
box = np.eye(3)*box_length
engine._write_restrt(xyz, vel, box, "initial.restrt")
x0,v0,b0,_ = engine._read_configuration("initial.restrt")
assert np.allclose(xyz, x0)
assert np.allclose(vel, v0)
assert np.allclose(box, b0)

# test that the distance between 2 particles is greater then 1.0 angstrom
# after 11 steps if they start 0.05 angstrom apart and move with a relative
# velocity of 0.1 angstrom/fs (dt = 1 fs)
path = Path(maxlen=20)
system = System()
system.config = ("initial.restrt", 0)
msg_file = FileIO(
    "msg-propagate_test.txt", "w", OutputFormatter("MSG_File"), backup=False
)
msg_file.open()
ens_set = {"interfaces":[-10, -10, 1.0]}
success, status = engine._propagate_from("propagate_test", path, system, ens_set, msg_file, reverse=False)
msg_file.close()
assert len(path.phasepoints)==11
assert np.allclose([pp.order[0] for pp in path.phasepoints], np.linspace(0.05, 1.05, 11))
