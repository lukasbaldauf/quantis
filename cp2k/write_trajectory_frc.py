def write_xyz_trajectory_frc(
    filename: str,
    pos: np.ndarray,
    vel: np.ndarray,
    frc: np.ndarray,
    names: list[str] | None,
    box: np.ndarray | None,
    step: int | None = None,
    append: bool = True,
) -> None:
    """Write a XYZ snapshot to a trajectory file.

    This is intended as a lightweight alternative for just
    dumping snapshots to a trajectory file.

    Args:
        filename: Path to the file to write to.
        pos: The positions to write.
        vel: The velocities to write.
        names: Atom names to write.
        box: The box dimensions/vectors
        step: If the `step` is given, then the step number is
            written to the header.
        append: Determines if we append (if True) or overwrite (if False)
            if the `filename` file exists.
    """
    npart = len(pos)
    if names is None:
        names = ["X"] * npart
    filemode = "a" if append else "w"
    with open(filename, filemode, encoding="utf-8") as output_file:
        output_file.write(f"{npart}\n")
        header = ["#"]
        if step is not None:
            header.append(f"Step: {step}")
        if box is not None:
            header.append(f'Box: {" ".join([f"{i:9.4f}" for i in box])}')
        header.append("\n")
        header_str = " ".join(header)
        output_file.write(header_str)
        for i in range(npart):
            line = _XYZ_BIG_VEL_FRC_FMT.format(
                names[i],
                pos[i, 0],
                pos[i, 1],
                pos[i, 2],
                vel[i, 0],
                vel[i, 1],
                vel[i, 2],
                frc[i, 0],
                frc[i, 1],
                frc[i, 2],
            )
            output_file.write(f"{line}\n")
