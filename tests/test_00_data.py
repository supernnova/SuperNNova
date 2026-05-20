"""Build the HDF5 database used by all subsequent RNN tests.

Runs first thanks to alphabetical file ordering (``test_00_data`` < ``test_rnn``).

Previously the data step was rebuilt before every test via
``testmanager.__enter__``. That was slow (21x rebuilds) and brittle — a
silent data failure could surface as a misleading train-step error. Building
it once at the start of the suite is faster and gives a clean, accurate
failure if the data step itself breaks.
"""
import os
import shlex
import shutil
import subprocess


def test_build_data():
    # Start from a clean dump dir so the build is fully deterministic.
    dump_dir = os.path.join(os.path.dirname(__file__), "dump")
    for sub in ("predictions", "preprocessed", "processed", "models"):
        shutil.rmtree(os.path.join(dump_dir, sub), ignore_errors=True)

    cmd = (
        "python run.py --data "
        "--dump_dir tests/dump "
        "--raw_dir tests/raw "
        "--fits_dir tests/fits"
    )
    subprocess.check_call(shlex.split(cmd))

    # Sanity check: the HDF5 database must exist after the data step.
    db = os.path.join(dump_dir, "processed", "database.h5")
    assert os.path.isfile(db), f"Expected database at {db}"
