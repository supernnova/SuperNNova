import os
import shlex
import shutil
import subprocess
from contextlib import ContextDecorator


class testmanager(ContextDecorator):
    """Per-test setup/teardown.

    The HDF5 database in ``tests/dump/processed/`` is normally built once by
    ``tests/test_00_data.py`` and shared by every test, so we deliberately
    leave ``processed/`` alone between tests.

    If the database is missing — e.g. someone ran ``pytest tests/test_rnn.py``
    without first running ``test_00_data.py`` — we rebuild it here so any
    subset of tests can be run on its own, matching the old behaviour.
    """

    def __enter__(self):

        self.dir_path = os.path.dirname(os.path.realpath(__file__))

        # Clean stale per-test outputs only — keep processed/ (the dataset).
        for folder in ["predictions", "models"]:
            shutil.rmtree(
                os.path.join(self.dir_path, "dump", folder), ignore_errors=True
            )

        # Build the dataset on demand if it isn't there yet.
        db = os.path.join(self.dir_path, "dump", "processed", "database.h5")
        if not os.path.isfile(db):
            cmd = (
                "python run.py --data "
                "--dump_dir tests/dump "
                "--raw_dir tests/raw "
                "--fits_dir tests/fits"
            )
            subprocess.check_call(shlex.split(cmd))

    def __exit__(self, type, value, traceback):

        # Same: keep processed/ so following tests can reuse the database.
        for folder in ["predictions"]:
            shutil.rmtree(
                os.path.join(self.dir_path, "dump", folder), ignore_errors=True
            )


def call_cmd(cmd):

    try:
        subprocess.check_call(shlex.split(cmd))
    except subprocess.CalledProcessError:
        assert False
