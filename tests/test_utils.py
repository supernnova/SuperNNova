import os
import shlex
import shutil
import subprocess
from contextlib import ContextDecorator


class testmanager(ContextDecorator):

    def __enter__(self):

        self.dir_path = os.path.dirname(os.path.realpath(__file__))

        for folder in ["predictions", "preprocessed", "processed", "models"]:
            shutil.rmtree(
                os.path.join(self.dir_path, "dump", folder), ignore_errors=True
            )

        cmd = f"python run.py --data --dump_dir tests/dump"
        subprocess.check_call(shlex.split(cmd))

    def __exit__(self, type, value, traceback):

        for folder in ["predictions", "preprocessed", "processed"]:
            shutil.rmtree(
                os.path.join(self.dir_path, "dump", folder), ignore_errors=True
            )


def call_cmd(cmd):

    try:
        subprocess.check_call(shlex.split(cmd))
    except subprocess.CalledProcessError:
        assert False
