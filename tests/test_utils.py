import shlex
import subprocess


def call_cmd(cmd):

    try:
        subprocess.check_call(shlex.split(cmd))
    except subprocess.CalledProcessError as e:
        raise AssertionError(
            f"Command failed with exit code {e.returncode}: {cmd}"
        ) from e
