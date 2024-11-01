import shlex
import subprocess


def call_cmd(cmd):

    try:
        subprocess.check_call(shlex.split(cmd))
    except subprocess.CalledProcessError:
        assert False
