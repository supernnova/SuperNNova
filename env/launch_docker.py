import os
import shlex
import argparse
import subprocess
from pathlib import Path


def launch_docker():

    os.getcwd()
    os.getuid()
    snn_dir = os.path.abspath(Path(os.path.dirname(os.path.realpath(__file__))).parent)
    dump_dir = os.path.abspath(Path(snn_dir).parent)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, choices=['cpu','gpu','gpu10'],help="Use which image gpu or cpu")
    parser.add_argument("--dump_dir", default=dump_dir, help='Dir to dump results')
    parser.add_argument("--raw_dir", help='Optional dir to point towards data')

    args = parser.parse_args()

    cmd = (
        "docker run -it --rm ")

    cmd += " --gpus all " if 'gpu' in args.image else ""

    cmd += (
        f" -v {snn_dir}:/u/home/SuperNNova"
        f" -v {args.dump_dir}:/u/home/snndump"
        )

    if args.raw_dir:
        cmd += f" -v {args.raw_dir}:/u/home/raw"

    cmd += (
        f" -e HOST_USER_ID={os.getuid()} "
        f" -e HOST_USER_GID={os.getgid()} "
        f" rnn-{args.image}:latest"
    )
    try:
        subprocess.check_call(shlex.split(cmd))
    except Exception as err:
        print(err)
        print("Possible errors:")
        print("You may not have a GPU.")
        print(f"You may not have built the images ==> call make {args.image}")


if __name__ == "__main__":

    launch_docker()
