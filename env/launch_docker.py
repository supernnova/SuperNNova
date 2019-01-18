import os
import shlex
import argparse
import subprocess
from pathlib import Path


def launch_docker():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", default="../../sndump", help="Dump dir")
    parser.add_argument("--use_cuda", action="store_true", help="Use gpu image")

    args = parser.parse_args()

    pwd = os.getcwd()
    UID = os.getuid()

    if args.use_cuda:
        cmd = (
            f"nvidia-docker run -it --rm --user {UID}"
            f" -v {pwd}/../../SuperNNova:/home/SuperNNova"
            f" -v {pwd}/{args.dump_dir}:/home/sndump rnn-gpu:latest"
        )
        try:
            subprocess.check_call(shlex.split(cmd))
        except Exception as err:
            print(err)
            print("Possible errors:")
            print("You may not have installed nvidia-docker.")
            print("You may not have a GPU.")
            print("You may not have built the images ==> call make gpu or make cpu")
    else:

        cmd = (
            f"docker run -it --rm --user {UID}"
            f" -v {pwd}/../../SuperNNova:/home/SuperNNova"
            f" -v {pwd}/{args.dump_dir}:/home/{Path(args.dump_dir).name} rnn-cpu:latest"
        )

        try:
            subprocess.check_call(shlex.split(cmd))
        except Exception as err:
            print(err)
            print("Possible errors:")
            print("You may not have installed docker.")
            print("You may not have built the images ==> call make gpu or make cpu")


if __name__ == "__main__":

    launch_docker()
