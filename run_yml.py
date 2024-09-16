import yaml
import json
import shlex
import argparse
import subprocess
from pathlib import Path

# import supernnova.conf as conf

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SuperNNova using yaml")

    parser.add_argument("yml", type=Path, help="Yaml configuration file")

    parser.add_argument(
        "--mode",
        choices=["data", "train_rnn", "validate_rnn", "plot_lcs"],
    )
    args = parser.parse_args()

    config = yaml.load(open(args.yml, "r"), Loader=yaml.Loader)

    cmd = f"python run.py --{args.mode} "

    for k, v in config.items():
        if isinstance(v, list):
            cmd += f"--{k} {' '.join(v)} "
        elif isinstance(v, bool):
            if v is True and k not in ["bidirectional", "random_length"]:
                cmd += f"--{k} "
        elif isinstance(v, dict):
            cmd += f"--{k} '{json.dumps(v)}' "
        else:
            cmd += f"--{k} {v} "

    print("")
    print(cmd)
    print("")

    subprocess.check_call(shlex.split(cmd))
