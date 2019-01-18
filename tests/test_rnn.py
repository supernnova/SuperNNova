import os
import pytest
from pathlib import Path
from tests.test_utils import call_cmd
from tests.test_utils import testmanager


@pytest.mark.parametrize(
    "dataset, redshift, norm, nb_classes",
    [
        ("photometry", None, "global", 2),
        ("photometry", None, "global", 3),
        ("photometry", None, "global", 7),
        ("photometry", None, "none", 2),
        ("photometry", None, "perfilter", 2),
        ("photometry", None, "perfilter", 2),
        ("photometry", None, "perfilter", 2),
        ("photometry", "zpho", "none", 2),
        ("photometry", "zspe", "none", 2),
        ("saltfit", None, "global", 2),
        ("saltfit", None, "global", 3),
        ("saltfit", None, "global", 7),
        ("saltfit", None, "none", 2),
        ("saltfit", None, "perfilter", 2),
        ("saltfit", None, "perfilter", 2),
        ("saltfit", None, "perfilter", 2),
        ("saltfit", "zpho", "none", 2),
        ("saltfit", "zspe", "none", 2),
    ],
)
@testmanager()
def test_rnn(dataset, redshift, norm, nb_classes):

    cmd = (
        f"python run.py --train_rnn --source_data {dataset} "
        f"--dump_dir tests/dump "
        f"--nb_epoch 1 --norm {norm} --nb_classes {nb_classes}"
    )
    if redshift is not None:
        cmd += f" --redshift {redshift}"
    call_cmd(cmd)

    cmd = (
        f"python run.py --validate_rnn --source_data {dataset} "
        f"--dump_dir tests/dump "
        f"--norm {norm} --nb_classes {nb_classes}"
    )
    if redshift is not None:
        cmd += f" --redshift {redshift}"
    call_cmd(cmd)

    # Check we do have predictions saved
    dir_path = os.path.dirname(os.path.realpath(__file__))
    assert len([e for e in (Path(dir_path) / "dump/models").glob("*/PRED*")]) == 1


if __name__ == "__main__":

    test_rnn("saltfit", None, "perfilter", 2)
