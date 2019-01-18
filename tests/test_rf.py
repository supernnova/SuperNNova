import os
import glob
import pytest
from tests.test_utils import call_cmd
from tests.test_utils import testmanager


@pytest.mark.parametrize("redshift", [None, "zpho", "zspe"])
@testmanager()
def test_rf(redshift):

    cmd = (
        "python run.py --train_rf --max_depth 1 --max_features 1 --n_estimators 1 --dump_dir tests/dump"
    )
    if redshift is not None:
        cmd += f" --redshift {redshift}"
    call_cmd(cmd)
    cmd = "python run.py --validate_rf --dump_dir tests/dump"
    if redshift is not None:
        cmd += f" --redshift {redshift}"
    call_cmd(cmd)

    # Check we do have predictions saved
    dir_path = os.path.dirname(os.path.realpath(__file__))
    assert len(glob.glob(os.path.join(dir_path, "dump", "predictions/*forest*"))) == 1


if __name__ == "__main__":

    test_rf()
