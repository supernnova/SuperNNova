"""Tests for multi-dimensional FITS column handling.

FITS tables can contain vector / array-valued columns (``TDIM > 1``).
``astropy.Table.to_pandas`` raises ``ValueError`` on such columns because
pandas cannot natively hold n-dim arrays in a column. The helpers in
``data_utils.py`` either drop those columns with a warning or skip them
at the column-filter step so the rest of the pipeline keeps working.
"""

import numpy as np
import pytest
from collections import OrderedDict

from astropy.table import Table

from supernnova.utils import data_utils, experiment_settings


def _write_fits_with_multidim(
    path, n_rows=5, vec_width=3, include_multidim=True
):
    """Create a small FITS table with scalar (+ optional multi-D) columns."""
    t = Table()
    t["SNID"] = np.array([f"{i:05d}" for i in range(n_rows)], dtype="S20")
    t["SNTYPE"] = np.array(["101"] * n_rows, dtype="S10")
    t["PEAKMJD"] = np.linspace(50000.0, 60000.0, n_rows).astype(np.float32)
    if include_multidim:
        # Per-row vector columns — these are what crash to_pandas()
        t["PSF_SIG"] = np.zeros((n_rows, vec_width), dtype=np.float32)
        t["FLUXCAL_ARR"] = np.ones((n_rows, vec_width), dtype=np.float32)
    t.write(str(path), format="fits", overwrite=True)


def _make_settings(sntypes=None, target_sntype="Ia"):
    """Minimal ExperimentSettings for process_header_FITS tests."""
    cli_args = {
        "sntypes": OrderedDict(sntypes or {"101": "Ia"}),
        "sntype_var": "SNTYPE",
        "target_sntype": target_sntype,
        "data_testing": False,
        "no_dump": True,
        "use_cuda": False,
        "model": "vanilla",
        "weight_decay": 0.0,
    }
    return experiment_settings.ExperimentSettings(cli_args)


class TestLoadPandasFromFit:
    """Behaviour of data_utils.load_pandas_from_fit on multi-D columns."""

    def test_drop_multidim_by_default(self, tmp_path, capsys):
        """Multi-D columns are silently dropped with a warning by default."""
        fits_path = tmp_path / "head.fits"
        _write_fits_with_multidim(fits_path)

        df = data_utils.load_pandas_from_fit(str(fits_path))

        # Scalar columns survive
        assert "SNID" in df.columns
        assert "SNTYPE" in df.columns
        assert "PEAKMJD" in df.columns
        # Multi-D ones are gone
        assert "PSF_SIG" not in df.columns
        assert "FLUXCAL_ARR" not in df.columns
        assert len(df) == 5

        # User-visible warning was emitted
        captured = capsys.readouterr().out
        assert "Skipping multi-dim FITS cols" in captured
        assert "PSF_SIG" in captured

    def test_no_multidim_no_warning(self, tmp_path, capsys):
        """Files without multi-D columns produce no warning."""
        fits_path = tmp_path / "head.fits"
        _write_fits_with_multidim(fits_path, include_multidim=False)

        df = data_utils.load_pandas_from_fit(str(fits_path))

        assert set(df.columns) >= {"SNID", "SNTYPE", "PEAKMJD"}
        captured = capsys.readouterr().out
        assert "Skipping multi-dim" not in captured

    def test_columns_filter_avoids_loading_multidim(self, tmp_path, capsys):
        """A columns whitelist of scalar names skips multi-D cols silently."""
        fits_path = tmp_path / "head.fits"
        _write_fits_with_multidim(fits_path)

        df = data_utils.load_pandas_from_fit(
            str(fits_path), columns=["SNID", "SNTYPE", "PEAKMJD"]
        )

        assert sorted(df.columns) == sorted(["SNID", "SNTYPE", "PEAKMJD"])
        # Multi-D cols filtered out before the ndim check -> no warning
        captured = capsys.readouterr().out
        assert "Skipping multi-dim" not in captured

    def test_columns_filter_drops_requested_multidim(self, tmp_path, capsys):
        """If a multi-D column is explicitly requested it is still dropped."""
        fits_path = tmp_path / "head.fits"
        _write_fits_with_multidim(fits_path)

        df = data_utils.load_pandas_from_fit(
            str(fits_path), columns=["SNID", "PSF_SIG"]
        )

        assert "SNID" in df.columns
        assert "PSF_SIG" not in df.columns
        captured = capsys.readouterr().out
        assert "Skipping multi-dim" in captured

    def test_columns_filter_ignores_missing_names(self, tmp_path):
        """Unknown column names in the whitelist are silently ignored."""
        fits_path = tmp_path / "head.fits"
        _write_fits_with_multidim(fits_path, include_multidim=False)

        df = data_utils.load_pandas_from_fit(
            str(fits_path), columns=["SNID", "DOES_NOT_EXIST"]
        )

        assert list(df.columns) == ["SNID"]

    def test_multidim_error_raises(self, tmp_path):
        """multidim='error' surfaces the problem to the caller."""
        fits_path = tmp_path / "head.fits"
        _write_fits_with_multidim(fits_path)

        with pytest.raises(ValueError, match="Multi-dim FITS columns"):
            data_utils.load_pandas_from_fit(str(fits_path), multidim="error")

    def test_unknown_multidim_strategy_raises(self, tmp_path):
        """Unknown multidim strategy strings are rejected."""
        fits_path = tmp_path / "head.fits"
        _write_fits_with_multidim(fits_path)

        with pytest.raises(ValueError, match="Unknown multidim strategy"):
            data_utils.load_pandas_from_fit(str(fits_path), multidim="bogus")

    def test_dataframe_values_preserved(self, tmp_path):
        """Surviving scalar columns keep their original values."""
        fits_path = tmp_path / "head.fits"
        _write_fits_with_multidim(fits_path, n_rows=4)

        df = data_utils.load_pandas_from_fit(str(fits_path))

        # PEAKMJD was np.linspace(50000, 60000, 4)
        expected = np.linspace(50000.0, 60000.0, 4).astype(np.float32)
        np.testing.assert_allclose(df["PEAKMJD"].values, expected, rtol=1e-5)


def _write_phot_fits_with_multidim(path, n_rows=6, vec_width=4):
    """Create a PHOT-like FITS table with scalar + multi-D columns.

    Mirrors the columns process_single_FITS wants from a SNANA PHOT file
    (MJD, FLUXCAL, FLUXCALERR, FLT) plus per-row vector columns that would
    otherwise crash to_pandas().
    """
    t = Table()
    t["MJD"] = np.linspace(50000.0, 60000.0, n_rows).astype(np.float32)
    t["FLUXCAL"] = np.linspace(1.0, 10.0, n_rows).astype(np.float32)
    t["FLUXCALERR"] = np.full(n_rows, 0.1, dtype=np.float32)
    t["FLT"] = np.array(["g "] * n_rows, dtype="S2")
    # Per-row vector — the kind of column SNANA pipelines can carry along
    # (e.g. PSF info, per-aperture fluxes) that crashes to_pandas().
    t["PSF_VEC"] = np.zeros((n_rows, vec_width), dtype=np.float32)
    t["APER_FLUX"] = np.ones((n_rows, vec_width), dtype=np.float32)
    t.write(str(path), format="fits", overwrite=True)


def _write_phot_fits_band_only(path, n_rows=4, vec_width=3):
    """PHOT-like FITS where the filter column is named BAND instead of FLT."""
    t = Table()
    t["MJD"] = np.linspace(50000.0, 60000.0, n_rows).astype(np.float32)
    t["FLUXCAL"] = np.linspace(1.0, 10.0, n_rows).astype(np.float32)
    t["FLUXCALERR"] = np.full(n_rows, 0.1, dtype=np.float32)
    t["BAND"] = np.array(["r "] * n_rows, dtype="S2")
    t["PSF_VEC"] = np.zeros((n_rows, vec_width), dtype=np.float32)
    t.write(str(path), format="fits", overwrite=True)


class TestPhotFITSMultidim:
    """The PHOT branch of process_single_FITS passes a column whitelist
    into load_pandas_from_fit. These tests exercise the same call shape."""

    def test_phot_columns_whitelist_drops_multidim(self, tmp_path, capsys):
        fits_path = tmp_path / "DES_PHOT.fits"
        _write_phot_fits_with_multidim(fits_path)

        phot_columns = ["MJD", "FLUXCAL", "FLUXCALERR", "FLT", "BAND"]
        df = data_utils.load_pandas_from_fit(
            str(fits_path), columns=phot_columns
        )

        # Scalar PHOT cols survive; multi-D cols never made it in
        assert sorted(df.columns) == sorted(
            ["MJD", "FLUXCAL", "FLUXCALERR", "FLT"]
        )
        assert "PSF_VEC" not in df.columns
        assert "APER_FLUX" not in df.columns
        # No warning because the multi-D cols were filtered out at subset
        # step, never reaching the ndim check.
        captured = capsys.readouterr().out
        assert "Skipping multi-dim" not in captured

    def test_phot_band_alt_name_silently_ignored_when_missing(self, tmp_path):
        """Passing BAND alongside FLT is safe even when only FLT exists."""
        fits_path = tmp_path / "DES_PHOT.fits"
        _write_phot_fits_with_multidim(fits_path)

        phot_columns = ["MJD", "FLUXCAL", "FLUXCALERR", "FLT", "BAND"]
        df = data_utils.load_pandas_from_fit(
            str(fits_path), columns=phot_columns
        )

        assert "FLT" in df.columns
        assert "BAND" not in df.columns

    def test_phot_band_only_file_returns_band(self, tmp_path):
        """When the FITS only has BAND, the helper returns BAND."""
        fits_path = tmp_path / "DES_PHOT.fits"
        _write_phot_fits_band_only(fits_path)

        phot_columns = ["MJD", "FLUXCAL", "FLUXCALERR", "FLT", "BAND"]
        df = data_utils.load_pandas_from_fit(
            str(fits_path), columns=phot_columns
        )

        # FLT is absent in the file -> not in df; downstream code does
        # df.rename(columns={"BAND": "FLT"}) to recover.
        assert "BAND" in df.columns
        assert "FLT" not in df.columns

    def test_phot_reject_column_forwarded(self, tmp_path):
        """If settings.phot_reject is set, that column is included."""
        fits_path = tmp_path / "DES_PHOT.fits"
        # Augment the synthetic table with a PHOTFLAG-like scalar column.
        t = Table()
        t["MJD"] = np.linspace(50000.0, 60000.0, 4).astype(np.float32)
        t["FLUXCAL"] = np.ones(4, dtype=np.float32)
        t["FLUXCALERR"] = np.full(4, 0.1, dtype=np.float32)
        t["FLT"] = np.array(["g "] * 4, dtype="S2")
        t["PHOTFLAG"] = np.array([0, 1, 0, 4], dtype=np.int32)
        t["PSF_VEC"] = np.zeros((4, 3), dtype=np.float32)
        t.write(str(fits_path), format="fits", overwrite=True)

        phot_columns = ["MJD", "FLUXCAL", "FLUXCALERR", "FLT", "BAND", "PHOTFLAG"]
        df = data_utils.load_pandas_from_fit(
            str(fits_path), columns=phot_columns
        )

        assert "PHOTFLAG" in df.columns
        assert "PSF_VEC" not in df.columns
        assert df["PHOTFLAG"].tolist() == [0, 1, 0, 4]


class TestProcessHeaderFITSWithMultidim:
    """End-to-end: process_header_FITS should not crash on multi-D HEAD."""

    def test_reads_header_with_multidim_columns(self, tmp_path):
        fits_path = tmp_path / "DES_HEAD.fits"
        _write_fits_with_multidim(fits_path)
        settings = _make_settings()

        df = data_utils.process_header_FITS(
            str(fits_path),
            settings,
            columns=["SNID", "target_2classes", "SNTYPE"],
        )

        assert sorted(df.columns) == sorted(
            ["SNID", "target_2classes", "SNTYPE"]
        )
        assert len(df) == 5
        # All rows are SNTYPE=101 -> mapped to "Ia" -> binary class 0
        assert (df["target_2classes"] == 0).all()

    def test_no_columns_arg_still_drops_multidim(self, tmp_path):
        """Without an explicit columns whitelist, multi-D cols are dropped."""
        fits_path = tmp_path / "DES_HEAD.fits"
        _write_fits_with_multidim(fits_path)
        settings = _make_settings()

        df = data_utils.process_header_FITS(str(fits_path), settings)

        assert "PSF_SIG" not in df.columns
        assert "FLUXCAL_ARR" not in df.columns
        assert "SNID" in df.columns
        assert "target_2classes" in df.columns
