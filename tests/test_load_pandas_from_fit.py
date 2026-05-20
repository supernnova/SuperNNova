"""Tests for ``supernnova.utils.data_utils.load_pandas_from_fit``.

Covers the multi-dimensional-FITS-column fix:

* ``astropy.Table.to_pandas()`` raises on FITS columns with ``ndim > 1``.
  The helper now (A) drops such columns with a warning, and (B) when a
  ``columns`` subset is given, narrows the table first so unwanted multi-dim
  columns can't trigger the failure.

Each test builds a tiny in-memory FITS file in a ``tmp_path`` so the suite
doesn't depend on any external fixtures.
"""
import numpy as np
import pytest
from astropy.io import fits

from supernnova.utils.data_utils import load_pandas_from_fit


def _write_fits(path, *, with_multidim=True):
    """Write a 4-row BinTable FITS at ``path``.

    Columns: SNID (string), SNTYPE (int), FLUXCAL (float). When
    ``with_multidim=True`` an extra PSF column with shape (n, 5) is added —
    this is what historically broke ``Table.to_pandas()``.
    """
    n = 4
    cols = [
        fits.Column(name="SNID", format="3A", array=np.array(["A1", "A2", "A3", "A4"])),
        fits.Column(name="SNTYPE", format="J", array=np.array([1, 2, 1, 3], dtype=np.int32)),
        fits.Column(
            name="FLUXCAL",
            format="E",
            array=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        ),
    ]
    if with_multidim:
        psf = np.arange(n * 5, dtype=np.float32).reshape(n, 5)
        cols.append(fits.Column(name="PSF", format="5E", array=psf))
    fits.BinTableHDU.from_columns(cols).writeto(str(path))


def test_columns_none_drops_multidim(tmp_path, capsys):
    """With no column filter, multi-dim columns are dropped (with a warning)
    instead of crashing ``to_pandas()``."""
    path = tmp_path / "with_psf.fits"
    _write_fits(path, with_multidim=True)

    df = load_pandas_from_fit(str(path))

    assert "PSF" not in df.columns, "multi-dim PSF should be dropped"
    assert {"SNID", "SNTYPE", "FLUXCAL"} <= set(df.columns)
    assert len(df) == 4
    # The helper warns via logging_utils.print_yellow, which prints to stdout.
    out = capsys.readouterr().out
    assert "PSF" in out and "multi-dim" in out


def test_columns_excludes_multidim_silently(tmp_path, capsys):
    """When ``columns`` excludes the multi-dim column, no warning fires
    (the subset step removes it before the multi-dim check)."""
    path = tmp_path / "with_psf.fits"
    _write_fits(path, with_multidim=True)

    df = load_pandas_from_fit(str(path), columns=["SNID", "SNTYPE"])

    assert list(df.columns) == ["SNID", "SNTYPE"]
    out = capsys.readouterr().out
    assert "multi-dim" not in out, "no warning expected when caller filters it out"


def test_columns_missing_names_are_ignored(tmp_path):
    """Requested names that aren't in the file are silently skipped so
    callers can pass alternatives (e.g. both 'FLT' and 'BAND')."""
    path = tmp_path / "with_psf.fits"
    _write_fits(path, with_multidim=True)

    df = load_pandas_from_fit(
        str(path), columns=["SNID", "SNTYPE", "DOES_NOT_EXIST"]
    )

    assert set(df.columns) == {"SNID", "SNTYPE"}


def test_columns_requesting_multidim_still_drops_it(tmp_path, capsys):
    """If the caller does request the multi-dim column, the safety net
    still drops it rather than crashing."""
    path = tmp_path / "with_psf.fits"
    _write_fits(path, with_multidim=True)

    df = load_pandas_from_fit(str(path), columns=["SNID", "PSF"])

    assert "PSF" not in df.columns
    assert list(df.columns) == ["SNID"]
    out = capsys.readouterr().out
    assert "PSF" in out


def test_columns_all_missing_falls_back_to_full_table(tmp_path):
    """If none of the requested names exist, fall back to reading
    everything so the multi-dim drop can still produce a usable frame
    rather than an empty one."""
    path = tmp_path / "with_psf.fits"
    _write_fits(path, with_multidim=True)

    df = load_pandas_from_fit(str(path), columns=["NOPE"])

    # Full table came through (minus the dropped multi-dim column).
    assert "PSF" not in df.columns
    assert {"SNID", "SNTYPE", "FLUXCAL"} <= set(df.columns)


def test_wellformed_fits_unchanged(tmp_path, capsys):
    """Files with no multi-dim columns behave exactly as before: no
    warning, all columns present, values preserved."""
    path = tmp_path / "clean.fits"
    _write_fits(path, with_multidim=False)

    df = load_pandas_from_fit(str(path))

    assert set(df.columns) == {"SNID", "SNTYPE", "FLUXCAL"}
    assert len(df) == 4
    np.testing.assert_array_equal(df["SNTYPE"].values, [1, 2, 1, 3])
    np.testing.assert_allclose(df["FLUXCAL"].values, [10.0, 20.0, 30.0, 40.0])
    out = capsys.readouterr().out
    assert "multi-dim" not in out


def test_columns_preserves_caller_order(tmp_path):
    """The returned DataFrame keeps the column order requested by the caller
    (filtering preserves the caller's list ordering)."""
    path = tmp_path / "clean.fits"
    _write_fits(path, with_multidim=False)

    df = load_pandas_from_fit(str(path), columns=["FLUXCAL", "SNID"])

    assert list(df.columns) == ["FLUXCAL", "SNID"]
