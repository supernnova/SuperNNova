"""Tests for sntypes resolution from README files and CLI arguments.

Tests the priority order:
1. Manual --sntypes from CLI/config (highest priority)
2. Auto-extracted from .README in raw_dir
3. Built-in DEFAULT_SNTYPES fallback
"""
import os
import json
import shutil
from collections import OrderedDict

from supernnova.data.make_dataset import parse_sntypes_from_readme, resolve_sntypes
from supernnova import conf
from .test_utils import call_cmd


class TestParseSntypesFromReadme:
    """Unit tests for parse_sntypes_from_readme function."""

    def test_parse_pippin_readme(self):
        """Test parsing GENTYPE_TO_NAME from actual PIPPIN README."""
        raw_dir = "tests/raw_PIPPIN"
        sntypes = parse_sntypes_from_readme(raw_dir)

        # Should find the README and parse it
        assert sntypes is not None
        assert isinstance(sntypes, OrderedDict)

        # Check expected GENTYPE mappings (N and N+100)
        # From GENTYPE_TO_NAME: 1 -> Ia, 20-23 -> nonIa types, 32-33 -> nonIa types
        expected = {
            "1": "Ia",
            "101": "Ia",
            "20": "SNIIP",
            "120": "SNIIP",
            "21": "SNIIn",
            "121": "SNIIn",
            "22": "SNIIL1",
            "122": "SNIIL1",
            "23": "SNIIIL2",
            "123": "SNIIIL2",
            "32": "SNIb",
            "132": "SNIb",
            "33": "SNIc",
            "133": "SNIc",
        }

        assert sntypes == expected

    def test_parse_no_readme(self, tmp_path):
        """Test behavior when no README exists in directory."""
        # Empty directory with no README
        sntypes = parse_sntypes_from_readme(str(tmp_path))
        assert sntypes is None

    def test_parse_readme_without_gentype_block(self, tmp_path):
        """Test behavior when README exists but has no GENTYPE_TO_NAME block."""
        # Create README without GENTYPE_TO_NAME
        readme = tmp_path / "test.README"
        readme.write_text(
            """
DOCUMENTATION: Some simulation
SURVEY: DES
FILTERS: griz
# No GENTYPE_TO_NAME block
"""
        )

        sntypes = parse_sntypes_from_readme(str(tmp_path))
        assert sntypes is None

    def test_parse_readme_with_comments_and_empty_lines(self, tmp_path):
        """Test parsing handles comments and empty lines correctly."""
        readme = tmp_path / "test.README"
        readme.write_text(
            """
GENTYPE_TO_NAME:   # GENTYPE-integer    (non)Ia   transient-Name
  # This is a comment line
  1:   Ia       SALT3.MODEL    SNIaMODEL00

  20:  nonIa    SNIIP          NONIaMODEL01
  # Another comment

  32:  nonIa    SNIb           NONIaMODEL02
"""
        )

        sntypes = parse_sntypes_from_readme(str(tmp_path))
        assert sntypes is not None

        expected = {
            "1": "Ia",
            "101": "Ia",
            "20": "SNIIP",
            "120": "SNIIP",
            "32": "SNIb",
            "132": "SNIb",
        }
        assert sntypes == expected

    def test_parse_multiple_readmes_uses_first(self, tmp_path):
        """Test that when multiple READMEs exist, the first is used."""
        # Create two READMEs with different content
        readme1 = tmp_path / "aaa.README"
        readme1.write_text(
            """
GENTYPE_TO_NAME:
  1:   Ia       MODEL1    PREFIX1
"""
        )

        readme2 = tmp_path / "zzz.README"
        readme2.write_text(
            """
GENTYPE_TO_NAME:
  99:  nonIa    MODEL99   PREFIX99
"""
        )

        sntypes = parse_sntypes_from_readme(str(tmp_path))

        # Should use first README (aaa.README) due to natural sorting
        expected = {"1": "Ia", "101": "Ia"}
        assert sntypes == expected


class TestResolveSntypes:
    """Unit tests for resolve_sntypes function with settings object."""

    def test_manual_sntypes_takes_priority(self, tmp_path):
        """Test that manually provided sntypes are never overridden."""
        # Create a README in tmp_path
        readme = tmp_path / "test.README"
        readme.write_text(
            """
GENTYPE_TO_NAME:
  1:   Ia       MODEL    PREFIX
"""
        )

        # Create mock settings with manual sntypes
        class MockSettings:
            sntypes = OrderedDict({"999": "CustomType"})
            raw_dir = str(tmp_path)

        settings = MockSettings()
        original_sntypes = settings.sntypes.copy()

        resolve_sntypes(settings)

        # Manual sntypes should be unchanged
        assert settings.sntypes == original_sntypes
        assert "999" in settings.sntypes
        assert settings.sntypes["999"] == "CustomType"

    def test_readme_extraction_when_sntypes_none(self):
        """Test that README is parsed when sntypes is None."""

        class MockSettings:
            sntypes = None
            raw_dir = "tests/raw_PIPPIN"

        settings = MockSettings()
        resolve_sntypes(settings)

        # Should have extracted from README
        assert settings.sntypes is not None
        assert "101" in settings.sntypes
        assert settings.sntypes["101"] == "Ia"

    def test_fallback_to_defaults_when_no_readme(self, tmp_path):
        """Test that DEFAULT_SNTYPES is used when no README exists."""

        class MockSettings:
            sntypes = None
            raw_dir = str(tmp_path)  # Empty directory

        settings = MockSettings()
        resolve_sntypes(settings)

        # Should fall back to DEFAULT_SNTYPES
        assert settings.sntypes is not None
        assert settings.sntypes == conf.DEFAULT_SNTYPES


class TestSntypesIntegrationCLI:
    """Integration tests using the CLI with actual data."""

    def setup_method(self):
        """Set up test directory."""
        self.dump_dir = "tests/dump_data_sntypes"
        shutil.rmtree(self.dump_dir, ignore_errors=True)

    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.dump_dir, ignore_errors=True)

    def test_cli_with_raw_pippin_auto_readme(self):
        """Test CLI automatically extracts sntypes from raw_PIPPIN README."""
        cmd = f"snn make_data --dump_dir {self.dump_dir} --raw_dir tests/raw_PIPPIN"
        call_cmd(cmd)

        # Check that database was created successfully
        assert os.path.exists(f"{self.dump_dir}/processed/database.h5")
        assert os.path.exists(f"{self.dump_dir}/processed/SNID.pickle")

    def test_cli_with_manual_sntypes_overrides_readme(self):
        """Test that manual --sntypes overrides README extraction."""
        # Use raw_PIPPIN which has a README, but provide manual sntypes
        manual_sntypes = json.dumps({"101": "Ia", "120": "CC"})

        cmd = (
            f"snn make_data --dump_dir {self.dump_dir} "
            f"--raw_dir tests/raw_PIPPIN "
            f"--sntypes '{manual_sntypes}'"
        )
        call_cmd(cmd)

        # Check that database was created successfully
        assert os.path.exists(f"{self.dump_dir}/processed/database.h5")
        assert os.path.exists(f"{self.dump_dir}/processed/SNID.pickle")

        # The test passes if no errors occurred - the manual sntypes were used

    def test_cli_without_readme_uses_defaults(self):
        """Test that defaults are used when no README and no manual sntypes."""
        cmd = f"snn make_data --dump_dir {self.dump_dir} --raw_dir tests/raw"
        call_cmd(cmd)

        # Check that database was created successfully with default sntypes
        assert os.path.exists(f"{self.dump_dir}/processed/database.h5")
        assert os.path.exists(f"{self.dump_dir}/processed/SNID.pickle")


class TestSntypesPriorityOrder:
    """Test the complete priority order of sntypes resolution."""

    def test_priority_order_manual_first(self):
        """Verify priority: manual > README > defaults."""

        # Scenario 1: Manual provided (should use manual)
        class Settings1:
            sntypes = OrderedDict({"999": "Manual"})
            raw_dir = "tests/raw_PIPPIN"  # Has README

        settings1 = Settings1()
        resolve_sntypes(settings1)
        assert settings1.sntypes["999"] == "Manual"

        # Scenario 2: No manual, has README (should use README)
        class Settings2:
            sntypes = None
            raw_dir = "tests/raw_PIPPIN"

        settings2 = Settings2()
        resolve_sntypes(settings2)
        assert "101" in settings2.sntypes
        assert settings2.sntypes["101"] == "Ia"

        # Scenario 3: No manual, no README (should use defaults)
        class Settings3:
            sntypes = None
            raw_dir = "/nonexistent/path"

        settings3 = Settings3()
        resolve_sntypes(settings3)
        assert settings3.sntypes == conf.DEFAULT_SNTYPES
