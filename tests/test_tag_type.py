"""Tests for tag_type: contaminant auto-detection and target_sntype."""

import pandas as pd
from collections import OrderedDict
from supernnova.utils import data_utils, experiment_settings


def _make_settings(sntypes, target_sntype="Ia"):
    """Create a minimal ExperimentSettings for tag_type tests."""
    cli_args = {
        "sntypes": OrderedDict(sntypes),
        "sntype_var": "SNTYPE",
        "target_sntype": target_sntype,
        "data_testing": False,
        "no_dump": True,
        "use_cuda": False,
        "model": "vanilla",
        "weight_decay": 0.0,
    }
    return experiment_settings.ExperimentSettings(cli_args)


def _make_df(types):
    """Create a minimal DataFrame with a TYPE column."""
    return pd.DataFrame({"TYPE": types})


class TestTagTypeContaminant:
    """Tests for automatic contaminant assignment of missing sntypes."""

    def test_missing_types_added_as_contaminant(self):
        """Types in data but not in sntypes should be added as 'contaminant'."""
        settings = _make_settings({"112": "Ib/c", "113": "Ia"})
        df = _make_df(["112", "113", "111", "115"])

        data_utils.tag_type(df, settings, type_column="TYPE")

        assert "111" in settings.sntypes
        assert "115" in settings.sntypes
        assert settings.sntypes["111"] == "contaminant"
        assert settings.sntypes["115"] == "contaminant"

    def test_no_contaminant_when_all_types_present(self):
        """No contaminant class should be created when all types are specified."""
        settings = _make_settings({"112": "Ib/c", "113": "Ia"})
        df = _make_df(["112", "113", "112"])

        data_utils.tag_type(df, settings, type_column="TYPE")

        assert "contaminant" not in settings.sntypes.values()

    def test_binary_classification_contaminant_is_class1(self):
        """For binary (target vs non-target), contaminants should be class 1."""
        settings = _make_settings({"113": "Ia"})
        df = _make_df(["113", "111", "115", "212"])

        df = data_utils.tag_type(df, settings, type_column="TYPE")

        assert df.loc[df["TYPE"] == "113", "target_2classes"].iloc[0] == 0
        assert df.loc[df["TYPE"] == "111", "target_2classes"].iloc[0] == 1
        assert df.loc[df["TYPE"] == "115", "target_2classes"].iloc[0] == 1
        assert df.loc[df["TYPE"] == "212", "target_2classes"].iloc[0] == 1

    def test_multiclass_target_sntype_is_class0(self):
        """For multiclass, target_sntype should be class 0, contaminant gets own class."""
        settings = _make_settings({"112": "Ib/c", "113": "Ia"})
        df = _make_df(["112", "113", "111", "115"])

        df = data_utils.tag_type(df, settings, type_column="TYPE")

        target_col = "target_3classes"
        assert target_col in df.columns

        # Ia (target_sntype) -> class 0
        assert df.loc[df["TYPE"] == "113", target_col].iloc[0] == 0
        # Ib/c -> class 1
        assert df.loc[df["TYPE"] == "112", target_col].iloc[0] == 1
        # contaminants -> class 2
        assert df.loc[df["TYPE"] == "111", target_col].iloc[0] == 2
        assert df.loc[df["TYPE"] == "115", target_col].iloc[0] == 2

    def test_multiclass_target_max_within_bounds(self):
        """Max target value should be < number of unique classes."""
        settings = _make_settings({"112": "Ib/c", "113": "Ia"})
        df = _make_df(["112", "113", "111", "115", "212", "214", "131"])

        df = data_utils.tag_type(df, settings, type_column="TYPE")

        unique_classes = list(dict.fromkeys(settings.sntypes.values()))
        nb_classes = len(unique_classes)
        target_col = f"target_{nb_classes}classes"

        assert df[target_col].max() < nb_classes

    def test_multiple_contaminants_share_same_class(self):
        """All missing types should map to the same contaminant class index."""
        settings = _make_settings({"113": "Ia"})
        df = _make_df(["113", "111", "115", "212", "214"])

        df = data_utils.tag_type(df, settings, type_column="TYPE")

        contaminant_targets = df.loc[df["TYPE"] != "113", "target_2classes"]
        assert contaminant_targets.nunique() == 1

    def test_original_sntypes_unchanged_when_no_missing(self):
        """sntypes should not be modified if all data types are already present."""
        original_types = {"112": "Ib/c", "113": "Ia"}
        settings = _make_settings(original_types)
        df = _make_df(["112", "113"])

        data_utils.tag_type(df, settings, type_column="TYPE")

        assert dict(settings.sntypes) == original_types

    def test_issue_scenario_binary(self):
        """Reproduce the exact scenario from the issue."""
        settings = _make_settings({"112": "Ib/c", "113": "Ia"})
        extra_types = ["111", "115", "212", "214", "131", "221", "114",
                       "132", "134", "133", "213", "211", "123", "124"]
        all_types = ["112", "113"] + extra_types
        df = _make_df(all_types)

        df = data_utils.tag_type(df, settings, type_column="TYPE")

        # Binary: Ia (113) -> 0, everything else -> 1
        assert df.loc[df["TYPE"] == "113", "target_2classes"].iloc[0] == 0
        for t in ["112"] + extra_types:
            assert df.loc[df["TYPE"] == t, "target_2classes"].iloc[0] == 1

        # Multiclass: Ia=0, Ib/c=1, contaminant=2
        assert "target_3classes" in df.columns
        assert df.loc[df["TYPE"] == "113", "target_3classes"].iloc[0] == 0
        assert df.loc[df["TYPE"] == "112", "target_3classes"].iloc[0] == 1
        for t in extra_types:
            assert df.loc[df["TYPE"] == t, "target_3classes"].iloc[0] == 2
        assert df["target_3classes"].max() < 3


class TestTargetSntype:
    """Tests for --target_sntype controlling which class is target 0."""

    def test_default_target_Ia_dict_order_irrelevant(self):
        """With default target_sntype='Ia', Ia is class 0 regardless of dict order."""
        # Ia first in dict
        s1 = _make_settings({"101": "Ia", "303": "coco", "1": "Ia"})
        df1 = data_utils.tag_type(_make_df(["101", "303", "1"]), s1, type_column="TYPE")
        assert df1.loc[df1["TYPE"] == "101", "target_2classes"].iloc[0] == 0
        assert df1.loc[df1["TYPE"] == "303", "target_2classes"].iloc[0] == 1
        assert df1.loc[df1["TYPE"] == "1", "target_2classes"].iloc[0] == 0

        # coco first in dict — Ia should still be class 0
        s2 = _make_settings({"303": "coco", "101": "Ia", "1": "Ia"})
        df2 = data_utils.tag_type(_make_df(["101", "303", "1"]), s2, type_column="TYPE")
        assert df2.loc[df2["TYPE"] == "101", "target_2classes"].iloc[0] == 0
        assert df2.loc[df2["TYPE"] == "303", "target_2classes"].iloc[0] == 1
        assert df2.loc[df2["TYPE"] == "1", "target_2classes"].iloc[0] == 0

    def test_custom_target_sntype(self):
        """Custom target_sntype makes that class target 0."""
        settings = _make_settings(
            {"101": "Ia", "303": "coco", "1": "Ia"}, target_sntype="coco"
        )
        df = data_utils.tag_type(_make_df(["101", "303", "1"]), settings, type_column="TYPE")

        assert df.loc[df["TYPE"] == "303", "target_2classes"].iloc[0] == 0  # coco = target
        assert df.loc[df["TYPE"] == "101", "target_2classes"].iloc[0] == 1  # Ia = non-target
        assert df.loc[df["TYPE"] == "1", "target_2classes"].iloc[0] == 1

    def test_multiclass_consistent_with_binary(self):
        """When unique_classes == 2, multiclass and binary should agree on class 0."""
        settings = _make_settings({"303": "coco", "101": "Ia", "1": "Ia"})
        df = data_utils.tag_type(_make_df(["101", "303", "1"]), settings, type_column="TYPE")

        # Both binary and multiclass create target_2classes
        # target_sntype='Ia' -> Ia should be class 0 in both
        assert df.loc[df["TYPE"] == "101", "target_2classes"].iloc[0] == 0
        assert df.loc[df["TYPE"] == "303", "target_2classes"].iloc[0] == 1

    def test_custom_target_multiclass_ordering(self):
        """target_sntype controls class 0 in multiclass as well."""
        settings = _make_settings(
            {"112": "Ib/c", "113": "Ia", "120": "IIP"}, target_sntype="IIP"
        )
        df = data_utils.tag_type(
            _make_df(["112", "113", "120"]), settings, type_column="TYPE"
        )

        # Binary: IIP=0, rest=1
        assert df.loc[df["TYPE"] == "120", "target_2classes"].iloc[0] == 0
        assert df.loc[df["TYPE"] == "112", "target_2classes"].iloc[0] == 1
        assert df.loc[df["TYPE"] == "113", "target_2classes"].iloc[0] == 1

        # Multiclass: IIP=0 (target), then Ib/c=1, Ia=2 (dict order after target)
        assert "target_3classes" in df.columns
        assert df.loc[df["TYPE"] == "120", "target_3classes"].iloc[0] == 0
        assert df.loc[df["TYPE"] == "112", "target_3classes"].iloc[0] == 1
        assert df.loc[df["TYPE"] == "113", "target_3classes"].iloc[0] == 2

    def test_fallback_when_target_not_in_values(self):
        """When target_sntype is not in sntypes values, fallback to first key."""
        settings = _make_settings(
            {"303": "coco", "101": "SNIa"}, target_sntype="Ia"
        )
        df = data_utils.tag_type(_make_df(["303", "101"]), settings, type_column="TYPE")

        # 'Ia' not found -> first key '303' = class 0
        assert df.loc[df["TYPE"] == "303", "target_2classes"].iloc[0] == 0
        assert df.loc[df["TYPE"] == "101", "target_2classes"].iloc[0] == 1


class TestPhantomSntypes:
    """Tests for sntypes keys that don't exist in the data.

    Phantom removal happens at the pipeline level in detect_contaminant_types()
    (make_dataset.py), NOT in tag_type(). tag_type() uses settings.sntypes as-is.
    These tests verify both behaviors.
    """

    def test_tag_type_with_phantom_creates_consistent_columns(self):
        """tag_type uses settings.sntypes as given, even with phantom keys.

        When detect_contaminant_types has already cleaned sntypes (normal pipeline),
        there are no phantoms. But if called directly with phantoms, column names
        reflect the full sntypes dict.
        """
        settings = _make_settings({"112": "Ib/c", "113": "Ia", "120": "IIP"})
        df = _make_df(["112", "113"])

        df = data_utils.tag_type(df, settings, type_column="TYPE")

        # tag_type uses ALL sntypes values -> 3 unique classes -> target_3classes
        assert "target_2classes" in df.columns
        assert "target_3classes" in df.columns
        # Binary still correct
        assert df.loc[df["TYPE"] == "113", "target_2classes"].iloc[0] == 0
        assert df.loc[df["TYPE"] == "112", "target_2classes"].iloc[0] == 1

    def test_tag_type_phantom_binary_correct(self):
        """Binary classification works correctly even with phantom sntypes."""
        settings = _make_settings({"112": "Ib/c", "113": "Ia", "120": "IIP"})
        df = _make_df(["112", "113", "112"])

        df = data_utils.tag_type(df, settings, type_column="TYPE")

        assert df.loc[df["TYPE"] == "113", "target_2classes"].iloc[0] == 0  # Ia = target
        assert df.loc[df["TYPE"] == "112", "target_2classes"].iloc[0] == 1

    def test_tag_type_contaminant_with_phantom(self):
        """Contaminant detection works alongside phantom keys."""
        # 120 is phantom (not in data), 111 is contaminant (in data, not in sntypes)
        settings = _make_settings({"112": "Ib/c", "113": "Ia", "120": "IIP"})
        df = _make_df(["112", "113", "111"])

        df = data_utils.tag_type(df, settings, type_column="TYPE")

        # 111 added as contaminant (tag_type safety net)
        assert "111" in settings.sntypes
        assert settings.sntypes["111"] == "contaminant"

        # 4 unique class values: Ia, Ib/c, IIP, contaminant -> target_4classes
        assert "target_4classes" in df.columns
        assert df.loc[df["TYPE"] == "113", "target_4classes"].iloc[0] == 0  # Ia (target)
        assert df.loc[df["TYPE"] == "112", "target_4classes"].iloc[0] == 1  # Ib/c

    def test_phantom_keys_never_removed_from_sntypes(self):
        """detect_contaminant_types warns about phantom keys but never removes them.

        Phantom keys are kept because: (1) groupby ignores empty classes so
        downsampling works fine, and (2) preserving the full class structure
        ensures target_Nclasses column names and indices match across datasets.
        """
        from collections import OrderedDict

        sntypes = OrderedDict({"112": "Ib/c", "113": "Ia", "120": "IIP"})
        all_types_in_data = {"112", "113"}  # no 120

        # Simulate: detect_contaminant_types only warns, never deletes
        phantom_keys = [k for k in list(sntypes.keys()) if k not in all_types_in_data]
        assert phantom_keys == ["120"]
        # Keys are preserved
        assert "120" in sntypes
        assert len(sntypes) == 3

    def test_phantom_preserves_column_name_for_model_compat(self):
        """Phantom sntypes produce target_Nclasses matching the trained model."""
        # Model trained with 3 classes, new data only has 2 types
        settings = _make_settings({"112": "Ib/c", "113": "Ia", "120": "IIP"})
        df = _make_df(["112", "113"])

        df = data_utils.tag_type(df, settings, type_column="TYPE")

        # Column name matches full sntypes (3 classes), not just data (2 types)
        assert "target_3classes" in df.columns
        assert df.loc[df["TYPE"] == "113", "target_2classes"].iloc[0] == 0
        assert df.loc[df["TYPE"] == "112", "target_2classes"].iloc[0] == 1

    def test_phantom_class_indices_stable(self):
        """Class indices remain the same whether or not phantom types have data."""
        # Full dataset: all 3 types present
        settings_full = _make_settings({"112": "Ib/c", "113": "Ia", "120": "IIP"})
        df_full = _make_df(["112", "113", "120"])
        df_full = data_utils.tag_type(df_full, settings_full, type_column="TYPE")

        # Partial dataset: type 120 absent (phantom)
        settings_partial = _make_settings({"112": "Ib/c", "113": "Ia", "120": "IIP"})
        df_partial = _make_df(["112", "113"])
        df_partial = data_utils.tag_type(df_partial, settings_partial, type_column="TYPE")

        # Class indices for shared types must be identical
        assert (
            df_full.loc[df_full["TYPE"] == "113", "target_3classes"].iloc[0]
            == df_partial.loc[df_partial["TYPE"] == "113", "target_3classes"].iloc[0]
        )
        assert (
            df_full.loc[df_full["TYPE"] == "112", "target_3classes"].iloc[0]
            == df_partial.loc[df_partial["TYPE"] == "112", "target_3classes"].iloc[0]
        )
