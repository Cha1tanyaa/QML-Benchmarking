import pytest
import sys
from pathlib import Path

path_to_add_to_sys = Path(__file__).resolve().parents[3]
if str(path_to_add_to_sys) not in sys.path:
    sys.path.insert(0, str(path_to_add_to_sys))


from paper_extension.benchmarks.additions_hyperparameter_search import filter_compatible_models

# Define common test data
ALL_MODELS_WITH_SETTINGS = ["ImgA", "ImgB", "GPA", "GPB", "SeqA", "SeqB", "ExtraModel", "OnlyInAllSettings"]
IMAGE_MODELS = ["ImgA", "ImgB", "GPA"]  # GPA is also general purpose
GENERAL_PURPOSE_MODELS = ["GPA", "GPB", "ExtraModel"]
SEQUENCE_MODELS = ["SeqA", "SeqB", "GPB"] # GPB is also general purpose
SYNTHETIC_DATASET_STEMS = ["synth", "synthetic_dataset"]

def test_filter_bars_and_stripes():
    dataset_stem = "bars_and_stripes_data"
    # Expected: Models from IMAGE_MODELS or GENERAL_PURPOSE_MODELS that are in ALL_MODELS_WITH_SETTINGS
    expected = sorted(["ImgA", "ImgB", "GPA", "GPB", "ExtraModel"])
    actual = filter_compatible_models(dataset_stem, ALL_MODELS_WITH_SETTINGS, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual) == expected

def test_filter_synthetic_dataset():
    dataset_stem = "my_synth_data_test" # "synth" is in SYNTHETIC_DATASET_STEMS
    # Expected: Models from GENERAL_PURPOSE_MODELS that are in ALL_MODELS_WITH_SETTINGS
    expected = sorted(["GPA", "GPB", "ExtraModel"])
    actual = filter_compatible_models(dataset_stem, ALL_MODELS_WITH_SETTINGS, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual) == expected

def test_filter_stock_dataset():
    dataset_stem = "stock_prices_daily"
    # Expected: Models from SEQUENCE_MODELS or GENERAL_PURPOSE_MODELS that are in ALL_MODELS_WITH_SETTINGS
    expected = sorted(["SeqA", "SeqB", "GPB", "GPA", "ExtraModel"])
    actual = filter_compatible_models(dataset_stem, ALL_MODELS_WITH_SETTINGS, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual) == expected

def test_filter_creditcard_dataset():
    dataset_stem = "creditcard_fraud_detection"
    # Expected: Models from GENERAL_PURPOSE_MODELS that are in ALL_MODELS_WITH_SETTINGS
    expected = sorted(["GPA", "GPB", "ExtraModel"])
    actual = filter_compatible_models(dataset_stem, ALL_MODELS_WITH_SETTINGS, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual) == expected

def test_filter_unknown_dataset_type():
    dataset_stem = "unknown_dataset_type_XYZ"
    # Expected: Models from GENERAL_PURPOSE_MODELS (default) that are in ALL_MODELS_WITH_SETTINGS
    expected = sorted(["GPA", "GPB", "ExtraModel"])
    actual = filter_compatible_models(dataset_stem, ALL_MODELS_WITH_SETTINGS, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual) == expected

def test_filter_dataset_stem_case_insensitivity():
    dataset_stem_bs_upper = "BARS_AND_STRIPES_DATA"
    expected_bs = sorted(["ImgA", "ImgB", "GPA", "GPB", "ExtraModel"])
    actual_bs = filter_compatible_models(dataset_stem_bs_upper, ALL_MODELS_WITH_SETTINGS, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual_bs) == expected_bs

    dataset_stem_synth_mixed = "My_Synth_Data_Test" # "synth" (lowercase) is in SYNTHETIC_DATASET_STEMS
    expected_synth = sorted(["GPA", "GPB", "ExtraModel"])
    actual_synth = filter_compatible_models(dataset_stem_synth_mixed, ALL_MODELS_WITH_SETTINGS, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual_synth) == expected_synth

    dataset_stem_stock_mixed = "Stock_Prices_Daily"
    expected_stock = sorted(["SeqA", "SeqB", "GPB", "GPA", "ExtraModel"])
    actual_stock = filter_compatible_models(dataset_stem_stock_mixed, ALL_MODELS_WITH_SETTINGS, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual_stock) == expected_stock

def test_filter_with_restrictive_all_model_names():
    all_models_subset = ["ImgA", "GPA", "SeqA", "OnlyInAllSettings"] # More restrictive list
    
    dataset_stem_bs = "bars_and_stripes_data"
    expected_bs = sorted(["ImgA", "GPA"]) 
    actual_bs = filter_compatible_models(dataset_stem_bs, all_models_subset, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual_bs) == expected_bs

    dataset_stem_synth = "my_synth_data_test"
    expected_synth = sorted(["GPA"])
    actual_synth = filter_compatible_models(dataset_stem_synth, all_models_subset, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual_synth) == expected_synth

    dataset_stem_stock = "stock_prices_daily"
    expected_stock = sorted(["SeqA", "GPA"])
    actual_stock = filter_compatible_models(dataset_stem_stock, all_models_subset, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual_stock) == expected_stock
    
    dataset_stem_unknown = "unknown_type"
    expected_unknown = sorted(["GPA"]) # Only GPA from GENERAL_PURPOSE_MODELS is in all_models_subset
    actual_unknown = filter_compatible_models(dataset_stem_unknown, all_models_subset, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual_unknown) == expected_unknown


def test_filter_empty_all_model_names():
    all_models_empty = []
    dataset_stem = "bars_and_stripes_data"
    expected = []
    actual = filter_compatible_models(dataset_stem, all_models_empty, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual) == expected

def test_filter_empty_specific_model_lists():
    dataset_stem_bs = "bars_and_stripes_data"
    image_models_empty = []
    # Should use GENERAL_PURPOSE_MODELS for bars_and_stripes if IMAGE_MODELS is empty
    expected_bs_no_img = sorted(["GPA", "GPB", "ExtraModel"]) 
    actual_bs_no_img = filter_compatible_models(dataset_stem_bs, ALL_MODELS_WITH_SETTINGS, image_models_empty, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual_bs_no_img) == expected_bs_no_img

    dataset_stem_synth = "my_synth_data_test"
    general_purpose_models_empty = []
    # Synthetic datasets rely on GENERAL_PURPOSE_MODELS
    expected_synth_no_gp = [] 
    actual_synth_no_gp = filter_compatible_models(dataset_stem_synth, ALL_MODELS_WITH_SETTINGS, IMAGE_MODELS, general_purpose_models_empty, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual_synth_no_gp) == expected_synth_no_gp
    
    dataset_stem_unknown = "unknown_dataset_type_XYZ"
    # Unknown defaults to general purpose, which is now empty
    expected_unknown_all_empty = []
    actual_unknown_all_empty = filter_compatible_models(dataset_stem_unknown, ALL_MODELS_WITH_SETTINGS, [], [], [], SYNTHETIC_DATASET_STEMS)
    assert sorted(actual_unknown_all_empty) == expected_unknown_all_empty

def test_filter_no_overlap_with_all_models():
    all_models_no_overlap = ["Zebra", "Yak"] # No overlap with IMAGE_MODELS, GENERAL_PURPOSE_MODELS etc.
    dataset_stem = "bars_and_stripes_data"
    expected = [] 
    actual = filter_compatible_models(dataset_stem, all_models_no_overlap, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual) == expected

def test_filter_synthetic_partial_match():
    dataset_stem = "main_synthetic_dataset_v2" # "synthetic_dataset" is a substring
    expected = sorted(["GPA", "GPB", "ExtraModel"])
    actual = filter_compatible_models(dataset_stem, ALL_MODELS_WITH_SETTINGS, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual) == expected

def test_filter_synthetic_no_match_if_not_substring_falls_to_general():
    dataset_stem = "completely_different_synth_name" 
    # Does not match synthetic stems, should fall to the 'else' case (general purpose)
    expected = sorted(["GPA", "GPB", "ExtraModel"]) 
    actual = filter_compatible_models(dataset_stem, ALL_MODELS_WITH_SETTINGS, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert sorted(actual) == expected

def test_model_only_in_all_settings_not_categorized():
    # "OnlyInAllSettings" is in ALL_MODELS_WITH_SETTINGS but not in IMAGE_MODELS, GENERAL_PURPOSE_MODELS, or SEQUENCE_MODELS
    # It should not appear unless a category it belongs to is selected.
    dataset_stem = "bars_and_stripes_data"
    # Expected: Models from IMAGE_MODELS or GENERAL_PURPOSE_MODELS that are in ALL_MODELS_WITH_SETTINGS
    # "OnlyInAllSettings" should not be here.
    expected = sorted(["ImgA", "ImgB", "GPA", "GPB", "ExtraModel"])
    actual = filter_compatible_models(dataset_stem, ALL_MODELS_WITH_SETTINGS, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert "OnlyInAllSettings" not in actual
    assert sorted(actual) == expected

    dataset_stem_unknown = "unknown_dataset_type_XYZ"
    # Expected: Models from GENERAL_PURPOSE_MODELS (default) that are in ALL_MODELS_WITH_SETTINGS
    # "OnlyInAllSettings" should not be here.
    expected_unknown = sorted(["GPA", "GPB", "ExtraModel"])
    actual_unknown = filter_compatible_models(dataset_stem_unknown, ALL_MODELS_WITH_SETTINGS, IMAGE_MODELS, GENERAL_PURPOSE_MODELS, SEQUENCE_MODELS, SYNTHETIC_DATASET_STEMS)
    assert "OnlyInAllSettings" not in actual_unknown
    assert sorted(actual_unknown) == expected_unknown

if __name__ == "__main__":
    pytest.main([__file__])