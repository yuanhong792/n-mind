import pytest
from nmind.evaluation.phase1_io import validate_metadata

def test_metadata_missing_field_rejected():
    with pytest.raises(ValueError, match="missing required metadata"):
        validate_metadata({"seed": 1})

def test_metadata_complete_accepted():
    validate_metadata({"torch_version":"x", "torch_build":"cpu", "python_version":"x",
                       "git_revision":"x", "data_fingerprint":"x", "config_hash":"x",
                       "seed":1, "sample_count":2, "effective_sample_count":2})
