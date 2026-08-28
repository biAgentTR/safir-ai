import pytest
import math
from pydantic import ValidationError
from src.vlm.schemas import VLMSceneObservation
from src.vlm.time_normalizer import normalize_observation_time

def test_time_normalizer_normal():
    obs = VLMSceneObservation(observed_label="test", relative_start_sec=13.5, relative_end_sec=20.0)
    norm = normalize_observation_time(obs, 60.0, 30.0)
    assert norm.global_start_sec == 73.5
    assert norm.global_end_sec == 80.0
    assert norm.time_status == "valid"

def test_time_normalizer_zero_offset():
    obs = VLMSceneObservation(observed_label="test", relative_start_sec=5.0, relative_end_sec=10.0)
    norm = normalize_observation_time(obs, 0.0, 30.0)
    assert norm.global_start_sec == 5.0
    assert norm.global_end_sec == 10.0

def test_time_normalizer_invalid_start_greater_than_end():
    with pytest.raises(ValidationError):
        VLMSceneObservation(observed_label="test", relative_start_sec=10.0, relative_end_sec=5.0)

def test_time_normalizer_invalid_nan():
    obs = VLMSceneObservation(observed_label="test", relative_start_sec=float("nan"), relative_end_sec=5.0)
    norm = normalize_observation_time(obs, 0.0, 30.0)
    assert norm.time_status == "invalid"

def test_time_normalizer_invalid_pos_inf():
    with pytest.raises(ValidationError):
        VLMSceneObservation(observed_label="test", relative_start_sec=float("inf"), relative_end_sec=5.0)

def test_time_normalizer_invalid_neg_inf():
    obs = VLMSceneObservation(observed_label="test", relative_start_sec=float("-inf"), relative_end_sec=5.0)
    norm = normalize_observation_time(obs, 0.0, 30.0)
    assert norm.time_status == "invalid"

def test_time_normalizer_invalid_large_negative():
    obs = VLMSceneObservation(observed_label="test", relative_start_sec=-5.0, relative_end_sec=5.0)
    norm = normalize_observation_time(obs, 0.0, 30.0)
    assert norm.time_status == "invalid"

def test_time_normalizer_invalid_large_exceed():
    obs = VLMSceneObservation(observed_label="test", relative_start_sec=10.0, relative_end_sec=40.0)
    norm = normalize_observation_time(obs, 0.0, 30.0)
    assert norm.time_status == "invalid"

def test_time_normalizer_missing():
    obs = VLMSceneObservation(observed_label="test", relative_start_sec=None, relative_end_sec=None)
    norm = normalize_observation_time(obs, 0.0, 30.0)
    assert norm.time_status == "missing"
    assert norm.global_start_sec is None
    assert norm.global_end_sec is None

def test_time_normalizer_only_start():
    obs = VLMSceneObservation(observed_label="test", relative_start_sec=10.0, relative_end_sec=None)
    norm = normalize_observation_time(obs, 0.0, 30.0)
    assert norm.time_status == "valid"
    assert norm.normalized_relative_end_sec == 10.0
    assert norm.was_adjusted == True

def test_time_normalizer_only_end():
    obs = VLMSceneObservation(observed_label="test", relative_start_sec=None, relative_end_sec=10.0)
    norm = normalize_observation_time(obs, 0.0, 30.0)
    assert norm.time_status == "invalid"

def test_time_normalizer_adjusted_small_negative():
    obs = VLMSceneObservation(observed_label="test", relative_start_sec=-1.0, relative_end_sec=10.0)
    norm = normalize_observation_time(obs, 0.0, 30.0)
    assert norm.time_status == "valid"
    assert norm.normalized_relative_start_sec == 0.0
    assert norm.was_adjusted == True

def test_time_normalizer_adjusted_small_exceed():
    obs = VLMSceneObservation(observed_label="test", relative_start_sec=10.0, relative_end_sec=31.0)
    norm = normalize_observation_time(obs, 0.0, 30.0)
    assert norm.time_status == "valid"
    assert norm.normalized_relative_end_sec == 30.0
    assert norm.was_adjusted == True
    
def test_time_normalizer_preserves_original():
    obs = VLMSceneObservation(observed_label="test", relative_start_sec=-1.0, relative_end_sec=31.0)
    norm = normalize_observation_time(obs, 0.0, 30.0)
    assert norm.original_relative_start_sec == -1.0
    assert norm.original_relative_end_sec == 31.0
    assert norm.normalized_relative_start_sec == 0.0
    assert norm.normalized_relative_end_sec == 30.0
    assert len(norm.adjustment_reasons) > 0

def test_time_normalizer_duration_none():
    obs = VLMSceneObservation(observed_label='test', relative_start_sec=10.0, relative_end_sec=20.0)
    norm = normalize_observation_time(obs, 5.0, None)
    assert norm.time_status == 'valid'
    assert norm.global_start_sec == 15.0
    assert norm.global_end_sec == 25.0

