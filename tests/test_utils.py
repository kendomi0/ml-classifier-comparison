import pytest
from utils import get_user_choice, do_nothing

@pytest.mark.parametrize("user_input", ["blobs", "noisy moons", "noisy circles", "anisotropic", "varied"])
def test_get_user_choice_valid(monkeypatch, user_input):
    test_dict = {"blobs": 1, "noisy moons": 2, "noisy circles": 3, "anisotropic": 4, "varied": 5}
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = get_user_choice(test_dict)
    assert result == user_input

@pytest.mark.parametrize("user_input", ["bLOBS", "nOISY MOONS", "NOISY CIRCLES", "aNISOtropic", "VARIed"])
def test_get_user_choice_valid_case_insensitive(monkeypatch, user_input):
    test_dict = {"blobs": 1, "noisy moons": 2, "noisy circles": 3, "anisotropic": 4, "varied": 5}
    monkeypatch.setattr("builtins.input", lambda _: user_input)
    result = get_user_choice(test_dict)
    assert result == user_input.lower()

def test_get_user_choice_invalid_then_valid(monkeypatch):
    test_dict = {"blobs": 1, "noisy moons": 2}
    inputs = iter(["invalid_dataset", "blobs"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    result = get_user_choice(test_dict)
    assert result == "blobs"

def test_do_nothing():
    x = 5
    result = do_nothing(x)
    assert result == x