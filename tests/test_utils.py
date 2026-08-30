import pytest
from utils import do_nothing

def test_do_nothing():
    x = 5
    result = do_nothing(x)
    assert result == x