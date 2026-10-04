import pytest

from pyqpanda_alg import QOracleLearning as q


def test_walsh_matches_direct_signed_sums_for_all_small_boolean_tables():
    # Independent O(N squared) definition, including non-affine functions.
    from itertools import product

    for size in (2, 4, 8):
        for table in product((0, 1), repeat=size):
            expected = []
            for y in range(size):
                total = sum(
                    -1 if fx ^ ((x & y).bit_count() & 1) else 1
                    for x, fx in enumerate(table)
                )
                amplitude = total / size
                expected.append(float(amplitude * amplitude))
            assert q.walsh_probabilities(table) == tuple(expected)


def test_walsh_preserves_input_and_rejects_invalid_tables():
    table = [0, 0, 1, 1, 1, 1, 0, 1]
    original = table[:]
    probabilities = q.walsh_probabilities(table)
    assert table == original
    assert isinstance(probabilities, tuple)
    assert all(isinstance(p, float) for p in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)
    for invalid in ([], [0], [0, 1, 0], [0, 2]):
        with pytest.raises(ValueError):
            q.walsh_probabilities(invalid)
