import math

import pytest

from pyqpanda_alg import QPE


def test_exact_grid_reference_is_one_hot_and_decodes_exactly():
    for bits in range(1, 7):
        size = 1 << bits
        for outcome in range(size):
            phase = outcome / size
            probabilities = QPE.qpe_reference_probabilities(phase, bits)
            assert sum(probabilities) == pytest.approx(1.0)
            assert probabilities[outcome] == pytest.approx(1.0, abs=1e-12)
            assert QPE.decode_phase(probabilities, bits) == phase


def test_non_grid_phase_decodes_to_nearest_high_probability_bin():
    for bits in (3, 5, 7):
        resolution = 1.0 / (1 << bits)
        for phase in (0.071, 0.247, 0.499, 0.731, 0.997):
            probabilities = QPE.qpe_reference_probabilities(phase, bits)
            estimate = QPE.decode_phase(probabilities, bits)
            assert QPE.circular_phase_distance(estimate, phase) <= resolution
            assert QPE.circular_phase_distance(
                QPE.nearest_phase_grid_point(phase, bits),
                phase,
            ) <= resolution / 2 + 1e-15


def test_phase_periodicity_and_circular_distance():
    assert QPE.canonical_phase(1.25) == pytest.approx(0.25)
    assert QPE.canonical_phase(-0.25) == pytest.approx(0.75)
    assert QPE.qpe_reference_probabilities(0.125, 4) == pytest.approx(
        QPE.qpe_reference_probabilities(1.125, 4)
    )
    assert QPE.circular_phase_distance(0.99, 0.01) == pytest.approx(0.02)


def test_mapping_decoder_uses_binary_integer_convention():
    mapping = {
        "000": 0.0,
        "001": 0.1,
        "100": 0.8,
        "111": 0.1,
    }
    assert QPE.decode_phase_mapping(mapping, 3) == pytest.approx(0.5)


def test_validation_rejects_invalid_inputs():
    for bad in (0, -1, 25, 1.5, True):
        with pytest.raises(ValueError):
            QPE.qpe_reference_probabilities(0.5, bad)

    for bad_phase in (math.nan, math.inf, -math.inf, "not-a-phase"):
        with pytest.raises(ValueError):
            QPE.canonical_phase(bad_phase)

    with pytest.raises(ValueError, match="expected 8"):
        QPE.decode_phase([1.0, 0.0], 3)
    with pytest.raises(ValueError, match="non-negative"):
        QPE.decode_phase([1.0, -1.0, 0.0, 0.0], 2)
    with pytest.raises(ValueError, match="positive mass"):
        QPE.decode_phase([0.0] * 4, 2)
    with pytest.raises(ValueError, match="3-bit"):
        QPE.decode_phase_mapping({"10": 1.0}, 3)


def test_cpuqvm_z_eigenphase_when_pyqpanda3_is_available():
    pytest.importorskip("pyqpanda3")
    from pyqpanda3.core import QCircuit, X, Z

    def prepare_one(qubits):
        circuit = QCircuit()
        circuit << X(qubits[0])
        return circuit

    def z_unitary(qubits):
        circuit = QCircuit()
        circuit << Z(qubits[0])
        return circuit

    result = QPE.run_qpe(
        z_unitary,
        target_width=1,
        precision_bits=3,
        prepare_eigenstate=prepare_one,
    )
    assert QPE.circular_phase_distance(result.phase, 0.5) < 1e-9
    assert result.bitstring == "100"
