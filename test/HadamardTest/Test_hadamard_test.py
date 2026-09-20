import math

import pytest

from pyqpanda_alg import HadamardTest


def test_reference_probabilities_decode_real_and_imag_components():
    expectation = 0.6 + 0.8j

    real = HadamardTest.reference_probabilities(expectation, "real")
    imag = HadamardTest.reference_probabilities(expectation, "imag")

    assert real == pytest.approx((0.8, 0.2))
    assert imag == pytest.approx((0.9, 0.1))
    assert HadamardTest.decode_component(real) == pytest.approx(0.6)
    assert HadamardTest.decode_component(imag) == pytest.approx(0.8)
    assert HadamardTest.combine_components(
        HadamardTest.decode_component(real),
        HadamardTest.decode_component(imag),
    ) == pytest.approx(expectation)


def test_reference_extremes():
    assert HadamardTest.reference_probabilities(1.0, "real") == (1.0, 0.0)
    assert HadamardTest.reference_probabilities(-1.0, "real") == (0.0, 1.0)
    assert HadamardTest.decode_component((3, 1)) == pytest.approx(0.5)


def test_validation_is_fail_closed():
    with pytest.raises(ValueError, match="component"):
        HadamardTest.reference_probabilities(0.0, "magnitude")
    with pytest.raises(ValueError, match="magnitude"):
        HadamardTest.reference_probabilities(1.01 + 0j, "real")
    with pytest.raises(ValueError, match="finite"):
        HadamardTest.reference_probabilities(complex(math.nan, 0), "real")
    with pytest.raises(ValueError, match="two"):
        HadamardTest.decode_component((1.0,))
    with pytest.raises(ValueError, match="non-negative"):
        HadamardTest.decode_component((1.0, -1.0))
    with pytest.raises(ValueError, match="positive mass"):
        HadamardTest.decode_component((0.0, 0.0))


def test_cpuqvm_z_expectation_when_pyqpanda3_is_available():
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

    real = HadamardTest.run_hadamard_test(
        z_unitary,
        target_width=1,
        prepare_state=prepare_one,
        component="real",
    )
    imag = HadamardTest.run_hadamard_test(
        z_unitary,
        target_width=1,
        prepare_state=prepare_one,
        component="imag",
    )

    assert real.value == pytest.approx(-1.0, abs=1e-9)
    assert imag.value == pytest.approx(0.0, abs=1e-9)
