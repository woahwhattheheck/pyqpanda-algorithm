"""Hadamard Test example for the real expectation of Z on |1>."""

from pyqpanda3.core import QCircuit, X, Z

from pyqpanda_alg import HadamardTest


def prepare_one(qubits):
    circuit = QCircuit()
    circuit << X(qubits[0])
    return circuit


def z_unitary(qubits):
    circuit = QCircuit()
    circuit << Z(qubits[0])
    return circuit


if __name__ == "__main__":
    result = HadamardTest.run_hadamard_test(
        z_unitary,
        target_width=1,
        prepare_state=prepare_one,
        component="real",
    )
    print("Re <1|Z|1>:", result.value)
    print("P(0), P(1):", result.probabilities)
