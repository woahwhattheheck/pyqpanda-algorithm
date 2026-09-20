"""Minimal Quantum Phase Estimation example.

The Z eigenstate |1> has eigenvalue -1 = exp(2*pi*i*0.5), so three precision
qubits should identify phase 0.5 exactly.
"""

from pyqpanda3.core import QCircuit, X, Z

from pyqpanda_alg import QPE


def prepare_one(qubits):
    circuit = QCircuit()
    circuit << X(qubits[0])
    return circuit


def z_unitary(qubits):
    circuit = QCircuit()
    circuit << Z(qubits[0])
    return circuit


if __name__ == "__main__":
    result = QPE.run_qpe(
        z_unitary,
        target_width=1,
        precision_bits=3,
        prepare_eigenstate=prepare_one,
    )
    print("phase:", result.phase)
    print("maximum bitstring:", result.bitstring)
    print("probabilities:", result.probability_dict())
