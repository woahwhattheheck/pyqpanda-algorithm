"""Deutsch-Jozsa example: distinguish constant and balanced affine oracles."""

from pyqpanda3.core import CPUQVM, QProg

from pyqpanda_alg.DeutschJozsa import DeutschJozsa, affine_oracle


def classify_oracle(mask, bias=0):
    """Execute one Deutsch-Jozsa oracle instance and return its promise class."""
    qvm = CPUQVM()
    qubits = QProg(len(mask) + 1).qubits()
    data_qubits = qubits[:-1]
    target_qubit = qubits[-1]

    algorithm = DeutschJozsa(
        n_inputs=len(mask),
        oracle=affine_oracle(mask, bias=bias),
    )

    program = QProg()
    program << algorithm.cir(data_qubits, target_qubit)
    qvm.run(program, shots=256)

    probabilities = qvm.result().get_prob_dict(data_qubits)
    dominant_state = max(probabilities, key=probabilities.get)
    return DeutschJozsa.classify(dominant_state), probabilities


if __name__ == "__main__":
    constant_class, constant_probabilities = classify_oracle([0, 0, 0], bias=1)
    balanced_class, balanced_probabilities = classify_oracle([1, 1, 0], bias=0)

    print("constant oracle:", constant_class, constant_probabilities)
    print("balanced oracle:", balanced_class, balanced_probabilities)
