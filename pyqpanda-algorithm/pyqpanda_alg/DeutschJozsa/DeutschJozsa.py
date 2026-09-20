# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Deutsch-Jozsa oracle construction and constant/balanced classification.

The Deutsch-Jozsa promise problem asks whether a Boolean function is constant
(the same output for every input) or balanced (0 on exactly half of its
inputs). On an ideal quantum backend the circuit built here resolves that
promise with one oracle query.

Qubit convention
----------------
data_qubits[0] is the least-significant bit of a truth-table index. The final
data-register state 00...0 means constant; every non-zero result means
balanced.
"""

from typing import Callable, Sequence

from pyqpanda3.core import H, QCircuit, X


Promise = str
Oracle = Callable[[Sequence[object], object], QCircuit]


def _binary_sequence(values: Sequence[int], name: str) -> tuple[int, ...]:
    """Normalize a non-empty sequence containing only Boolean / binary values."""
    normalized = tuple(int(value) for value in values)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if any(value not in (0, 1) for value in normalized):
        raise ValueError(f"{name} must contain only 0/1 values")
    return normalized


def validate_truth_table(table: Sequence[int]) -> Promise:
    """Validate a Deutsch-Jozsa truth table and return its promised class."""
    values = _binary_sequence(table, "table")
    size = len(values)
    if size < 2 or size & (size - 1):
        raise ValueError("table length must be a power of two and at least 2")

    ones = sum(values)
    if ones in (0, size):
        return "constant"
    if ones * 2 == size:
        return "balanced"
    raise ValueError(
        "Deutsch-Jozsa promise violated: truth table is neither constant nor balanced"
    )


def affine_oracle(mask: Sequence[int], bias: int = 0) -> Oracle:
    """Create a reversible oracle for f(x) = bias XOR (mask dot x mod 2).

    An all-zero mask is constant. Any non-zero mask is balanced, making this a
    compact way to construct arbitrarily wide promised Deutsch-Jozsa examples.
    """
    mask_bits = _binary_sequence(mask, "mask")
    if int(bias) not in (0, 1):
        raise ValueError("bias must be 0 or 1")
    bias_bit = int(bias)

    def oracle(data_qubits: Sequence[object], target_qubit: object) -> QCircuit:
        data = tuple(data_qubits)
        if len(data) != len(mask_bits):
            raise ValueError(
                f"oracle expects {len(mask_bits)} data qubits, got {len(data)}"
            )

        circuit = QCircuit()
        if bias_bit:
            circuit << X(target_qubit)
        for bit, qubit in zip(mask_bits, data):
            if bit:
                circuit << X(target_qubit).control([qubit])
        return circuit

    oracle.promise = "constant" if not any(mask_bits) else "balanced"
    oracle.mask = mask_bits
    oracle.bias = bias_bit
    oracle.n_inputs = len(mask_bits)
    return oracle


def truth_table_oracle(table: Sequence[int]) -> Oracle:
    """Create a reversible oracle from a promised Boolean truth table.

    The table index is interpreted little-endian with respect to data_qubits.
    For every input row where f(x) == 1, a multi-controlled X targets the
    answer qubit; controls corresponding to zero bits are temporarily inverted.
    """
    values = _binary_sequence(table, "table")
    promise = validate_truth_table(values)
    n_inputs = len(values).bit_length() - 1

    def oracle(data_qubits: Sequence[object], target_qubit: object) -> QCircuit:
        data = tuple(data_qubits)
        if len(data) != n_inputs:
            raise ValueError(f"oracle expects {n_inputs} data qubits, got {len(data)}")

        circuit = QCircuit()
        for basis_index, output in enumerate(values):
            if not output:
                continue

            zero_controls = [
                data[bit]
                for bit in range(n_inputs)
                if ((basis_index >> bit) & 1) == 0
            ]
            for qubit in zero_controls:
                circuit << X(qubit)

            circuit << X(target_qubit).control(list(data))

            for qubit in reversed(zero_controls):
                circuit << X(qubit)

        return circuit

    oracle.promise = promise
    oracle.truth_table = values
    oracle.n_inputs = n_inputs
    return oracle


def build_deutsch_jozsa_circuit(
    data_qubits: Sequence[object],
    target_qubit: object,
    oracle: Oracle,
) -> QCircuit:
    """Build the standard Deutsch-Jozsa circuit for a reversible oracle."""
    data = tuple(data_qubits)
    if not data:
        raise ValueError("at least one data qubit is required")
    if not callable(oracle):
        raise TypeError("oracle must be callable")

    circuit = QCircuit()

    # Prepare |0...0>|1>, then enter the phase-kickback basis.
    circuit << X(target_qubit)
    for qubit in data:
        circuit << H(qubit)
    circuit << H(target_qubit)

    oracle_circuit = oracle(data, target_qubit)
    if not isinstance(oracle_circuit, QCircuit):
        raise TypeError("oracle must return a QCircuit")
    circuit << oracle_circuit

    # 00...0 after the final Hadamards means constant; non-zero means balanced.
    for qubit in data:
        circuit << H(qubit)

    return circuit


def classify_measurement(bitstring: str) -> Promise:
    """Classify one ideal Deutsch-Jozsa data-register measurement."""
    if not isinstance(bitstring, str):
        raise TypeError("bitstring must be a string")
    bits = bitstring.replace(" ", "")
    if not bits or any(bit not in "01" for bit in bits):
        raise ValueError("bitstring must contain one or more binary digits")
    return "constant" if set(bits) == {"0"} else "balanced"


class DeutschJozsa:
    """Reusable Deutsch-Jozsa circuit builder.

    Execute the returned circuit and inspect only the data register. On an ideal
    backend, 00...0 identifies a constant function; every non-zero state
    identifies a balanced function.
    """

    def __init__(self, n_inputs: int, oracle: Oracle):
        if not isinstance(n_inputs, int) or isinstance(n_inputs, bool) or n_inputs < 1:
            raise ValueError("n_inputs must be a positive integer")
        if not callable(oracle):
            raise TypeError("oracle must be callable")
        self.n_inputs = n_inputs
        self.oracle = oracle

    def cir(
        self,
        data_qubits: Sequence[object],
        target_qubit: object,
    ) -> QCircuit:
        """Return the Deutsch-Jozsa circuit over caller-owned qubits."""
        data = tuple(data_qubits)
        if len(data) != self.n_inputs:
            raise ValueError(
                f"expected {self.n_inputs} data qubits, got {len(data)}"
            )
        return build_deutsch_jozsa_circuit(data, target_qubit, self.oracle)

    @staticmethod
    def classify(bitstring: str) -> Promise:
        """Map an ideal data-register measurement to the promised function class."""
        return classify_measurement(bitstring)
