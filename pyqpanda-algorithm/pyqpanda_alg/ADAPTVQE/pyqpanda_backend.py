# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyQPanda state-vector adapter for the backend-agnostic ADAPT-VQE driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Mapping, Sequence

import numpy as np
from pyqpanda3.core import QCircuit, QProg
from pyqpanda3.quantum_info import StateVector

from .adapt_vqe import EnergyRequest, OperatorId


CircuitFactory = Callable[[Sequence, Hashable, float], QCircuit]
ReferenceCircuitFactory = Callable[[Sequence], QCircuit]


@dataclass(frozen=True)
class PauliTerm:
    """One Hamiltonian term represented as coefficient times a Pauli string."""

    coefficient: complex
    paulis: Mapping[int, str]

    def __post_init__(self) -> None:
        for qubit, pauli in self.paulis.items():
            if int(qubit) < 0:
                raise ValueError("Pauli-term qubit indices must be non-negative")
            if str(pauli).upper() not in {"I", "X", "Y", "Z"}:
                raise ValueError(f"unsupported Pauli operator: {pauli!r}")


def _apply_pauli(
    state: np.ndarray,
    num_qubits: int,
    paulis: Mapping[int, str],
) -> np.ndarray:
    transformed = np.zeros_like(state, dtype=complex)

    for source_index, amplitude in enumerate(state):
        target_index = source_index
        phase = 1.0 + 0.0j

        for qubit, raw_pauli in paulis.items():
            q = int(qubit)
            if q >= num_qubits:
                raise ValueError(
                    f"Pauli term references qubit {q}, but backend has {num_qubits}"
                )

            pauli = str(raw_pauli).upper()
            if pauli == "I":
                continue

            bit = (source_index >> q) & 1
            if pauli == "X":
                target_index ^= 1 << q
            elif pauli == "Y":
                target_index ^= 1 << q
                phase *= 1j if bit == 0 else -1j
            elif pauli == "Z" and bit:
                phase *= -1.0

        transformed[target_index] += phase * amplitude

    return transformed


def statevector_pauli_expectation(
    state: Sequence[complex],
    hamiltonian: Sequence[PauliTerm],
    *,
    num_qubits: int | None = None,
) -> float:
    """Return the real expectation value of a Pauli-sum Hamiltonian."""

    vector = np.asarray(state, dtype=complex).reshape(-1)
    if vector.size == 0 or vector.size & (vector.size - 1):
        raise ValueError("statevector length must be a non-zero power of two")

    inferred = int(np.log2(vector.size))
    if num_qubits is None:
        num_qubits = inferred
    if num_qubits != inferred:
        raise ValueError(
            f"statevector encodes {inferred} qubits, num_qubits={num_qubits}"
        )

    norm = float(np.vdot(vector, vector).real)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("statevector has invalid norm")

    energy = 0.0 + 0.0j
    for term in hamiltonian:
        applied = _apply_pauli(vector, num_qubits, term.paulis)
        energy += complex(term.coefficient) * np.vdot(vector, applied) / norm

    if abs(energy.imag) > 1.0e-9:
        raise ValueError(
            "Hamiltonian expectation has a non-negligible imaginary component: "
            f"{energy.imag}"
        )
    return float(energy.real)


class PyQPandaStatevectorEvaluator:
    """Evaluate ADAPT ansatz energies using PyQPanda3 StateVector."""

    def __init__(
        self,
        num_qubits: int,
        hamiltonian: Sequence[PauliTerm],
        operator_circuit_factory: CircuitFactory,
        *,
        reference_circuit_factory: ReferenceCircuitFactory | None = None,
    ) -> None:
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        if not hamiltonian:
            raise ValueError("hamiltonian must contain at least one Pauli term")

        self.num_qubits = int(num_qubits)
        self.hamiltonian = tuple(hamiltonian)
        self.operator_circuit_factory = operator_circuit_factory
        self.reference_circuit_factory = reference_circuit_factory

    def statevector(
        self,
        operators: Sequence[OperatorId],
        parameters: Sequence[float],
    ) -> np.ndarray:
        parameters = np.asarray(parameters, dtype=float).reshape(-1)
        if parameters.size != len(operators):
            raise ValueError(
                f"expected {len(operators)} parameters, got {parameters.size}"
            )

        program = QProg(self.num_qubits)
        qubits = program.qubits()
        circuit = QCircuit()

        if self.reference_circuit_factory is not None:
            circuit << self.reference_circuit_factory(qubits)

        for operator, theta in zip(operators, parameters):
            circuit << self.operator_circuit_factory(
                qubits, operator, float(theta)
            )

        return np.asarray(
            StateVector(self.num_qubits).evolve(circuit).ndarray(),
            dtype=complex,
        )

    def __call__(
        self,
        operators: Sequence[OperatorId],
        parameters: Sequence[float],
    ) -> float:
        state = self.statevector(operators, parameters)
        return statevector_pauli_expectation(
            state,
            self.hamiltonian,
            num_qubits=self.num_qubits,
        )

    def batch(self, requests: Sequence[EnergyRequest]) -> list[float]:
        """Batch-compatible API for local or drop-in cloud evaluator use."""

        return [self(operators, parameters) for operators, parameters in requests]
